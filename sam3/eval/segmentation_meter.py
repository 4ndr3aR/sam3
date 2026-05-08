# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Segmentation evaluation meter for computing IoU, Dice, and other pixel-level metrics.

This meter evaluates segmentation predictions directly on pixel masks without
matching queries to targets. It computes standard segmentation metrics:
- Intersection over Union (IoU / Jaccard)
- Dice Coefficient (F1 score for segmentation)
- Precision, Recall
- Pixel Accuracy
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SegmentationMeter:
    """
    Computes segmentation metrics by comparing predicted masks to ground truth.

    This meter uses a greedy matching approach: for each ground truth mask,
    find the predicted mask with highest IoU above a threshold. Unmatched
    predictions are counted as false positives, unmatched ground truths as
    false negatives.

    Important implementation details:
    - Ground-truth masks are grouped per image using ``num_boxes``.
    - Predicted masks are treated as ``[num_images, max_queries, H, W]``.
      The leading dimension is *not* assumed to be the configured dataloader
      batch size; in some training/eval setups it is a flattened image/frame
      dimension.
    - Prediction masks are resized in a single batched interpolate call to the
      ground-truth resolution before any boolean operations.
    - Presence/objectness scores are converted to probabilities if needed and
      used to filter padded / very low-confidence prediction slots.

    Metrics computed:
    - mean_IoU: Mean Intersection over Union across all classes/images
    - mean_Dice: Mean Dice coefficient
    - precision: TP / (TP + FP)
    - recall: TP / (TP + FN)
    - pixel_accuracy: Correctly classified pixels / total pixels
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        name: str = "segmentation",
        score_threshold_buckets: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            iou_threshold: Minimum IoU for a prediction to count as a true positive.
            score_threshold: Minimum prediction score/objectness probability used for the
                backward-compatible headline metrics.
            mask_threshold: Threshold applied after converting mask logits to probabilities.
            name: Name prefix for metrics.
            score_threshold_buckets: Optional score thresholds for diagnostic
                precision/recall/F1 sweeps. Defaults to 0.1, 0.2, ..., 0.9.
        """
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.name = name
        if score_threshold_buckets is None:
            score_threshold_buckets = tuple(round(0.1 * i, 1) for i in range(1, 10))
        self.score_threshold_buckets = tuple(float(v) for v in score_threshold_buckets)
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self._total_intersection = 0.0
        self._total_union = 0.0
        self._total_predicted_positive = 0.0
        self._total_ground_truth_positive = 0.0
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0

        # Score-threshold sweep for detection-style counts. These counters are
        # separate from the headline metrics above so existing logging and
        # checkpointing code remains backward compatible.
        self._threshold_stats = {
            threshold: {"tp": 0, "fp": 0, "fn": 0, "kept": 0}
            for threshold in self.score_threshold_buckets
        }

        # Lightweight diagnostics to make score-threshold pathologies visible.
        # If score_source_missing is positive, the meter is assigning score=1
        # to every query, so every threshold bucket will be identical.
        self._score_source_missing = 0
        self._score_source_tensor = 0
        self._score_source_postprocessed = 0
        self._score_source_mask_confidence = 0
        self._score_count = 0
        self._score_sum = 0.0
        self._score_min = float("inf")
        self._score_max = float("-inf")

        # Per-image statistics for detailed reporting
        self._image_ious: List[float] = []
        self._image_dices: List[float] = []

        # Pixel-level accuracy
        self._correct_pixels = 0
        self._total_pixels = 0

    def update(
        self,
        find_stages: Any,
        find_metadatas: List[Dict],
        model: Any,
        batch: Any,
        key: str,
    ):
        """Update metrics with predictions from a batch.
        Args:
            find_stages: Model output(s) containing predictions.
            find_metadatas: Metadata for each sample in the batch.
            model: The model (for accessing configuration if needed).
            batch: Input batch containing ground truth (BatchedDatapoint object).
            key: Batch key (e.g., "coco100").
        """

        # find_metadatas and model are intentionally kept: when raw model outputs
        # do not expose scores, some training stacks provide COCO-style scores
        # only through a postprocessor attached to the model/evaluator.

        stage_targets = self._get_stage_targets(batch=batch, key=key)
        if stage_targets is None:
            return

        # Use stage 0 (first stage queries) for ground truth masks
        gt_masks = getattr(stage_targets, "segments", None)
        # Get num_boxes to know how many masks per image
        num_boxes = getattr(stage_targets, "num_boxes", None)
        if gt_masks is None or num_boxes is None:
            logging.warning(
                "SegmentationMeter(%s): missing ground truth masks or num_boxes for key=%s. "
                "gt_masks=%s num_boxes=%s",
                self.name,
                key,
                None if gt_masks is None else tuple(gt_masks.shape),
                num_boxes,
            )
            return

        # Organize masks by image
        masks_by_image = self._group_gt_masks(gt_masks=gt_masks, num_boxes=num_boxes)
        preds = find_stages[-1] if isinstance(find_stages, list) else find_stages
        if not isinstance(preds, dict):
            logging.warning(
                "SegmentationMeter(%s): expected dict predictions, got %s for key=%s",
                self.name,
                type(preds),
                key,
            )
            return

        pred_masks = preds.get("pred_masks", None)
        if pred_masks is None:
            logging.warning(
                "SegmentationMeter(%s): no pred_masks found for key=%s. Available keys: %s",
                self.name,
                key,
                list(preds.keys()),
            )
            return

        pred_masks = self._normalize_pred_masks(pred_masks)
        pred_scores = self._extract_prediction_scores(
            preds=preds,
            pred_masks=pred_masks,
            find_stages=find_stages,
            find_metadatas=find_metadatas,
            model=model,
            batch=batch,
            key=key,
        )

        num_gt_images = len(masks_by_image)
        num_pred_images = pred_masks.shape[0]
        num_score_images = pred_scores.shape[0]
        num_images = min(num_gt_images, num_pred_images, num_score_images)

        if not (num_gt_images == num_pred_images == num_score_images):
            logging.warning(
                "SegmentationMeter(%s): image count mismatch for key=%s: gt=%d pred=%d scores=%d. "
                "Processing first %d items.",
                self.name,
                key,
                num_gt_images,
                num_pred_images,
                num_score_images,
                num_images,
            )

        logging.debug(
            "SegmentationMeter(%s): grouped %d gt images; pred_masks=%s; pred_scores=%s",
            self.name,
            num_gt_images,
            tuple(pred_masks.shape),
            tuple(pred_scores.shape),
        )

        for image_idx in range(num_images):
            gt_img_masks = masks_by_image[image_idx]
            pred_img_masks = pred_masks[image_idx]  # [num_queries, H, W]
            img_scores = pred_scores[image_idx]     # [num_queries]

            if len(gt_img_masks) > 0:
                target_hw = tuple(int(v) for v in gt_img_masks[0].shape[-2:])
                pred_img_masks = self._resize_pred_masks(pred_img_masks, target_hw)
                if tuple(pred_img_masks.shape[-2:]) != target_hw:
                    raise RuntimeError(
                        f"Resize failed for image_idx={image_idx}: "
                        f"pred={tuple(pred_img_masks.shape)} gt_hw={target_hw}"
                    )

            self._update_score_diagnostics(img_scores)

            # Headline metrics keep the historical behavior controlled by
            # self.score_threshold.
            pred_img_masks_filtered, img_scores_filtered = self._filter_predictions(
                pred_img_masks,
                img_scores,
                score_threshold=self.score_threshold,
            )
            self._process_image(
                gt_masks=gt_img_masks,
                pred_masks=pred_img_masks_filtered,
                scores=img_scores_filtered,
            )

            # Diagnostic precision/recall/F1 sweep. We use the same resized masks
            # and the same greedy IoU matching, changing only the score threshold.
            for threshold in self.score_threshold_buckets:
                tp, fp, fn, kept = self._compute_match_counts(
                    gt_masks=gt_img_masks,
                    pred_masks=pred_img_masks,
                    scores=img_scores,
                    score_threshold=threshold,
                )
                self._threshold_stats[threshold]["tp"] += tp
                self._threshold_stats[threshold]["fp"] += fp
                self._threshold_stats[threshold]["fn"] += fn
                self._threshold_stats[threshold]["kept"] += kept

    def _get_stage_targets(self, batch: Any, key: str):
        # Get ground truth masks from batch.find_targets[stage].segments
        # batch is a BatchedDatapoint, masks are in find_targets
        find_targets = getattr(batch, "find_targets", None)
        if find_targets is None or len(find_targets) == 0:
            logging.warning(
                "SegmentationMeter(%s): no find_targets found for key=%s.",
                self.name,
                key,
            )
            return None
        return find_targets[0]

    def _group_gt_masks(
        self,
        gt_masks: torch.Tensor,
        num_boxes: Sequence[Any],
    ) -> List[List[torch.Tensor]]:
        masks_by_image: List[List[torch.Tensor]] = []
        offset = 0
        total_masks = int(gt_masks.shape[0])

        for image_idx, nb in enumerate(num_boxes):
            nb_int = int(nb.item()) if isinstance(nb, torch.Tensor) else int(nb)
            if nb_int < 0:
                raise ValueError(f"num_boxes[{image_idx}] must be >= 0, got {nb_int}")
            next_offset = min(offset + nb_int, total_masks)
            image_masks = [m for m in gt_masks[offset:next_offset] if m is not None]
            masks_by_image.append(image_masks)
            offset = next_offset

        if offset != total_masks:
            logging.warning(
                "SegmentationMeter(%s): gt mask count mismatch after grouping: consumed=%d total=%d. "
                "Any trailing masks are ignored.",
                self.name,
                offset,
                total_masks,
            )

        return masks_by_image

    def _normalize_pred_masks(self, pred_masks: torch.Tensor) -> torch.Tensor:
        """Return prediction masks as [num_images, num_queries, H, W]."""
        if pred_masks.dim() == 5:
            if pred_masks.shape[2] != 1:
                raise ValueError(
                    f"Expected pred_masks [B,N,1,H,W] when dim=5, got {tuple(pred_masks.shape)}"
                )
            pred_masks = pred_masks.squeeze(2)
        elif pred_masks.dim() == 4:
            pass
        elif pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported pred_masks shape: {tuple(pred_masks.shape)}")

        if pred_masks.dim() != 4:
            raise ValueError(f"Expected pred_masks to normalize to 4 dims, got {tuple(pred_masks.shape)}")

        return pred_masks.float()

    def _extract_prediction_scores(
        self,
        preds: Dict[str, Any],
        pred_masks: torch.Tensor,
        find_stages: Any = None,
        find_metadatas: Optional[List[Dict]] = None,
        model: Any = None,
        batch: Any = None,
        key: Optional[str] = None,
    ) -> torch.Tensor:
        """Return scores as [num_images, num_queries] probabilities in [0, 1].

        COCO evaluation does not compute confidence scores inside coco_eval.py:
        it receives postprocessed predictions and then reads prediction["scores"].
        During training, this meter sees raw find-stage outputs instead. We
        therefore try, in order:
          1. score/objectness tensors in the raw nested prediction dict;
          2. COCO-style postprocessed prediction containers, when available;
          3. a clearly marked mask-confidence fallback.

        The fallback is not COCO-equivalent, but it is much better than silently
        assigning score=1 to every query because it makes threshold sweeps
        informative and exposes the issue through diagnostics.
        """
        scores = self._find_score_tensor_in_nested_predictions(preds, pred_masks)
        if scores is not None:
            self._score_source_tensor += int(pred_masks.shape[0] * pred_masks.shape[1])
            return scores

        scores = self._try_postprocessed_scores(
            preds=preds,
            pred_masks=pred_masks,
            find_stages=find_stages,
            find_metadatas=find_metadatas,
            model=model,
            batch=batch,
            key=key,
        )
        if scores is not None:
            self._score_source_postprocessed += int(pred_masks.shape[0] * pred_masks.shape[1])
            return scores

        # Last-resort fallback: use the mask logit's own foreground confidence.
        # For each query, this is the maximum foreground probability over pixels.
        # This is NOT the same as the COCO detection score, but avoids the old
        # pathological behavior where every query received score=1.0.
        fallback_scores = self._mask_confidence_scores(pred_masks)
        n = int(pred_masks.shape[0] * pred_masks.shape[1])
        self._score_source_missing += n
        self._score_source_mask_confidence += n
        logging.warning(
            "SegmentationMeter(%s): no raw or postprocessed detection scores found for key=%s. "
            "Using mask-confidence fallback. Available top-level prediction keys: %s",
            self.name,
            key,
            sorted(str(k) for k in preds.keys()),
        )
        return fallback_scores

    def _find_score_tensor_in_nested_predictions(
        self,
        preds: Dict[str, Any],
        pred_masks: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Find a score tensor in raw predictions, including nested dicts/lists."""
        score_key_fragments = (
            "score",
            "objectness",
            "presence",
            "confidence",
            "conf",
            "prob",
            "logit",
        )
        bad_key_fragments = (
            "mask",
            "box",
            "bbox",
            "iou",
            "loss",
            "label",
            "class",
            "coord",
            "point",
            "pos",
            "embed",
            "feature",
        )

        candidates: List[Tuple[int, str, torch.Tensor]] = []

        def visit(obj: Any, path: str) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k_str = str(k)
                    visit(v, f"{path}.{k_str}" if path else k_str)
                return
            if isinstance(obj, (list, tuple)):
                # A COCO-style list of per-image dicts with variable-length
                # detections is handled separately in _try_postprocessed_scores.
                for i, v in enumerate(obj):
                    if isinstance(v, (dict, list, tuple)):
                        visit(v, f"{path}[{i}]")
                return
            if not isinstance(obj, torch.Tensor):
                return

            lower_path = path.lower()
            if not any(fragment in lower_path for fragment in score_key_fragments):
                return
            if any(fragment in lower_path for fragment in bad_key_fragments):
                return

            force_sigmoid = "logit" in lower_path
            normalized = self._normalize_score_tensor(obj, pred_masks, force_sigmoid=force_sigmoid)
            if normalized is not None:
                # Prefer explicit presence/objectness logits over generic scores.
                # In the raw model output used here, COCO-style detection scores
                # are usually produced later by a postprocessor. The closest raw
                # per-query confidence is the presence/objectness logit.
                priority = 0
                if lower_path.endswith("presence_logit_dec"):
                    priority -= 60
                elif lower_path.endswith("presence_logit"):
                    priority -= 55
                elif "presence" in lower_path and "logit" in lower_path:
                    priority -= 50
                elif "objectness" in lower_path:
                    priority -= 40
                elif "confidence" in lower_path or lower_path.endswith("conf"):
                    priority -= 20
                elif lower_path.endswith("scores") or lower_path.endswith("score"):
                    priority -= 10
                elif lower_path.endswith("pred_logits"):
                    priority -= 5
                priority += len(lower_path)
                candidates.append((priority, path, normalized))

        visit(preds, "")
        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        _, path, scores = candidates[0]
        logging.debug(
            "SegmentationMeter(%s): using prediction score tensor at '%s'.",
            self.name,
            path,
        )
        return scores

    def _normalize_score_tensor(
        self,
        scores: torch.Tensor,
        pred_masks: torch.Tensor,
        force_sigmoid: bool = False,
    ) -> Optional[torch.Tensor]:
        """Normalize a candidate score tensor to [num_images, num_queries]."""
        scores = scores.to(device=pred_masks.device, dtype=pred_masks.dtype)
        n_img, n_query = int(pred_masks.shape[0]), int(pred_masks.shape[1])

        while scores.dim() > 2 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)

        if scores.dim() == 0:
            return None
        if scores.dim() == 1:
            if scores.numel() == n_query and n_img == 1:
                scores = scores.unsqueeze(0)
            elif scores.numel() == n_img * n_query:
                scores = scores.reshape(n_img, n_query)
            else:
                return None
        elif scores.dim() == 2:
            if tuple(scores.shape) == (n_img, n_query):
                pass
            elif tuple(scores.shape) == (n_query, n_img):
                scores = scores.transpose(0, 1)
            elif scores.numel() == n_img * n_query:
                scores = scores.reshape(n_img, n_query)
            else:
                return None
        else:
            if scores.shape[0] == n_img and scores.shape[1] == n_query:
                # Collapse trailing class/feature dimensions. This is useful for
                # tensors such as [N, Q, C]; if logits are class-wise, max over C
                # is closer to detection confidence than mean over C.
                scores = scores.reshape(n_img, n_query, -1).max(-1).values
            else:
                return None

        if force_sigmoid:
            return torch.sigmoid(scores)
        return self._to_probabilities(scores)

    def _try_postprocessed_scores(
        self,
        preds: Dict[str, Any],
        pred_masks: torch.Tensor,
        find_stages: Any = None,
        find_metadatas: Optional[List[Dict]] = None,
        model: Any = None,
        batch: Any = None,
        key: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Try to recover COCO-style prediction["scores"] when present.

        This supports common already-postprocessed containers, but deliberately
        avoids guessing how to call an arbitrary postprocessor unless it exposes
        a compatible process_results method. COCO evaluation itself receives
        these postprocessed predictions from CocoEvaluator.postprocessor.
        """
        direct = self._scores_from_coco_style_container(preds, pred_masks)
        if direct is not None:
            return direct

        for container_key in ("predictions", "processed_results", "postprocessed", "coco_predictions", "results"):
            value = preds.get(container_key, None)
            direct = self._scores_from_coco_style_container(value, pred_masks)
            if direct is not None:
                return direct

        # Best-effort attempt for stacks where the model owns the same
        # postprocessor used by COCO evaluation. We only try very common calling
        # conventions and ignore failures to keep the meter non-invasive.
        postprocessors = []
        for attr in ("postprocessor", "post_processor", "coco_postprocessor", "evaluator_postprocessor"):
            pp = getattr(model, attr, None) if model is not None else None
            if pp is not None:
                postprocessors.append(pp)
        for pp in postprocessors:
            process_results = getattr(pp, "process_results", None)
            if process_results is None:
                continue
            for args in (
                (find_stages, find_metadatas, batch, key),
                (find_stages, find_metadatas, batch),
                (find_stages, batch),
                (preds, batch),
                (preds,),
            ):
                try:
                    processed = process_results(*args)
                except Exception:
                    continue
                direct = self._scores_from_coco_style_container(processed, pred_masks)
                if direct is not None:
                    return direct
        return None

    def _scores_from_coco_style_container(
        self,
        container: Any,
        pred_masks: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Extract scores from COCO-style dict/list predictions if aligned."""
        if container is None:
            return None
        n_img, n_query = int(pred_masks.shape[0]), int(pred_masks.shape[1])

        per_image: List[Any]
        if isinstance(container, dict):
            if "scores" in container:
                return self._normalize_score_tensor(container["scores"], pred_masks)
            # COCO postprocessors often return {image_id: {"scores": ...}}.
            values = list(container.values())
            if values and all(isinstance(v, dict) and "scores" in v for v in values):
                per_image = values
            else:
                return None
        elif isinstance(container, (list, tuple)) and all(isinstance(v, dict) and "scores" in v for v in container):
            per_image = list(container)
        else:
            return None

        if len(per_image) != n_img:
            return None

        rows = []
        for item in per_image:
            scores = item["scores"]
            if not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores, dtype=pred_masks.dtype, device=pred_masks.device)
            else:
                scores = scores.to(device=pred_masks.device, dtype=pred_masks.dtype)
            scores = scores.flatten()
            if scores.numel() != n_query:
                # Variable-length postprocessed detections cannot be aligned back
                # to raw query slots without the matching indices, so do not use
                # them for query-level threshold sweeps.
                return None
            rows.append(scores)

        return self._to_probabilities(torch.stack(rows, dim=0))

    def _mask_confidence_scores(self, pred_masks: torch.Tensor) -> torch.Tensor:
        """Last-resort per-query confidence from the mask logits themselves."""
        if pred_masks.numel() == 0:
            return torch.empty(pred_masks.shape[:2], dtype=pred_masks.dtype, device=pred_masks.device)
        probs = self._to_probabilities(pred_masks.float())
        return probs.flatten(2).max(dim=-1).values

    def _to_probabilities(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert logits-like tensors to probabilities when values are outside [0, 1]."""
        if tensor.numel() == 0:
            return tensor

        tensor_min = float(tensor.detach().min().item())
        tensor_max = float(tensor.detach().max().item())
        if tensor_min < 0.0 or tensor_max > 1.0:
            return torch.sigmoid(tensor)
        return tensor

    def _resize_pred_masks(
        self,
        pred_masks: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Resize [num_queries, H, W] predictions to target size in one batched op."""
        if pred_masks.dim() != 3:
            raise ValueError(f"Expected pred_masks [N,H,W], got {tuple(pred_masks.shape)}")

        if tuple(pred_masks.shape[-2:]) == tuple(target_hw):
            return pred_masks

        resized = F.interpolate(
            pred_masks.unsqueeze(1),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(1)

    def _filter_predictions(
        self,
        pred_masks: torch.Tensor,
        scores: torch.Tensor,
        score_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred_masks.dim() != 3:
            raise ValueError(f"Expected pred_masks [N,H,W], got {tuple(pred_masks.shape)}")
        if scores.dim() != 1:
            raise ValueError(f"Expected scores [N], got {tuple(scores.shape)}")
        if pred_masks.shape[0] != scores.shape[0]:
            raise ValueError(
                f"Prediction count mismatch: pred_masks={tuple(pred_masks.shape)} scores={tuple(scores.shape)}"
            )

        if score_threshold is None:
            score_threshold = self.score_threshold

        keep = scores >= score_threshold
        if keep.any():
            return pred_masks[keep], scores[keep]

        # Keep an empty tensor rather than padded slots; that makes FN/FP counting explicit.
        return pred_masks[:0], scores[:0]

    def _ensure_bool_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return mask
        mask = mask.float()
        mask = self._to_probabilities(mask)
        return mask >= self.mask_threshold

    def _update_score_diagnostics(self, scores: torch.Tensor) -> None:
        """Accumulate score-distribution diagnostics for sanity checks."""
        if scores.numel() == 0:
            return
        detached = scores.detach().float()
        self._score_count += int(detached.numel())
        self._score_sum += float(detached.sum().item())
        self._score_min = min(self._score_min, float(detached.min().item()))
        self._score_max = max(self._score_max, float(detached.max().item()))

    def _compute_match_counts(
        self,
        gt_masks: List[torch.Tensor],
        pred_masks: torch.Tensor,
        scores: torch.Tensor,
        score_threshold: float,
    ) -> Tuple[int, int, int, int]:
        """Return TP/FP/FN and number of kept predictions for one threshold.

        This uses the same greedy one-to-one IoU matching as _process_image, but
        does not update IoU, Dice, pixel-accuracy, or pixel-count accumulators.
        It is meant to sample the precision/recall/F1 trade-off induced by the
        prediction score threshold.
        """
        pred_masks, scores = self._filter_predictions(
            pred_masks,
            scores,
            score_threshold=score_threshold,
        )
        kept = int(pred_masks.shape[0])

        if len(gt_masks) == 0:
            return 0, kept, 0, kept

        if kept == 0:
            return 0, 0, len(gt_masks), 0

        pred_binary = self._ensure_bool_mask(pred_masks)
        if pred_binary.dim() != 3:
            raise ValueError(f"Expected binary pred masks [N,H,W], got {tuple(pred_binary.shape)}")

        sorted_indices = torch.argsort(scores, descending=True)
        matched_preds = set()
        tp = 0
        fn = 0

        for gt_mask in gt_masks:
            gt_bool = self._ensure_bool_mask(gt_mask)
            best_iou = -1.0
            best_pred_idx = -1

            for pred_idx_t in sorted_indices:
                pred_idx = int(pred_idx_t.item())
                if pred_idx in matched_preds:
                    continue

                pred_mask = pred_binary[pred_idx]
                if gt_bool.shape != pred_mask.shape:
                    raise RuntimeError(
                        "Shape mismatch after normalization/resizing: "
                        f"gt={tuple(gt_bool.shape)} pred={tuple(pred_mask.shape)}"
                    )

                intersection = (gt_bool & pred_mask).sum().float()
                union = (gt_bool | pred_mask).sum().float()
                iou = (intersection / union) if union > 0 else torch.tensor(0.0, device=intersection.device)
                iou_value = float(iou.item())

                if iou_value > best_iou:
                    best_iou = iou_value
                    best_pred_idx = pred_idx

            if best_iou >= self.iou_threshold and best_pred_idx >= 0:
                matched_preds.add(best_pred_idx)
                tp += 1
            else:
                fn += 1

        fp = kept - len(matched_preds)
        return tp, fp, fn, kept

    def _process_image(
        self,
        gt_masks: List[torch.Tensor],
        pred_masks: torch.Tensor,
        scores: torch.Tensor,
    ):
        """Process a single image's predictions."""
        if pred_masks.dim() != 3:
            raise ValueError(f"Expected pred_masks [N,H,W], got {tuple(pred_masks.shape)}")
        if scores.dim() != 1:
            raise ValueError(f"Expected scores [N], got {tuple(scores.shape)}")
        if pred_masks.shape[0] != scores.shape[0]:
            raise ValueError(
                f"Prediction count mismatch inside _process_image: pred_masks={tuple(pred_masks.shape)} scores={tuple(scores.shape)}"
            )

        if len(gt_masks) == 0:
            self._false_positives += int(pred_masks.shape[0])
            return

        if pred_masks.shape[0] == 0:
            self._false_negatives += len(gt_masks)
            return

        pred_binary = self._ensure_bool_mask(pred_masks)
        if pred_binary.dim() != 3:
            raise ValueError(f"Expected binary pred masks [N,H,W], got {tuple(pred_binary.shape)}")

        sorted_indices = torch.argsort(scores, descending=True)
        matched_preds = set()
        matched_gts = set()

        for gt_idx, gt_mask in enumerate(gt_masks):
            gt_bool = self._ensure_bool_mask(gt_mask)
            best_iou = -1.0
            best_pred_idx = -1

            for pred_idx_t in sorted_indices:
                pred_idx = int(pred_idx_t.item())
                if pred_idx in matched_preds:
                    continue

                pred_mask = pred_binary[pred_idx]
                if gt_bool.shape != pred_mask.shape:
                    raise RuntimeError(
                        "Shape mismatch after normalization/resizing: "
                        f"gt={tuple(gt_bool.shape)} pred={tuple(pred_mask.shape)}"
                    )

                intersection = (gt_bool & pred_mask).sum().float()
                union = (gt_bool | pred_mask).sum().float()
                iou = (intersection / union) if union > 0 else torch.tensor(0.0, device=intersection.device)
                iou_value = float(iou.item())

                if iou_value > best_iou:
                    best_iou = iou_value
                    best_pred_idx = pred_idx

            if best_iou >= self.iou_threshold and best_pred_idx >= 0:
                matched_preds.add(best_pred_idx)
                matched_gts.add(gt_idx)
                self._true_positives += 1
                self._image_ious.append(best_iou)
                dice = 2.0 * best_iou / (best_iou + 1.0) if best_iou > 0 else 0.0
                self._image_dices.append(dice)

                pred_mask = pred_binary[best_pred_idx]
                intersection = (gt_bool & pred_mask).sum().float().item()
                union = (gt_bool | pred_mask).sum().float().item()
                self._total_intersection += intersection
                self._total_union += union
                self._total_predicted_positive += float(pred_mask.sum().item())
                self._total_ground_truth_positive += float(gt_bool.sum().item())
            else:
                self._false_negatives += 1
                self._total_ground_truth_positive += float(gt_bool.sum().item())

        unmatched_preds = int(pred_masks.shape[0]) - len(matched_preds)
        self._false_positives += unmatched_preds

        for pred_idx in range(pred_binary.shape[0]):
            if pred_idx not in matched_preds:
                self._total_predicted_positive += float(pred_binary[pred_idx].sum().item())

        self._compute_pixel_accuracy(gt_masks, pred_binary, matched_preds)

    def _compute_pixel_accuracy(
        self,
        gt_masks: List[torch.Tensor],
        pred_binary: torch.Tensor,
        matched_preds: set,
    ):
        """Compute pixel-level accuracy using the union of GT masks and matched predictions."""
        if len(gt_masks) == 0:
            return

        if pred_binary.dim() != 3:
            raise ValueError(f"Expected pred_binary [N,H,W], got {tuple(pred_binary.shape)}")

        h, w = gt_masks[0].shape[-2:]
        gt_combined = torch.zeros(h, w, dtype=torch.bool, device=pred_binary.device)
        for gt_mask in gt_masks:
            gt_bool = self._ensure_bool_mask(gt_mask).to(device=pred_binary.device)
            if gt_bool.shape != gt_combined.shape:
                raise RuntimeError(
                    f"Ground-truth mask shape mismatch inside pixel accuracy: {tuple(gt_bool.shape)} vs {tuple(gt_combined.shape)}"
                )
            gt_combined |= gt_bool

        pred_combined = torch.zeros(h, w, dtype=torch.bool, device=pred_binary.device)
        for pred_idx in matched_preds:
            pred_mask = pred_binary[pred_idx]
            if pred_mask.shape != pred_combined.shape:
                raise RuntimeError(
                    f"Prediction mask shape mismatch inside pixel accuracy: {tuple(pred_mask.shape)} vs {tuple(pred_combined.shape)}"
                )
            pred_combined |= pred_mask

        correct = (gt_combined == pred_combined).sum().item()
        total = gt_combined.numel()
        self._correct_pixels += correct
        self._total_pixels += total

    def get_results(self) -> Dict[str, float]:
        """Compute and return all metrics."""
        results: Dict[str, float] = {}

        mean_iou = self._total_intersection / self._total_union if self._total_union > 0 else 0.0
        results[f"{self.name}/mean_IoU"] = mean_iou

        mean_dice = sum(self._image_dices) / len(self._image_dices) if self._image_dices else 0.0
        results[f"{self.name}/mean_Dice"] = mean_dice

        tp = self._true_positives
        fp = self._false_positives
        fn = self._false_negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        pixel_accuracy = self._correct_pixels / self._total_pixels if self._total_pixels > 0 else 0.0

        results[f"{self.name}/precision"] = precision
        results[f"{self.name}/recall"] = recall
        results[f"{self.name}/F1"] = f1
        results[f"{self.name}/pixel_accuracy"] = pixel_accuracy

        if self._image_ious:
            results[f"{self.name}/avg_IoU_per_match"] = sum(self._image_ious) / len(self._image_ious)
            results[f"{self.name}/min_IoU"] = min(self._image_ious)
            results[f"{self.name}/max_IoU"] = max(self._image_ious)

        results[f"{self.name}/true_positives"] = float(tp)
        results[f"{self.name}/false_positives"] = float(fp)
        results[f"{self.name}/false_negatives"] = float(fn)

        best_threshold = None
        best_f1 = -1.0
        for threshold in self.score_threshold_buckets:
            stats = self._threshold_stats[threshold]
            bucket_tp = stats["tp"]
            bucket_fp = stats["fp"]
            bucket_fn = stats["fn"]
            bucket_precision = bucket_tp / (bucket_tp + bucket_fp) if (bucket_tp + bucket_fp) > 0 else 0.0
            bucket_recall = bucket_tp / (bucket_tp + bucket_fn) if (bucket_tp + bucket_fn) > 0 else 0.0
            bucket_f1 = (
                2 * bucket_precision * bucket_recall / (bucket_precision + bucket_recall)
                if (bucket_precision + bucket_recall) > 0
                else 0.0
            )
            key_prefix = f"{self.name}/threshold_{threshold:.1f}"
            results[f"{key_prefix}/precision"] = bucket_precision
            results[f"{key_prefix}/recall"] = bucket_recall
            results[f"{key_prefix}/F1"] = bucket_f1
            results[f"{key_prefix}/true_positives"] = float(bucket_tp)
            results[f"{key_prefix}/false_positives"] = float(bucket_fp)
            results[f"{key_prefix}/false_negatives"] = float(bucket_fn)
            results[f"{key_prefix}/kept_predictions"] = float(stats["kept"])
            if bucket_f1 > best_f1:
                best_f1 = bucket_f1
                best_threshold = threshold

        if best_threshold is not None:
            results[f"{self.name}/best_threshold"] = float(best_threshold)
            results[f"{self.name}/best_threshold_F1"] = float(best_f1)

        if self._score_count > 0:
            results[f"{self.name}/score_min"] = float(self._score_min)
            results[f"{self.name}/score_max"] = float(self._score_max)
            results[f"{self.name}/score_mean"] = float(self._score_sum / self._score_count)
        results[f"{self.name}/score_count"] = float(self._score_count)
        results[f"{self.name}/score_source_missing"] = float(self._score_source_missing)
        results[f"{self.name}/score_source_tensor"] = float(self._score_source_tensor)
        results[f"{self.name}/score_source_postprocessed"] = float(self._score_source_postprocessed)
        results[f"{self.name}/score_source_mask_confidence"] = float(self._score_source_mask_confidence)
        results[f"{self.name}/score_threshold"] = float(self.score_threshold)
        results[f"{self.name}/mask_threshold"] = float(self.mask_threshold)

        return results

    def compute_synced(self) -> Dict[str, float]:
        """Synchronize metrics across processes and compute summary."""
        return self.get_results()

    def compute(self) -> Dict[str, float]:
        """Compute without synchronization."""
        return self.get_results()

    def is_better(self, new_value: float, old_value: float) -> bool:
        """Check if new value is better than old value. Higher is better."""
        return new_value > old_value

    def log_results(self, prefix: str = ""):
        """Print results to logging."""
        results = self.get_results()
        log_msg = f"{prefix}Segmentation Metrics:"
        for key, value in results.items():
            if isinstance(value, float):
                log_msg += f" {key}={value:.4f}"
            else:
                log_msg += f" {key}={value}"
        logging.info(log_msg)


class SimpleSegmentationMeter:
    """
    Simplified segmentation meter that computes IoU directly without query matching.

    This meter compares the union of all predicted masks to the union of all
    ground truth masks, computing standard segmentation metrics. This is useful
    for evaluating the overall segmentation quality without worrying about
    instance-level matching.
    """

    def __init__(self, name: str = "simple_segmentation"):
        self.name = name
        self._reset()

    def _reset(self):
        self._total_intersection = 0.0
        self._total_union = 0.0
        self._correct_pixels = 0
        self._total_pixels = 0
        self._image_count = 0

    def reset(self):
        self._reset()

    def update(
        self,
        find_stages: Any,
        find_metadatas: List[Dict],
        model: Any,
        batch: Any,
        key: str,
    ):
        del find_metadatas, model, key

        find_targets = getattr(batch, "find_targets", None)
        if find_targets is None or len(find_targets) == 0:
            return

        stage_targets = find_targets[0]
        gt_masks = getattr(stage_targets, "segments", None)
        num_boxes = getattr(stage_targets, "num_boxes", None)
        if gt_masks is None or num_boxes is None:
            return

        masks_by_image = []
        offset = 0
        for nb in num_boxes:
            nb_int = int(nb.item()) if isinstance(nb, torch.Tensor) else int(nb)
            next_offset = min(offset + nb_int, int(gt_masks.shape[0]))
            image_masks = [m for m in gt_masks[offset:next_offset] if m is not None]
            masks_by_image.append(image_masks)
            offset = next_offset

        preds = find_stages[-1] if isinstance(find_stages, list) else find_stages
        if not isinstance(preds, dict):
            return

        pred_masks = preds.get("pred_masks", None)
        if pred_masks is None:
            return

        if pred_masks.dim() == 5 and pred_masks.shape[2] == 1:
            pred_masks = pred_masks.squeeze(2)
        elif pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(0)

        if pred_masks.dim() != 4:
            return

        presence_scores = preds.get("objectness_ptr", None)
        if presence_scores is None:
            presence_scores = preds.get("scores", None)

        batch_size = min(len(masks_by_image), pred_masks.shape[0])
        for batch_idx in range(batch_size):
            gt_img_masks = masks_by_image[batch_idx]
            pred_img_masks = pred_masks[batch_idx]

            if len(gt_img_masks) > 0:
                gt_h, gt_w = gt_img_masks[0].shape[-2:]
                if tuple(pred_img_masks.shape[-2:]) != (gt_h, gt_w):
                    pred_img_masks = F.interpolate(
                        pred_img_masks.unsqueeze(1).float(),
                        size=(gt_h, gt_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

            if presence_scores is not None:
                img_scores = presence_scores[batch_idx]
                while img_scores.dim() > 1 and img_scores.shape[-1] == 1:
                    img_scores = img_scores.squeeze(-1)
                if img_scores.dim() > 1:
                    img_scores = img_scores.reshape(img_scores.shape[0], -1).mean(-1)
                if img_scores.numel() == pred_img_masks.shape[0]:
                    if img_scores.min() < 0 or img_scores.max() > 1:
                        img_scores = torch.sigmoid(img_scores)
                    valid_preds = img_scores > 0.5
                    pred_img_masks = pred_img_masks[valid_preds]

            self._process_image_simple(gt_img_masks, pred_img_masks)

    def _process_image_simple(self, gt_masks: List[torch.Tensor], pred_masks: torch.Tensor):
        if len(gt_masks) == 0 or len(pred_masks) == 0:
            return

        h, w = gt_masks[0].shape[-2:]

        gt_union = torch.zeros(h, w, dtype=torch.bool, device=pred_masks.device)
        for gt_mask in gt_masks:
            gt_bool = gt_mask if gt_mask.dtype == torch.bool else gt_mask > 0.5
            gt_union |= gt_bool

        pred_binary = pred_masks > 0.5
        pred_union = torch.zeros(h, w, dtype=torch.bool, device=pred_masks.device)
        for pred_mask in pred_binary:
            pred_union |= pred_mask

        intersection = (gt_union & pred_union).sum().float()
        union = (gt_union | pred_union).sum().float()

        self._total_intersection += intersection.item()
        self._total_union += union.item()

        correct = (gt_union == pred_union).sum().item()
        self._correct_pixels += correct
        self._total_pixels += gt_union.numel()
        self._image_count += 1

    def get_results(self) -> Dict[str, float]:
        results = {}

        if self._total_union > 0:
            results[f"{self.name}/IoU"] = self._total_intersection / self._total_union
        else:
            results[f"{self.name}/IoU"] = 0.0

        if self._total_pixels > 0:
            results[f"{self.name}/pixel_accuracy"] = self._correct_pixels / self._total_pixels
        else:
            results[f"{self.name}/pixel_accuracy"] = 0.0

        results[f"{self.name}/images_processed"] = float(self._image_count)
        return results

    def compute_synced(self) -> Dict[str, float]:
        return self.get_results()

    def compute(self) -> Dict[str, float]:
        return self.get_results()

    def is_better(self, new_value: float, old_value: float) -> bool:
        return new_value > old_value

    def log_results(self, prefix: str = ""):
        results = self.get_results()
        log_msg = f"{prefix}Simple Segmentation Metrics:"
        for key, value in results.items():
            if isinstance(value, float):
                log_msg += f" {key}={value:.4f}"
            else:
                log_msg += f" {key}={value}"
        logging.info(log_msg)
