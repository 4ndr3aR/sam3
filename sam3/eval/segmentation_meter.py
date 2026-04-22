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
    ):
        """
        Args:
            iou_threshold: Minimum IoU for a prediction to count as a true positive.
            score_threshold: Minimum prediction score/objectness probability.
            mask_threshold: Threshold applied after converting mask logits to probabilities.
            name: Name prefix for metrics.
        """
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.name = name
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

        del find_metadatas, model  # unused, kept for interface compatibility

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
        pred_scores = self._extract_prediction_scores(preds=preds, pred_masks=pred_masks)

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

            pred_img_masks, img_scores = self._filter_predictions(pred_img_masks, img_scores)
            self._process_image(gt_masks=gt_img_masks, pred_masks=pred_img_masks, scores=img_scores)

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
    ) -> torch.Tensor:
        """Return scores as [num_images, num_queries] probabilities in [0, 1]."""
        candidate_keys = (
            "objectness",
            "objectness_logits",
            "objectness_scores",
            "objectness_ptr",
            "scores",
            "pred_scores",
        )

        scores: Optional[torch.Tensor] = None
        for key in candidate_keys:
            value = preds.get(key, None)
            if value is not None:
                scores = value
                break

        if scores is None:
            return torch.ones(
                pred_masks.shape[:2],
                dtype=pred_masks.dtype,
                device=pred_masks.device,
            )

        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores, dtype=pred_masks.dtype, device=pred_masks.device)
        else:
            scores = scores.to(device=pred_masks.device, dtype=pred_masks.dtype)

        while scores.dim() > 2 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)

        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        elif scores.dim() > 2:
            # Collapse any trailing feature dimension conservatively.
            scores = scores.reshape(scores.shape[0], scores.shape[1], -1).mean(-1)

        if scores.shape[:2] != pred_masks.shape[:2]:
            raise ValueError(
                "Prediction score shape mismatch: "
                f"scores={tuple(scores.shape)} pred_masks={tuple(pred_masks.shape)}"
            )

        return self._to_probabilities(scores)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred_masks.dim() != 3:
            raise ValueError(f"Expected pred_masks [N,H,W], got {tuple(pred_masks.shape)}")
        if scores.dim() != 1:
            raise ValueError(f"Expected scores [N], got {tuple(scores.shape)}")
        if pred_masks.shape[0] != scores.shape[0]:
            raise ValueError(
                f"Prediction count mismatch: pred_masks={tuple(pred_masks.shape)} scores={tuple(scores.shape)}"
            )

        keep = scores >= self.score_threshold
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
