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
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np


class SegmentationMeter:
    """
    Computes segmentation metrics by comparing predicted masks to ground truth.

    This meter uses a greedy matching approach: for each ground truth mask,
    find the predicted mask with highest IoU above a threshold. Unmatched
    predictions are counted as false positives, unmatched ground truths as
    false negatives.

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
        name: str = "segmentation",
    ):
        """
        Args:
            iou_threshold: Minimum IoU for a prediction to count as a true positive.
            name: Name prefix for metrics.
        """
        self.iou_threshold = iou_threshold
        self.name = name

        # Accumulators
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

    def reset(self):
        """Reset all accumulators."""
        self._total_intersection = 0.0
        self._total_union = 0.0
        self._total_predicted_positive = 0.0
        self._total_ground_truth_positive = 0.0
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._image_ious = []
        self._image_dices = []
        self._correct_pixels = 0
        self._total_pixels = 0

    def update(
        self,
        find_stages: Any,
        find_metadatas: List[Dict],
        model: Any,
        batch: Dict,
        key: str,
    ):
        """
        Update metrics with predictions from a batch.

        Args:
            find_stages: Model output(s) containing predictions.
            find_metadatas: Metadata for each sample in the batch.
            model: The model (for accessing configuration if needed).
            batch: Input batch containing ground truth.
            key: Batch key (e.g., "coco100").
        """
        # Get ground truth masks from batch
        gt_masks = batch.get("find_masks", None)
        if gt_masks is None:
            logging.warning(f"No ground truth masks found in batch for key={key}")
            return

        # Extract predictions from model output
        # find_stages can be a single output or list of outputs (for aux losses)
        if isinstance(find_stages, list):
            # Use the last stage (final predictions)
            preds = find_stages[-1]
        else:
            preds = find_stages

        # Get predicted masks - typically shape [B, N, H, W] or [B, N, 1, H, W]
        pred_masks = preds.get("masks", None)
        if pred_masks is None:
            logging.warning(f"No predicted masks found in model output")
            return

        # Get presence scores to filter out empty predictions
        presence_scores = preds.get("objectness_ptr", None)
        if presence_scores is None:
            # Try alternative key
            presence_scores = preds.get("scores", None)

        # Process each image in the batch
        for batch_idx in range(len(gt_masks)):
            # Get ground truth masks for this image
            gt_img_masks = gt_masks[batch_idx]  # List of masks or tensor

            # Get predicted masks for this image
            pred_img_masks = pred_masks[batch_idx]  # [N, H, W] or [N, 1, H, W]

            # Get scores for this image
            if presence_scores is not None:
                img_scores = presence_scores[batch_idx]  # [N]
            else:
                img_scores = torch.ones(pred_img_masks.shape[0])

            self._process_image(gt_img_masks, pred_img_masks, img_scores)

    def _process_image(
        self,
        gt_masks: List[torch.Tensor],
        pred_masks: torch.Tensor,
        scores: torch.Tensor,
    ):
        """Process a single image's predictions."""
        if len(gt_masks) == 0:
            # No ground truth - all predictions are false positives
            self._false_positives += len(pred_masks)
            return

        # Ensure pred_masks is [N, H, W]
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)

        # Convert to binary (threshold at 0.5)
        pred_binary = pred_masks > 0.5

        # Sort predictions by score (descending)
        sorted_indices = torch.argsort(scores, descending=True)

        matched_preds = set()
        matched_gts = set()

        for gt_idx, gt_mask in enumerate(gt_masks):
            best_iou = -1.0
            best_pred_idx = -1

            # Find best matching prediction
            for pred_idx in sorted_indices:
                if pred_idx in matched_preds:
                    continue

                # Compute IoU
                pred_mask = pred_binary[pred_idx]

                # Handle different dtypes
                if gt_mask.dtype == torch.bool:
                    gt_bool = gt_mask
                else:
                    gt_bool = gt_mask > 0.5

                intersection = (gt_bool & pred_bool[pred_idx]).sum().float()
                union = (gt_bool | pred_bool[pred_idx]).sum().float()

                if union > 0:
                    iou = intersection / union
                else:
                    iou = 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx.item()

            # Check if best match meets threshold
            if best_iou >= self.iou_threshold and best_pred_idx >= 0:
                matched_preds.add(best_pred_idx)
                matched_gts.add(gt_idx)
                self._true_positives += 1
                self._image_ious.append(best_iou)
                dice = 2 * best_iou / (best_iou + 1) if best_iou > 0 else 0.0
                self._image_dices.append(dice)

                # Accumulate for mean IoU/Dice
                gt_mask = gt_masks[gt_idx]
                if gt_mask.dtype != torch.bool:
                    gt_bool = gt_mask > 0.5
                else:
                    gt_bool = gt_mask
                pred_mask = pred_binary[best_pred_idx]

                intersection = (gt_bool & pred_mask).sum().float().item()
                union = (gt_bool | pred_mask).sum().float().item()
                self._total_intersection += intersection
                self._total_union += union
            else:
                # No matching prediction - false negative
                self._false_negatives += 1

        # Unmatched predictions are false positives
        unmatched_preds = len(pred_masks) - len(matched_preds)
        self._false_positives += unmatched_preds

        # Compute pixel-level accuracy
        self._compute_pixel_accuracy(gt_masks, pred_masks, matched_preds, matched_gts)

    def _compute_pixel_accuracy(
        self,
        gt_masks: List[torch.Tensor],
        pred_masks: torch.Tensor,
        matched_preds: set,
        matched_gts: set,
    ):
        """Compute pixel-level accuracy."""
        # Create combined ground truth mask (union of all GT masks)
        if len(gt_masks) == 0:
            return

        h, w = pred_masks.shape[1], pred_masks.shape[2]
        gt_combined = torch.zeros(h, w, dtype=torch.bool, device=pred_masks.device)

        for gt_idx, gt_mask in enumerate(gt_masks):
            if gt_mask.dtype != torch.bool:
                gt_bool = gt_mask > 0.5
            else:
                gt_bool = gt_mask
            gt_combined |= gt_bool

        # Create combined prediction mask (only from matched predictions)
        pred_combined = torch.zeros(h, w, dtype=torch.bool, device=pred_masks.device)
        for pred_idx in matched_preds:
            pred_combined |= pred_masks[pred_idx] > 0.5

        # Compute accuracy
        correct = ((gt_combined == pred_combined)).sum().item()
        total = gt_combined.numel()

        self._correct_pixels += correct
        self._total_pixels += total

    def get_results(self) -> Dict[str, float]:
        """Compute and return all metrics."""
        results = {}

        # Mean IoU
        if self._total_union > 0:
            mean_iou = self._total_intersection / self._total_union
        else:
            mean_iou = 0.0
        results[f"{self.name}/mean_IoU"] = mean_iou

        # Mean Dice
        if len(self._image_dices) > 0:
            mean_dice = sum(self._image_dices) / len(self._image_dices)
        else:
            mean_dice = 0.0
        results[f"{self.name}/mean_Dice"] = mean_dice

        # Precision and Recall
        tp = self._true_positives
        fp = self._false_positives
        fn = self._false_negatives

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        results[f"{self.name}/precision"] = precision

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        results[f"{self.name}/recall"] = recall

        # F1 Score (harmonic mean of precision and recall)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        results[f"{self.name}/F1"] = f1

        # Pixel Accuracy
        if self._total_pixels > 0:
            pixel_accuracy = self._correct_pixels / self._total_pixels
        else:
            pixel_accuracy = 0.0
        results[f"{self.name}/pixel_accuracy"] = pixel_accuracy

        # Per-image statistics
        if len(self._image_ious) > 0:
            results[f"{self.name}/avg_IoU_per_match"] = sum(self._image_ious) / len(self._image_ious)
            results[f"{self.name}/min_IoU"] = min(self._image_ious)
            results[f"{self.name}/max_IoU"] = max(self._image_ious)

        # Counts
        results[f"{self.name}/true_positives"] = float(tp)
        results[f"{self.name}/false_positives"] = float(fp)
        results[f"{self.name}/false_negatives"] = float(fn)

        return results

    def compute_synced(self) -> Dict[str, float]:
        """
        Synchronize metrics across processes and compute summary.

        Returns:
            Dictionary of computed metrics.
        """
        # For simplicity, just return local results
        # In a distributed setting, you would gather results across processes
        return self.get_results()

    def compute(self) -> Dict[str, float]:
        """
        Compute without synchronization.

        Returns:
            Dictionary of computed metrics.
        """
        return self.get_results()

    def is_better(self, new_value: float, old_value: float) -> bool:
        """
        Check if new value is better than old value.
        Higher is better for this meter.
        """
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
        batch: Dict,
        key: str,
    ):
        gt_masks = batch.get("find_masks", None)
        if gt_masks is None:
            return

        if isinstance(find_stages, list):
            preds = find_stages[-1]
        else:
            preds = find_stages

        pred_masks = preds.get("masks", None)
        if pred_masks is None:
            return

        # Get presence scores
        presence_scores = preds.get("objectness_ptr", None)
        if presence_scores is None:
            presence_scores = preds.get("scores", None)

        for batch_idx in range(len(gt_masks)):
            gt_img_masks = gt_masks[batch_idx]
            pred_img_masks = pred_masks[batch_idx]

            if pred_img_masks.dim() == 4 and pred_img_masks.shape[1] == 1:
                pred_img_masks = pred_img_masks.squeeze(1)

            # Filter predictions by presence score (threshold 0.5)
            if presence_scores is not None:
                valid_preds = presence_scores[batch_idx] > 0.5
                if valid_preds.sum() == 0:
                    valid_preds = torch.ones_like(valid_preds, dtype=torch.bool)
                pred_img_masks = pred_img_masks[valid_preds]

            self._process_image_simple(gt_img_masks, pred_img_masks)

    def _process_image_simple(self, gt_masks: List[torch.Tensor], pred_masks: torch.Tensor):
        if len(gt_masks) == 0 or len(pred_masks) == 0:
            return

        h, w = pred_masks.shape[1], pred_masks.shape[2]

        # Union of all ground truth masks
        gt_union = torch.zeros(h, w, dtype=torch.bool, device=pred_masks.device)
        for gt_mask in gt_masks:
            if gt_mask.dtype != torch.bool:
                gt_bool = gt_mask > 0.5
            else:
                gt_bool = gt_mask
            gt_union |= gt_bool

        # Union of all predicted masks
        pred_binary = pred_masks > 0.5
        pred_union = torch.zeros(h, w, dtype=torch.bool, device=pred_masks.device)
        for pred_mask in pred_binary:
            pred_union |= pred_mask

        # Compute IoU
        intersection = (gt_union & pred_union).sum().float()
        union = (gt_union | pred_union).sum().float()

        self._total_intersection += intersection.item()
        self._total_union += union.item()

        # Pixel accuracy
        correct = ((gt_union == pred_union)).sum().item()
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
        """Synchronize metrics across processes and compute summary."""
        return self.get_results()

    def compute(self) -> Dict[str, float]:
        """Compute without synchronization."""
        return self.get_results()

    def is_better(self, new_value: float, old_value: float) -> bool:
        """Check if new value is better than old value. Higher is better."""
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
