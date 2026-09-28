"""Métriques de segmentation utilisées par l'application."""

import numpy as np


def compute_iou_per_class(
    pred_masks: dict, gt_masks: dict, class_names: list[str]
) -> dict[str, float]:
    """Calcule l'IoU de chaque classe demandée.

    Deux masques vides représentent un accord parfait et ont donc un IoU de 1.
    """
    ious = {}
    for class_name in class_names:
        pred_mask = pred_masks[class_name]
        gt_mask = gt_masks[class_name]
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        ious[class_name] = float(intersection / union) if union > 0 else 1.0
    return ious
