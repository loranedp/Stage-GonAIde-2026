"""Pixel metrics shared by OpenUS validation and test evaluation."""

from __future__ import annotations

import math

import torch


METRIC_NAMES = ("iou", "dice", "precision", "recall", "diff_surface")


def metrics_by_class(pred, target, scale_cm=None, image_height_px=None, eps=1e-6):
    """Return per-class metrics using the conventions of UNet/YOLO.

    ``pred`` and ``target`` are binary tensors shaped ``[C, H, W]``. Surface
    differences are expressed in cm² and are unavailable when either mask is
    empty or the physical scale is missing.
    """

    pred = torch.as_tensor(pred).bool()
    target = torch.as_tensor(target).bool()
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"Expected matching [C,H,W] masks, got {pred.shape} and {target.shape}")

    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    pred_area = pred.float().sum((1, 2))
    true_area = target.float().sum((1, 2))

    iou = intersection / (union + eps)
    iou[union == 0] = 1.0
    dice = 2 * intersection / (pred_area + true_area + eps)
    dice[(pred_area == 0) & (true_area == 0)] = 1.0
    # Intentionally keep the historical UNet/YOLO convention: an empty
    # denominator produces zero for precision and recall.
    precision = intersection / (pred_area + eps)
    recall = intersection / (true_area + eps)

    diff_surface = torch.full_like(intersection, torch.nan)
    if scale_cm is not None and image_height_px is not None:
        scale = float(scale_cm)
        height = float(image_height_px)
        if math.isfinite(scale) and scale > 0 and math.isfinite(height) and height > 0:
            diff_surface = (true_area - pred_area).abs() * (scale / height) ** 2
            diff_surface[(true_area == 0) | (pred_area == 0)] = torch.nan

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "diff_surface": diff_surface,
    }


def finite_mean(values):
    """Mean of finite values, or NaN when none are available."""

    tensor = torch.as_tensor(values, dtype=torch.float64)
    finite = torch.isfinite(tensor)
    return float(tensor[finite].mean()) if finite.any() else float("nan")
