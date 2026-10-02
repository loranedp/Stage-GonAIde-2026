"""Métriques communes aux pipelines de segmentation YOLO et U-Net."""

from __future__ import annotations

import numpy as np
import torch

from utils.post_traitement import split_eggs
from utils.unet_spatial import CROPPED_IMAGE_SIZE, MODEL_IMAGE_SIZE, unpad_array


def _median_instance_area_px(instance_labels):
    """Surface médiane en pixels des instances d'œufs, par image."""
    if instance_labels.ndim == 2:
        instance_labels = instance_labels.unsqueeze(0)
    if instance_labels.ndim != 3:
        raise ValueError("Les labels d'instances d'œufs doivent avoir la forme (B, H, W).")

    median_areas = []
    for labels in instance_labels:
        label_ids, counts = torch.unique(labels, return_counts=True)
        areas = counts[label_ids > 0].to(dtype=torch.float32)
        if areas.numel() == 0:
            median_areas.append(torch.zeros((), device=labels.device))
            continue
        median_areas.append(torch.quantile(areas, 0.5))
    return torch.stack(median_areas)


def _median_areas_px(areas_by_image, reference):
    """Médiane des surfaces indépendantes ; zéro sans instance."""
    if len(areas_by_image) != reference.shape[0]:
        raise ValueError("Une liste de surfaces est requise par image.")
    medians = []
    for areas in areas_by_image:
        if areas is None:
            raise ValueError("Les surfaces individuelles ne sont pas disponibles.")
        areas = torch.as_tensor(areas, device=reference.device, dtype=torch.float32)
        if areas.ndim != 1 or not torch.isfinite(areas).all() or (areas < 0).any():
            raise ValueError("Les surfaces doivent être un vecteur de valeurs finies positives ou nulles.")
        areas = areas[areas > 0]
        medians.append(
            torch.quantile(areas, 0.5)
            if areas.numel()
            else reference.new_zeros((), dtype=torch.float32)
        )
    return torch.stack(medians)


def metrics_from_counts(intersection, union, predicted_area, true_area):
    """Calcule IoU, Dice, précision et rappel depuis des comptes de pixels."""
    intersection = np.asarray(intersection, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    predicted_area = np.asarray(predicted_area, dtype=np.float64)
    true_area = np.asarray(true_area, dtype=np.float64)
    total_area = predicted_area + true_area
    return {
        "iou": np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0),
        "dice": np.divide(2 * intersection, total_area, out=np.ones_like(intersection), where=total_area > 0),
        "precision": np.divide(
            intersection, predicted_area,
            out=(true_area == 0).astype(np.float64), where=predicted_area > 0,
        ),
        "recall": np.divide(
            intersection, true_area,
            out=(predicted_area == 0).astype(np.float64), where=true_area > 0,
        ),
    }


def compute_iou_per_class(pred_masks, gt_masks, class_names):
    """Calcule l'IoU de chaque classe pour les masques de l'application."""
    intersections = []
    unions = []
    predicted_areas = []
    true_areas = []
    for class_name in class_names:
        pred_mask = pred_masks[class_name]
        gt_mask = gt_masks[class_name]
        intersections.append(np.logical_and(pred_mask, gt_mask).sum())
        unions.append(np.logical_or(pred_mask, gt_mask).sum())
        predicted_areas.append(pred_mask.sum())
        true_areas.append(gt_mask.sum())
    ious = metrics_from_counts(
        intersections, unions, predicted_areas, true_areas
    )["iou"]
    return dict(zip(class_names, map(float, ious)))


def metrics_by_class(
    true,
    preds,
    echelle,
    dataset_name,
    egg_instance_labels=None,
    egg_annotation_areas_px=None,
    image_height_px=None,
    egg_prediction_areas_px=None,
):
    """Calcule les métriques par image et classe dans un repère commun.

    Les tenseurs sont attendus au format ``[B, C, H, W]``. ``echelle`` est la
    hauteur physique de l'image en centimètres et ``image_height_px`` sa
    hauteur avant un éventuel padding de modèle.
    """
    intersection = (preds & true).float().sum((2, 3))
    union = (preds | true).float().sum((2, 3))

    iou_per_img = intersection / (union + 1e-6)
    iou_per_img[union == 0] = 1.0

    pred_area = preds.float().sum((2, 3))
    true_area = true.float().sum((2, 3))
    dice_per_img = (2 * intersection) / (pred_area + true_area + 1e-6)
    both_empty = (pred_area == 0) & (true_area == 0)
    dice_per_img[both_empty] = 1.0

    false_positive = pred_area - intersection
    false_negative = true_area - intersection
    precision_per_img = intersection / (intersection + false_positive + 1e-6)
    recall_per_img = intersection / (intersection + false_negative + 1e-6)

    if dataset_name != "oeufs":
        surface_true = true
        surface_preds = preds
        height_px = torch.as_tensor(
            true.shape[2] if image_height_px is None else image_height_px,
            device=true.device,
            dtype=torch.float32,
        ).reshape(-1)
        if (
            height_px.numel() not in (1, true.shape[0])
            or not torch.isfinite(height_px).all()
            or (height_px <= 0).any()
        ):
            raise ValueError("La hauteur doit être strictement positive, scalaire ou par image.")

        is_padded_unet = (
            image_height_px is not None
            and true.shape[-2:] == MODEL_IMAGE_SIZE[::-1]
            and torch.all(height_px == CROPPED_IMAGE_SIZE[1])
        )
        if is_padded_unet:
            surface_true = unpad_array(true)
            surface_preds = unpad_array(preds)

        true_surf_pixels = surface_true.float().sum((2, 3))
        pred_surf_pixels = surface_preds.float().sum((2, 3))
        diff_surf_pixels = (true_surf_pixels - pred_surf_pixels).abs()
        echelle = echelle.to(
            device=diff_surf_pixels.device, dtype=diff_surf_pixels.dtype
        ).reshape(-1)
        if (
            echelle.numel() not in (1, true.shape[0])
            or not torch.isfinite(echelle).all()
            or (echelle <= 0).any()
        ):
            raise ValueError("L'échelle doit être strictement positive, scalaire ou par image.")

        diff_surface = diff_surf_pixels * (echelle / height_px).reshape(-1, 1) ** 2
        missing_surface = (true_surf_pixels == 0) | (pred_surf_pixels == 0)
        diff_surface[missing_surface] = torch.nan
    else:
        if egg_annotation_areas_px is None:
            true_instance_labels = split_eggs(
                true,
                min_distance=8,
                min_area=100,
                ouverture_size=3,
                ouverture_iterations=1,
            )
            true_median_area = _median_instance_area_px(true_instance_labels)
        else:
            true_median_area = _median_areas_px(egg_annotation_areas_px, true)

        if egg_prediction_areas_px is not None:
            predicted_median_area = _median_areas_px(egg_prediction_areas_px, true)
        else:
            prediction_mask = preds
            if image_height_px is not None and preds.shape[-2:] == MODEL_IMAGE_SIZE[::-1]:
                prediction_mask = unpad_array(preds)
            if egg_instance_labels is None:
                egg_instance_labels = split_eggs(
                    prediction_mask,
                    min_distance=8,
                    min_area=100,
                    ouverture_size=3,
                    ouverture_iterations=1,
                )
            else:
                egg_instance_labels = torch.as_tensor(egg_instance_labels, device=true.device)
                if image_height_px is not None and egg_instance_labels.shape[-2:] == MODEL_IMAGE_SIZE[::-1]:
                    egg_instance_labels = unpad_array(egg_instance_labels)
            predicted_median_area = _median_instance_area_px(egg_instance_labels)

        if predicted_median_area.shape != true_median_area.shape:
            raise ValueError("Une carte d'instances prédites est requise par image.")
        height_px = torch.as_tensor(
            true.shape[2] if image_height_px is None else image_height_px,
            device=true.device,
            dtype=torch.float32,
        ).reshape(-1)
        if (
            height_px.numel() not in (1, true.shape[0])
            or not torch.isfinite(height_px).all()
            or (height_px <= 0).any()
        ):
            raise ValueError("La hauteur doit être strictement positive, scalaire ou par image.")
        echelle = echelle.to(device=true.device, dtype=torch.float32).reshape(-1)
        if (
            echelle.numel() not in (1, true.shape[0])
            or not torch.isfinite(echelle).all()
            or (echelle <= 0).any()
        ):
            raise ValueError("L'échelle doit être strictement positive, scalaire ou par image.")

        diff_surface = (true_median_area - predicted_median_area).abs().unsqueeze(1)
        diff_surface *= ((echelle * 10) / height_px).unsqueeze(1) ** 2
        missing_surface = (true_median_area == 0) | (predicted_median_area == 0)
        diff_surface[missing_surface.unsqueeze(1)] = torch.nan

    return (
        intersection,
        union,
        iou_per_img,
        dice_per_img,
        precision_per_img,
        recall_per_img,
        diff_surface,
    )


__all__ = [
    "_median_areas_px",
    "_median_instance_area_px",
    "compute_iou_per_class",
    "metrics_by_class",
    "metrics_from_counts",
]
