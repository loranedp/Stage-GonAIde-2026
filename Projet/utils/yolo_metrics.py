"""Helpers used to evaluate polygon-based YOLO26 instance segmentations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from utils.segmentation_metrics import metrics_from_counts


def _read_polygons(label_path: str | Path, image_shape: tuple[int, int]):
    """Read YOLO polygons and rasterise each one at the original image size."""

    height, width = image_shape
    polygons = []
    path = Path(label_path)
    if not path.exists():
        return polygons

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise ValueError(f"Polygone YOLO invalide dans {path}:{line_number}")

        class_id = int(parts[0])
        points = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points = np.rint(points).astype(np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        polygons.append((class_id, mask, int(mask.sum())))

    return polygons


def load_yolo_polygon_areas(
    label_path: str | Path,
    image_shape: tuple[int, int],
    *,
    class_ids: Iterable[int] = (0,),
    ignore_class_ids: Iterable[int] = (3,),
) -> list[int]:
    """Une surface rasterisée par ligne, sans fusion ni filtre de taille."""
    selected = set(class_ids) - set(ignore_class_ids)
    return [area for class_id, _, area in _read_polygons(label_path, image_shape)
            if class_id in selected]


def load_yolo_polygon_masks(
    label_path: str | Path,
    image_shape: tuple[int, int],
    num_classes: int,
    *,
    ignore_class_ids: Iterable[int] = (3,),
) -> np.ndarray:
    """Return independent per-class masks for YOLO polygon instances."""

    height, width = image_shape
    masks = np.zeros((num_classes, height, width), dtype=np.uint8)
    polygons = _read_polygons(label_path, image_shape)
    ignored = set(int(class_id) for class_id in ignore_class_ids)

    for class_id, mask, _ in polygons:
        if 0 <= class_id < num_classes and class_id not in ignored:
            masks[class_id] = np.maximum(masks[class_id], mask)
    return masks


def load_yolo_polygon_instances(
    label_path: str | Path,
    image_shape: tuple[int, int],
    *,
    class_ids: Iterable[int] = (0,),
    ignore_class_ids: Iterable[int] = (3,),
) -> np.ndarray:
    """Return a 2D map whose positive values identify YOLO instances."""

    height, width = image_shape
    instance_map = np.zeros((height, width), dtype=np.int32)
    selected = set(int(class_id) for class_id in class_ids)
    ignored = set(int(class_id) for class_id in ignore_class_ids)
    next_instance_id = 1

    for class_id, mask, _ in _read_polygons(label_path, image_shape):
        if class_id not in selected or class_id in ignored:
            continue
        instance_map[mask > 0] = next_instance_id
        next_instance_id += 1

    return instance_map


def result_to_instance_masks(
    result,
    class_names: Iterable[str],
) -> list[tuple[int, np.ndarray]]:
    """Convert an Ultralytics result to independent instance masks.

    The returned class indices follow ``class_names`` rather than the model's
    internal identifiers. Keeping one mask per detection prevents touching
    instances from being merged before they are exported as YOLO polygons.
    """

    height, width = result.orig_shape
    if result.masks is None or result.boxes is None:
        return []

    target_by_name = {
        class_name: class_index
        for class_index, class_name in enumerate(class_names)
    }
    names_by_id = {
        int(class_id): class_name for class_id, class_name in result.names.items()
    }
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    instances = []

    for class_id, polygon in zip(class_ids, result.masks.xy):
        target_index = target_by_name.get(names_by_id.get(class_id))
        points = np.asarray(polygon)
        if target_index is None or points.ndim != 2 or points.shape[0] < 3:
            continue

        instance_mask = np.zeros((height, width), dtype=np.uint8)
        points = np.rint(points).astype(np.int32)
        cv2.fillPoly(instance_mask, [points], 1)
        instances.append((target_index, instance_mask))

    return instances


def result_to_instance_areas(
    result, class_names: Iterable[str], *, class_ids: Iterable[int] = (0,),
) -> list[int]:
    """Mesure chaque détection indépendamment, y compris ses chevauchements."""
    selected = set(class_ids)
    return [int(np.count_nonzero(mask))
            for class_id, mask in result_to_instance_masks(result, class_names)
            if class_id in selected]


def result_to_masks(result, class_names: Iterable[str], num_classes: int) -> np.ndarray:
    """Convert one Ultralytics instance result to class-ordered masks."""

    class_names = tuple(class_names)
    height, width = result.orig_shape
    masks = np.zeros((num_classes, height, width), dtype=np.uint8)
    for target_index, instance_mask in result_to_instance_masks(result, class_names):
        masks[target_index] = np.maximum(masks[target_index], instance_mask)
    return masks


def result_to_instance_map(result, class_names: Iterable[str]) -> np.ndarray:
    """Convert the instance masks of one Ultralytics result to a 2D ID map."""

    height, width = result.orig_shape
    instance_map = np.zeros((height, width), dtype=np.int32)
    for instance_id, (_, instance_mask) in enumerate(
        result_to_instance_masks(result, class_names), start=1
    ):
        instance_map[instance_mask > 0] = instance_id

    return instance_map


def mask_to_yolo_polygons(
    mask_bool,
    class_id,
    target_size=(510, 380),
    *,
    largest_only=False,
):
    """Convertit un masque binaire en lignes polygonales YOLO normalisées."""
    if hasattr(mask_bool, "detach"):
        mask_array = mask_bool.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask_bool)
    if mask_array.ndim != 2:
        raise ValueError("Le masque à exporter doit avoir la forme (H, W).")

    resized = Image.fromarray((mask_array > 0.5).astype(np.uint8) * 255).resize(
        target_size, Image.NEAREST
    )
    mask_array = np.asarray(resized)
    contours, _ = cv2.findContours(
        np.ascontiguousarray(mask_array.astype(np.uint8)),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if largest_only and contours:
        contours = [max(contours, key=cv2.contourArea)]

    height, width = mask_array.shape
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        points = contour.reshape(-1, 2)
        coordinates = [f"{x / width:.6f} {y / height:.6f}" for x, y in points]
        polygons.append(f"{class_id} " + " ".join(coordinates))
    return polygons


def instance_masks_to_yolo_lines(
    instance_masks,
    image_shape: tuple[int, int],
) -> list[str]:
    """Convertit des instances en polygones dans le repère natif de l'image."""
    height, width = image_shape
    lines = []
    for class_index, instance_mask in instance_masks:
        instance_mask = np.asarray(instance_mask)
        if instance_mask.shape != (height, width):
            raise ValueError(
                "Le masque d'instance et l'image source n'ont pas les mêmes "
                f"dimensions : {instance_mask.shape} != {(height, width)}."
            )
        lines.extend(
            mask_to_yolo_polygons(
                instance_mask,
                class_index,
                target_size=(width, height),
                largest_only=True,
            )
        )
    return lines
