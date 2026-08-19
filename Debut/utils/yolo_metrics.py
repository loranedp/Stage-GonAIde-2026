"""Helpers used to evaluate the polygon-based YOLO26 segmentations.

The semantic YOLO dataset has one important property which is easy to miss when
rebuilding masks manually: polygons are rasterised into one class map.  Smaller
polygons overwrite larger polygons, so a gonade or an intestine hides the
underlying cavity in the semantic target.  The functions in this module mirror
that behaviour and keep the displayed target identical to the evaluated target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np


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


def load_yolo_polygon_masks(
    label_path: str | Path,
    image_shape: tuple[int, int],
    num_classes: int,
    *,
    semantic: bool,
    ignore_class_ids: Iterable[int] = (3,),
) -> np.ndarray:
    """Return class masks matching YOLO's polygon dataset behaviour.

    In semantic mode, the returned shape is ``(num_classes, H, W)`` and the
    polygons are assigned from largest to smallest.  Consequently, a smaller
    polygon overwrites a larger one, exactly as ``PolygonSemanticDataset``
    does.  Ignore polygons are assigned to background and therefore erase any
    larger polygon underneath them.

    In instance mode, polygons are unioned independently for each foreground
    class; overlaps between classes are intentionally preserved.
    """

    height, width = image_shape
    masks = np.zeros((num_classes, height, width), dtype=np.uint8)
    polygons = _read_polygons(label_path, image_shape)
    ignored = set(int(class_id) for class_id in ignore_class_ids)

    if not semantic:
        for class_id, mask, _ in polygons:
            if 0 <= class_id < num_classes and class_id not in ignored:
                masks[class_id] = np.maximum(masks[class_id], mask)
        return masks

    background = np.full((height, width), num_classes, dtype=np.int32)
    for class_id, mask, _ in sorted(polygons, key=lambda item: item[2], reverse=True):
        target_class = class_id if 0 <= class_id < num_classes and class_id not in ignored else num_classes
        background[mask > 0] = target_class

    for class_id in range(num_classes):
        masks[class_id] = (background == class_id).astype(np.uint8)
    return masks


def semantic_result_to_masks(
    semantic_map: np.ndarray,
    result_names: Mapping[int, str],
    class_names: Iterable[str],
) -> np.ndarray:
    """Convert a YOLO semantic class map to channels ordered by ``class_names``."""

    semantic_map = np.asarray(semantic_map)
    if semantic_map.ndim != 2:
        raise ValueError(f"Une carte sémantique 2D est attendue, reçu {semantic_map.shape}")

    ids_by_name = {name: int(class_id) for class_id, name in result_names.items()}
    missing = [name for name in class_names if name not in ids_by_name]
    if missing:
        raise ValueError(f"Classes absentes de la prédiction YOLO : {missing}")

    return np.stack(
        [(semantic_map == ids_by_name[name]).astype(np.uint8) for name in class_names],
        axis=0,
    )


def result_to_masks(result, class_names: Iterable[str], num_classes: int, *, semantic: bool) -> np.ndarray:
    """Convert one Ultralytics result to channels ordered by ``class_names``."""

    class_names = tuple(class_names)
    if semantic:
        semantic_map = result.semantic_mask.data.detach().cpu().numpy()
        return semantic_result_to_masks(semantic_map, result.names, class_names)

    height, width = result.orig_shape
    masks = np.zeros((num_classes, height, width), dtype=np.uint8)
    if result.masks is None or result.boxes is None:
        return masks

    names_by_id = {int(class_id): name for class_id, name in result.names.items()}
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    for instance_index, class_id in enumerate(class_ids):
        class_name = names_by_id.get(class_id)
        if class_name not in class_names:
            continue
        target_index = class_names.index(class_name)
        points = np.rint(result.masks.xy[instance_index]).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(masks[target_index], [points], 1)
    return masks


def metrics_from_counts(
    intersection: np.ndarray,
    union: np.ndarray,
    predicted_area: np.ndarray,
    true_area: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute stable IoU/Dice/precision/recall from pooled pixel counts."""

    intersection = np.asarray(intersection, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    predicted_area = np.asarray(predicted_area, dtype=np.float64)
    true_area = np.asarray(true_area, dtype=np.float64)

    total_area = predicted_area + true_area
    metrics = {
        "iou": np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0),
        "dice": np.divide(
            2 * intersection,
            total_area,
            out=np.ones_like(intersection),
            where=total_area > 0,
        ),
        "precision": np.divide(
            intersection,
            predicted_area,
            out=(true_area == 0).astype(np.float64),
            where=predicted_area > 0,
        ),
        "recall": np.divide(
            intersection,
            true_area,
            out=(predicted_area == 0).astype(np.float64),
            where=true_area > 0,
        ),
    }
    return metrics
