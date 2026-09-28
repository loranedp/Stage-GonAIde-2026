"""Répare les exports YOLO d'œufs depuis les overlays U-Net existants."""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from PIL import Image

from utils.stats_fct import INSTANCE_OVERLAY_COLORS, mask_to_yolo_polygons


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERLAY_DIR = PROJECT_ROOT / "UNet/results/eval/UNet/oeufs"
DEFAULT_MASK_DIR = PROJECT_ROOT / "UNet/masks/eval/UNet/oeufs"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data/COCO/images/oeufs"
ALPHA = 0.45


def _instance_color(instance_id):
    if instance_id <= len(INSTANCE_OVERLAY_COLORS):
        return INSTANCE_OVERLAY_COLORS[instance_id - 1]
    hue = (instance_id * 0.618033988749895) % 1.0
    return np.rint(hsv_to_rgb([hue, 0.75, 0.95]) * 255).astype(np.uint8)


def _rasterize_yolo_union(label_path, image_size):
    width, height = image_size
    union = np.zeros((height, width), dtype=np.uint8)
    with label_path.open() as label_file:
        for line in label_file:
            parts = line.split()
            if not parts:
                continue
            coords = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= width
            coords[:, 1] *= height
            cv2.fillPoly(union, [np.rint(coords).astype(np.int32)], 1)
    return union.astype(bool)


def recover_instance_labels(base_image, overlay, union_mask, max_instances=100):
    """Décode la palette d'instances d'un overlay sans relancer le modèle."""
    base_image = np.asarray(base_image, dtype=np.uint8)
    overlay = np.asarray(overlay, dtype=np.uint8)
    if base_image.shape != overlay.shape or base_image.shape[:2] != union_mask.shape:
        raise ValueError("L'image, l'overlay et le masque doivent avoir les mêmes dimensions.")

    labels = np.zeros(union_mask.shape, dtype=np.int32)
    for instance_id in range(1, max_instances + 1):
        color = _instance_color(instance_id)
        expected = (
            base_image.astype(np.float32) * (1 - ALPHA)
            + color.astype(np.float32) * ALPHA
        ).astype(np.uint8)
        matches = np.all(expected == overlay, axis=2) & union_mask
        if not np.any(matches):
            # ``split_eggs`` renumérote toujours les instances sans trou.
            break
        if np.any(matches & (labels > 0)):
            raise ValueError(f"Couleurs ambiguës pour l'instance {instance_id}.")
        labels[matches] = instance_id

    reconstructed = base_image.copy()
    for instance_id in np.unique(labels):
        if instance_id <= 0:
            continue
        pixels = labels == instance_id
        color = _instance_color(int(instance_id))
        reconstructed[pixels] = (
            reconstructed[pixels].astype(np.float32) * (1 - ALPHA)
            + color.astype(np.float32) * ALPHA
        ).astype(np.uint8)
    if not np.array_equal(reconstructed, overlay):
        raise ValueError("L'overlay reconstruit ne correspond pas exactement au PNG enregistré.")
    return labels


def _yolo_lines_from_instances(labels):
    lines = []
    for instance_id in np.unique(labels):
        if instance_id <= 0:
            continue
        polygons = mask_to_yolo_polygons(
            torch.from_numpy(labels == instance_id),
            class_id=0,
            target_size=(labels.shape[1], labels.shape[0]),
            largest_only=True,
        )
        if len(polygons) != 1:
            raise ValueError(f"L'instance {instance_id} n'a pas produit exactement un polygone.")
        lines.append(polygons[0])
    return lines


def recover_exports(overlay_dir, mask_dir, image_dir, apply=False):
    overlay_dir, mask_dir, image_dir = map(Path, (overlay_dir, mask_dir, image_dir))
    recovered = {}
    for overlay_path in sorted(overlay_dir.glob("*.png")):
        stem = overlay_path.stem
        label_path = mask_dir / f"pred_{stem}.txt"
        image_path = image_dir / f"{stem}.jpg"
        if not label_path.exists() or not image_path.exists():
            raise FileNotFoundError(f"Entrée manquante pour {stem}.")
        base = np.asarray(Image.open(image_path).convert("RGB"))
        overlay = np.asarray(Image.open(overlay_path).convert("RGB"))
        union = _rasterize_yolo_union(label_path, (base.shape[1], base.shape[0]))
        try:
            labels = recover_instance_labels(base, overlay, union)
        except (ValueError, IndexError) as error:
            raise ValueError(f"Échec de la reconstruction pour {stem}: {error}") from error
        recovered[label_path] = _yolo_lines_from_instances(labels)

    if apply:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = mask_dir.with_name(f"{mask_dir.name}_binary_backup_{timestamp}")
        shutil.copytree(mask_dir, backup_dir)
        for label_path, lines in recovered.items():
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"Sauvegarde des anciens exports : {backup_dir}")
    print(f"{len(recovered)} exports vérifiés" + (" et corrigés." if apply else "."))
    return recovered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    recover_exports(args.overlay_dir, args.mask_dir, args.image_dir, apply=args.apply)


if __name__ == "__main__":
    main()
