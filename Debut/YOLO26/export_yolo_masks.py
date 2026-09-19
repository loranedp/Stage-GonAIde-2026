"""Réexporte les prédictions YOLO dans le repère du crop U-Net."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


YOLO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = YOLO_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.stats_fct import mask_to_yolo_polygons
from utils.yolo_metrics import result_to_instance_masks, result_to_masks


CROP = (85, 33, 510, 380)
DATASETS = {
    "3classes": ("Cavite", "Gonade", "Intestin"),
    "2classes": ("Cavite", "Gonade"),
    "oeufs": ("Oeuf",),
}


def validation_paths(fold_index: int) -> list[Path]:
    """Charge les images du fold sauvegardé par le notebook."""
    fold_path = YOLO_ROOT / "folds" / f"val_fold_{fold_index - 1}.txt"
    if not fold_path.exists():
        raise FileNotFoundError(f"Fold de validation introuvable : {fold_path}")

    paths = []
    for line in fold_path.read_text().splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.is_absolute():
            path = (YOLO_ROOT / path).resolve()
        paths.append(path)
    return paths


def checkpoint_path(dataset: str, fold_index: int) -> Path:
    return (
        YOLO_ROOT
        / "runs"
        / "segment"
        / f"train_{dataset}_instance_fold_{fold_index}"
        / "weights"
        / "best.pt"
    )


def export_fold(
    dataset: str,
    fold_index: int,
    confidence: float,
    device: str,
) -> int:
    """Prédit et exporte un fold sans relancer l'entraînement."""
    names = DATASETS[dataset]
    image_paths = validation_paths(fold_index)
    checkpoint = checkpoint_path(dataset, fold_index)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}")

    model = YOLO(str(checkpoint))
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        retina_masks=True,
        conf=confidence,
        verbose=False,
        seed=42,
        workers=0,
        device=device,
        amp=False,
    )
    if len(predictions) != len(image_paths):
        raise RuntimeError(
            f"Le fold {fold_index} contient {len(image_paths)} images, "
            f"mais YOLO a retourné {len(predictions)} prédictions."
        )

    output_dir = YOLO_ROOT / "masks" / "eval" / "instance" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_x, crop_y, crop_width, crop_height = CROP

    for image_path, prediction in zip(image_paths, predictions):
        masks = result_to_masks(
            prediction,
            names,
            len(names),
        )
        if masks.shape[1] < crop_y + crop_height or masks.shape[2] < crop_x + crop_width:
            raise ValueError(
                f"Crop {CROP} incompatible avec {image_path} : "
                f"masque de forme {masks.shape}."
            )

        output_path = output_dir / f"pred_{image_path.stem}.txt"
        lines = []
        instance_masks = result_to_instance_masks(prediction, names)
        for class_index, instance_mask in instance_masks:
            cropped_mask = instance_mask[
                crop_y:crop_y + crop_height,
                crop_x:crop_x + crop_width,
            ]
            lines.extend(
                mask_to_yolo_polygons(
                    torch.from_numpy(cropped_mask),
                    class_index,
                    target_size=(crop_width, crop_height),
                    largest_only=True,
                )
            )
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return len(image_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    args = parser.parse_args()

    if not 0 <= args.confidence <= 1:
        parser.error("--confidence doit être compris entre 0 et 1")

    total = 0
    for fold_index in args.folds or range(1, 6):
        if not 1 <= fold_index <= 5:
            parser.error("--fold doit être compris entre 1 et 5")
        count = export_fold(
            args.dataset,
            fold_index,
            args.confidence,
            args.device,
        )
        total += count
        print(f"Fold {fold_index} : {count} prédictions exportées")
    print(f"Total : {total} prédictions exportées")


if __name__ == "__main__":
    main()
