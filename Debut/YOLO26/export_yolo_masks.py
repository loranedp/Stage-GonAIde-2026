"""Réexporte les prédictions YOLO dans le repère natif des images source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


YOLO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = YOLO_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.stats_fct import mask_to_yolo_polygons
from utils.yolo_metrics import result_to_instance_masks


DATASETS = {
    "3classes": ("Cavite", "Gonade", "Intestin"),
    "2classes": ("Cavite", "Gonade"),
    "oeufs": ("Oeuf",),
    "oeufsclasses": ("Gonade", "Oeuf"),
}


def instance_masks_to_yolo_lines(
    instance_masks,
    image_shape: tuple[int, int],
) -> list[str]:
    """Convertit des instances dans le repère natif de l'image source."""
    height, width = image_shape
    lines = []
    for class_index, instance_mask in instance_masks:
        if instance_mask.shape != (height, width):
            raise ValueError(
                "Le masque d'instance et l'image source n'ont pas les mêmes "
                f"dimensions : {instance_mask.shape} != {(height, width)}."
            )
        lines.extend(
            mask_to_yolo_polygons(
                torch.from_numpy(instance_mask),
                class_index,
                target_size=(width, height),
                largest_only=True,
            )
        )
    return lines


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
    from ultralytics import YOLO

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
    for image_path, prediction in zip(image_paths, predictions):
        output_path = output_dir / f"pred_{image_path.stem}.txt"
        instance_masks = result_to_instance_masks(prediction, names)
        lines = instance_masks_to_yolo_lines(
            instance_masks,
            prediction.orig_shape,
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
