"""Ré-entraînement (fine-tuning) des deux modèles de l'ensemble sur les données
d'origine + les images ajoutées via l'app. Chaque fold est continué indépendamment
depuis son checkpoint existant et réécrit au même emplacement."""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
import metadata_utils
from dataset import MultiClassCMEDataset
from utils.models_config import BCE_DiceLoss, DiceLoss

import segmentation_models_pytorch as smp


def _dice_score(model: nn.Module, loader: Optional[DataLoader], device: torch.device) -> Optional[float]:
    """Évalue model (mode eval, sans gradient) sur loader et retourne le score de Dice moyen
    (1 - DiceLoss), c'est-à-dire le recouvrement moyen prédiction/vérité terrain sur des
    images non vues par ce fold pendant son entraînement. Retourne None si loader est vide."""
    if loader is None:
        return None
    dice_loss_fn = DiceLoss()
    model.eval()
    total_score = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            loss = dice_loss_fn(model(images), masks)
            bsz = images.size(0)
            total_score += (1.0 - loss.item()) * bsz
            count += bsz
    return total_score / count if count else None


def train_model(
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 4,
    progress_callback: Optional[Callable[[int, int, int, int, float], None]] = None,
) -> dict:
    """Fine-tune chaque fold indépendamment sur un vrai split train/validation (groupé par
    id_poisson, cf. dataset.MultiClassCMEDataset.fold_indices) : chaque fold s'entraîne sur
    les autres groupes de poissons et est évalué (score de Dice) sur son groupe held-out,
    jamais vu pendant son propre entraînement. progress_callback(fold_idx, num_folds, epoch,
    total_epochs, loss)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_folds = len(config.FOLD_MODEL_PATHS)

    full_dataset = MultiClassCMEDataset()
    if len(full_dataset) == 0:
        raise ValueError(
            "Aucune donnée d'entraînement disponible (vérifier data/COCO, "
            "data/train_added, et les correspondances dans utils/correspondance_echo_final.xlsx)."
        )

    criterion = BCE_DiceLoss(bce_weight=0.5, dice_weight=0.5)
    in_channels = 3

    fold_metrics = []
    for fold_num, path in enumerate(config.FOLD_MODEL_PATHS, start=1):
        train_indices, val_indices = full_dataset.fold_indices(fold_num - 1, num_folds)
        if not train_indices:
            raise ValueError(
                f"Fold {fold_num} : aucune donnée d'entraînement restante après avoir isolé "
                "le groupe de validation (trop peu de poissons distincts pour le nombre de folds)."
            )

        train_loader = DataLoader(
            Subset(full_dataset, train_indices), batch_size=batch_size, shuffle=True
        )
        val_loader = (
            DataLoader(Subset(full_dataset, val_indices), batch_size=batch_size, shuffle=False)
            if val_indices
            else None
        )

        model = smp.Unet(
            encoder_name=config.ENCODER_NAME,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=config.NUM_CLASSES
        )
        model.load_state_dict(torch.load(path, map_location=device), strict=True)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        last_loss = 0.0
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            count = 0
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                bsz = images.size(0)
                running_loss += loss.item() * bsz
                count += bsz

            last_loss = running_loss / count if count else 0.0
            if progress_callback is not None:
                progress_callback(fold_num, num_folds, epoch, epochs, last_loss)

        val_dice = _dice_score(model, val_loader, device)

        model.eval()
        torch.save(model.state_dict(), path)

        fold_metrics.append(
            {
                "fold": fold_num,
                "train_loss": last_loss,
                "val_dice": val_dice,
                "num_train": len(train_indices),
                "num_val": len(val_indices),
            }
        )

    return {
        "num_samples": len(full_dataset),
        "final_losses": [m["train_loss"] for m in fold_metrics],
        "fold_metrics": fold_metrics,
    }
