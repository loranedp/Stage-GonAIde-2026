"""U-Net pour la segmentation et la séparation d'œufs rapprochés.

Le modèle suit le principe décrit dans l'article fourni : il prédit un masque
binaire des œufs ainsi que deux cartes de déplacement vers le centre de
chaque instance (horizontalement et verticalement). Les cartes H/V peuvent
ensuite être utilisées par :func:`postprocess_egg_instances` pour séparer les
œufs qui se touchent.

La tête est volontairement limitée à la segmentation : aucune classification
du stade de maturation n'est réalisée.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import watershed
import segmentation_models_pytorch as smp


class EggSegmentationHVUNet(nn.Module):
    """U-Net produisant ``[mask, horizontal, vertical]``.

    Args:
        in_channels: Nombre de canaux de l'image d'entrée.
        encoder_name: Encodeur accepté par ``segmentation_models_pytorch``.
        encoder_weights: Poids de pré-entraînement, ou ``None``.
    """

    OUTPUT_CHANNELS = 3

    def __init__(
        self,
        in_channels: int = 3,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=self.OUTPUT_CHANNELS,
            activation=None,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"images doit être [B, C, H, W], reçu {tuple(images.shape)}"
            )
        return self.base_model(images)

    @staticmethod
    def split_output(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sépare les logits en masque, carte H et carte V."""
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError("logits doit avoir la forme [B, 3, H, W]")
        return logits[:, 0:1], logits[:, 1:2], logits[:, 2:3]


def _as_instance_mask(instance: Any, height: int, width: int) -> np.ndarray:
    """Convertit une instance en masque booléen.

    ``instance`` peut être un masque 2D, un polygone ``[(x, y), ...]`` ou une
    liste plate COCO ``[x1, y1, x2, y2, ...]``. Les polygones sont rasterisés
    avec OpenCV uniquement au moment de la préparation des cibles.
    """
    import cv2

    # Une annotation COCO peut contenir plusieurs anneaux/polygones pour une
    # même instance. On les fusionne avant de calculer le centroïde.
    if isinstance(instance, (list, tuple)) and instance and isinstance(
        instance[0], (list, tuple, np.ndarray)
    ):
        polygons_mask = np.zeros((height, width), dtype=bool)
        for polygon in instance:
            polygons_mask |= _as_instance_mask(polygon, height, width)
        return polygons_mask

    array = np.asarray(instance)
    if array.ndim == 2 and array.shape == (height, width):
        return array.astype(bool)

    if array.ndim == 1:
        if array.size < 6 or array.size % 2:
            raise ValueError("Un polygone doit contenir au moins trois points")
        array = array.reshape(-1, 2)
    elif array.ndim == 2 and array.shape[1] == 2:
        pass
    else:
        raise ValueError("Instance attendue : masque [H,W] ou polygone [N,2]")

    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.rint(array).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def generate_hv_targets(
    instances: Sequence[Any] | np.ndarray,
    height: int | None = None,
    width: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Génère ``(mask, H, V)`` à partir de polygones ou masques d'instances.

    H et V sont nuls hors des œufs. Dans chaque instance, ils représentent la
    distance signée et normalisée du pixel vers le centre de l'instance :
    ``H = (cx - x) / rayon_x`` et ``V = (cy - y) / rayon_y``.
    """
    if isinstance(instances, np.ndarray) and instances.ndim == 3:
        if height is None or width is None:
            _, height, width = instances.shape
        instances = [instances[index] for index in range(instances.shape[0])]
    elif height is None or width is None:
        raise ValueError("height et width sont requis pour une liste de polygones")

    if height <= 0 or width <= 0:
        raise ValueError("height et width doivent être positifs")

    mask = np.zeros((height, width), dtype=np.float32)
    horizontal = np.zeros_like(mask)
    vertical = np.zeros_like(mask)

    for instance in instances:
        instance_mask = _as_instance_mask(instance, height, width)
        ys, xs = np.nonzero(instance_mask)
        if len(xs) == 0:
            continue

        # Le centroïde des pixels est stable même pour des annotations non
        # convexes et évite d'imposer une forme géométrique aux œufs.
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        radius_x = max(float(np.max(np.abs(xs - center_x))), 1.0)
        radius_y = max(float(np.max(np.abs(ys - center_y))), 1.0)

        mask[instance_mask] = 1.0
        horizontal[instance_mask] = (center_x - xs) / radius_x
        vertical[instance_mask] = (center_y - ys) / radius_y

    return mask, horizontal, vertical


class EggSegmentationHVLoss(nn.Module):
    """Loss BCE/Dice pour le masque et Smooth-L1 pour les cartes H/V."""

    def __init__(
        self,
        mask_weight: float = 10.0,
        hv_weight: float = 1.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        if mask_weight <= 0 or hv_weight < 0:
            raise ValueError("mask_weight doit être > 0 et hv_weight >= 0")
        self.mask_weight = mask_weight
        self.hv_weight = hv_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError("logits doit avoir la forme [B, 3, H, W]")
        if targets.shape != logits.shape:
            raise ValueError(
                f"targets doit avoir la même forme que logits, reçu {tuple(targets.shape)}"
            )

        target_mask = targets[:, 0:1].float().clamp(0, 1)
        target_hv = targets[:, 1:3].float().clamp(-1, 1)
        mask_logits = logits[:, 0:1]
        hv_logits = logits[:, 1:3]

        bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        probabilities = torch.sigmoid(mask_logits)
        intersection = (probabilities * target_mask).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
        dice = ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        mask_loss = 0.5 * bce + 0.5 * (1.0 - dice)

        # Les valeurs H/V hors masque sont artificiellement nulles et ne
        # doivent pas dominer la régression lorsque les œufs sont petits.
        valid = target_mask.expand_as(target_hv)
        hv_error = F.smooth_l1_loss(hv_logits * valid, target_hv * valid)
        return self.mask_weight * mask_loss + self.hv_weight * hv_error


def postprocess_egg_instances(
    logits: torch.Tensor | np.ndarray,
    mask_threshold: float = 0.5,
    min_distance: int = 8,
    min_area: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Transforme une prédiction en masque binaire et labels d'instances.

    Args:
        logits: Un tenseur ``[3,H,W]`` ou ``[1,3,H,W]``.
        mask_threshold: Seuil de probabilité du masque œuf.
        min_distance: Distance minimale entre deux marqueurs.
        min_area: Aire minimale conservée pour une instance.

    Returns:
        ``(egg_mask, instance_labels)`` où le masque est booléen et les labels
        sont des entiers, avec 0 pour le fond.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().float().cpu()
        if logits.ndim == 4:
            if logits.shape[0] != 1:
                raise ValueError("Le post-traitement ne traite qu'une image à la fois")
            logits = logits[0]
        logits = logits.numpy()
    logits = np.asarray(logits)
    if logits.ndim != 3 or logits.shape[0] != 3:
        raise ValueError("logits doit avoir la forme [3,H,W] ou [1,3,H,W]")
    if not 0 < mask_threshold < 1:
        raise ValueError("mask_threshold doit être compris entre 0 et 1")

    mask_probability = 1.0 / (1.0 + np.exp(-np.clip(logits[0], -60, 60)))
    egg_mask = mask_probability >= mask_threshold
    if not egg_mask.any():
        return egg_mask, np.zeros_like(egg_mask, dtype=np.int32)

    horizontal = np.tanh(logits[1])
    vertical = np.tanh(logits[2])
    magnitude = np.sqrt(horizontal * horizontal + vertical * vertical)

    # Les minima de magnitude correspondent aux centres prédits. Le filtre de
    # distance évite de créer plusieurs marqueurs dans un même œuf.
    distance = ndimage.distance_transform_edt(egg_mask)
    center_score = distance * (1.0 - np.minimum(magnitude, 1.0))
    local_max = center_score == ndimage.maximum_filter(
        center_score, size=max(3, 2 * min_distance + 1)
    )
    local_max &= egg_mask
    markers, marker_count = ndimage.label(local_max)

    if marker_count == 0:
        markers, marker_count = ndimage.label(egg_mask)
    labels = watershed(-center_score, markers=markers, mask=egg_mask)

    if min_area > 0:
        areas = np.bincount(labels.ravel())
        small_labels = np.flatnonzero(areas < min_area)
        labels[np.isin(labels, small_labels)] = 0
        # Réindexation compacte des instances restantes.
        unique = np.unique(labels)
        remap = np.zeros(int(unique.max()) + 1 if unique.size else 1, dtype=np.int32)
        remap[unique] = np.arange(unique.size, dtype=np.int32)
        labels = remap[labels]

    return labels > 0, labels.astype(np.int32)


__all__ = [
    "EggSegmentationHVUNet",
    "EggSegmentationHVLoss",
    "generate_hv_targets",
    "postprocess_egg_instances",
]
