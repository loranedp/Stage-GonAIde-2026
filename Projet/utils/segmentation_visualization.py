"""Visualisations communes aux pipelines de segmentation YOLO et U-Net."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, hsv_to_rgb
from matplotlib.patches import Patch
from PIL import Image

from utils.unet_spatial import unpad_array


CLASS_OVERLAY_COLORS = np.array(
    [[26, 12, 176], [69, 209, 179], [199, 182, 179]], dtype=np.uint8
)
INSTANCE_OVERLAY_COLORS = np.array(
    [
        [230, 70, 60],
        [55, 185, 75],
        [55, 125, 225],
        [205, 65, 175],
        [235, 160, 45],
        [35, 185, 190],
        [150, 75, 215],
        [225, 85, 125],
    ],
    dtype=np.uint8,
)


def tensor_to_numpy_image(tensor):
    """Convertit un tenseur CHW ou une image numpy en image HWC normalisée."""
    if hasattr(tensor, "detach"):
        image = tensor.detach().cpu().numpy()
    else:
        image = np.asarray(tensor)
    if image.ndim == 3:
        if image.shape[0] <= 4 and image.shape[-1] > 4:
            image = image[:3]
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        elif image.shape[-1] <= 4:
            image = image[..., :3]
    if image.size and image.max() > 1.0:
        image = image / 255.0
    return image


def overlay_mask(image_tensor, mask_tensor, alpha=0.4, target_size=None):
    """Superpose les canaux d'un masque sémantique avec la palette commune."""
    image = (tensor_to_numpy_image(image_tensor) * 255).astype(np.uint8)
    mask = (
        mask_tensor.detach().cpu().numpy()
        if hasattr(mask_tensor, "detach")
        else np.asarray(mask_tensor)
    ).astype(np.float32)
    if target_size is not None:
        image = cv2.resize(image, target_size)

    height, width = image.shape[:2]
    resized_mask = np.zeros((mask.shape[0], height, width), dtype=np.float32)
    for class_index, class_mask in enumerate(mask):
        resized_mask[class_index] = (
            cv2.resize(class_mask, (width, height))
            if class_mask.shape != (height, width)
            else class_mask
        )

    overlay = image.copy()
    for class_index, class_mask in enumerate(resized_mask):
        color = CLASS_OVERLAY_COLORS[class_index % len(CLASS_OVERLAY_COLORS)]
        pixels = class_mask > 0.5
        overlay[pixels] = (
            overlay[pixels].astype(np.float32) * (1 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
    return overlay


def overlay_colored_mask(
    image, mask, alpha=0.45, dataset_name=None, class_names=None, target_size=None
):
    """Superpose un masque sémantique ou une carte d'instances sur une image."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha doit être compris entre 0 et 1.")

    base = tensor_to_numpy_image(image)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=2)
    if base.shape[-1] == 1:
        base = np.repeat(base, 3, axis=2)
    base = np.clip(base, 0, 1) if np.issubdtype(base.dtype, np.floating) else base
    if base.size and base.max() <= 1.0:
        base = base * 255
    base = np.clip(base, 0, 255).astype(np.uint8)
    if target_size is not None:
        base = cv2.resize(base, target_size, interpolation=cv2.INTER_LINEAR)

    raw_mask = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    if dataset_name == "oeufs":
        if raw_mask.ndim == 3 and raw_mask.shape[0] == 1:
            raw_mask = raw_mask[0]
        if raw_mask.ndim != 2:
            raise ValueError("Le masque d'instances doit avoir la forme (H, W) ou (1, H, W).")
        labels = raw_mask
    else:
        if raw_mask.ndim == 2:
            raw_mask = raw_mask[None, ...]
        if raw_mask.ndim != 3:
            raise ValueError("Le masque sémantique doit avoir la forme (C, H, W).")
        labels = np.zeros(raw_mask.shape[1:], dtype=np.int32)
        for class_index, class_mask in enumerate(raw_mask):
            labels[class_mask > 0.5] = class_index + 1

    if labels.shape != base.shape[:2]:
        labels = cv2.resize(
            labels.astype(np.int32),
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    labels = np.rint(labels).astype(np.int64)
    output = base.copy()
    colors = INSTANCE_OVERLAY_COLORS if dataset_name == "oeufs" else CLASS_OVERLAY_COLORS
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        if dataset_name == "oeufs" and label_id > len(colors):
            hue = (int(label_id) * 0.618033988749895) % 1.0
            color = np.rint(hsv_to_rgb([hue, 0.75, 0.95]) * 255).astype(np.uint8)
        else:
            color = colors[(int(label_id) - 1) % len(colors)]
        pixels = labels == label_id
        output[pixels] = (
            output[pixels].astype(np.float32) * (1 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
    return output


def overlay_semantic_instances(
    image,
    semantic_mask,
    instance_mask,
    *,
    semantic_class_names=("Gonade",),
    alpha=0.4,
):
    """Compose un masque sémantique puis une carte d'instances colorées."""
    semantic_overlay = overlay_colored_mask(
        image,
        semantic_mask,
        alpha=alpha,
        dataset_name="semantic",
        class_names=semantic_class_names,
    )
    return overlay_colored_mask(
        semantic_overlay,
        instance_mask,
        alpha=alpha,
        dataset_name="oeufs",
        class_names=("Œuf",),
    )


def plot_segmentation_comparison(image, mask_true, mask_pred, is_eggs=False, alpha=0.45):
    """Affiche l'image, la vérité terrain et la prédiction côte à côte."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha doit être compris entre 0 et 1.")

    image_display = np.clip(tensor_to_numpy_image(image), 0, 1)
    height, width = image_display.shape[:2]

    def resize_label_map(label_map):
        if label_map.shape != (height, width):
            label_map = cv2.resize(
                label_map.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        return label_map

    def as_numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def semantic_labels(mask):
        mask = as_numpy(mask)
        if mask.ndim != 3 or not 1 <= mask.shape[0] <= 3:
            raise ValueError("Les masques doivent avoir la forme (C, H, W), avec 1 à 3 classes.")
        labels = np.zeros((height, width), dtype=np.uint8)
        for class_index, class_mask in enumerate(mask):
            labels[resize_label_map(class_mask) > 0.5] = class_index + 1
        return labels

    if is_eggs:
        def egg_labels(mask, name):
            mask = as_numpy(mask)
            if mask.ndim == 3 and mask.shape[0] == 1:
                return (resize_label_map(mask[0]) > 0.5).astype(np.int32)
            if mask.ndim != 2:
                raise ValueError(f"Le masque {name} des œufs doit avoir la forme (1, H, W) ou (H, W).")
            labels = np.rint(resize_label_map(mask)).astype(np.int32)
            if np.any(labels < 0):
                raise ValueError("Les identifiants d'instances doivent être positifs ou nuls.")
            return labels

        def instance_colormap(labels):
            count = max(1, int(labels.max()))
            colors = ["#000000"] + [plt.cm.tab20(index % 20) for index in range(count)]
            return ListedColormap(colors), BoundaryNorm(np.arange(-0.5, count + 1.5), len(colors))

        true_labels = egg_labels(mask_true, "réel")
        pred_labels = egg_labels(mask_pred, "prédit")
        true_cmap, true_norm = instance_colormap(true_labels)
        pred_cmap, pred_norm = instance_colormap(pred_labels)
        legend_handles = []
    else:
        true_array = as_numpy(mask_true)
        pred_array = as_numpy(mask_pred)
        if true_array.shape[0] != pred_array.shape[0]:
            raise ValueError("mask_true et mask_pred doivent avoir le même nombre de classes.")
        true_labels = semantic_labels(true_array)
        pred_labels = semantic_labels(pred_array)
        class_colors = ["#1A0CB0", "#66CCFF", "#8E44AD"]
        semantic_cmap = ListedColormap(["#000000", *class_colors])
        semantic_norm = BoundaryNorm(
            np.arange(-0.5, len(class_colors) + 1.5), semantic_cmap.N
        )
        true_cmap = pred_cmap = semantic_cmap
        true_norm = pred_norm = semantic_norm
        class_names = ["Cavité", "Gonade", "Intestin"]
        legend_handles = [
            Patch(color=class_colors[index], label=class_names[index])
            for index in range(true_array.shape[0])
        ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for axis, title, labels, cmap, norm in zip(
        axes,
        ["Image originale", "Vérité terrain", "Prédiction"],
        [None, true_labels, pred_labels],
        [None, true_cmap, pred_cmap],
        [None, true_norm, pred_norm],
    ):
        axis.imshow(image_display)
        if labels is not None:
            axis.imshow(np.ma.masked_where(labels == 0, labels), cmap=cmap, norm=norm, alpha=alpha)
        axis.set_title(title)
        axis.axis("off")
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=min(5, len(legend_handles)))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
    return fig, axes


def save_overlay_comparison(
    output_path,
    image,
    true_mask,
    pred_mask,
    *,
    dataset_name=None,
    class_names=None,
    true_instances=None,
    pred_instances=None,
    alpha=0.4,
):
    """Sauvegarde une comparaison annotation/prédiction côte à côte."""
    if true_instances is not None or pred_instances is not None:
        if true_instances is None or pred_instances is None:
            raise ValueError("Les deux cartes d'instances sont requises.")
        true_overlay = overlay_semantic_instances(image, true_mask, true_instances, alpha=alpha)
        pred_overlay = overlay_semantic_instances(image, pred_mask, pred_instances, alpha=alpha)
    else:
        true_overlay = overlay_colored_mask(
            image, true_mask, alpha=alpha, dataset_name=dataset_name, class_names=class_names
        )
        pred_overlay = overlay_colored_mask(
            image, pred_mask, alpha=alpha, dataset_name=dataset_name, class_names=class_names
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.hstack((true_overlay, pred_overlay))).save(output_path)
    return output_path


def save_validation_overlays(results_dir, best_images, dataset_name=None, class_names=None):
    """Sauvegarde les superpositions du meilleur epoch U-Net."""
    for file_name, image, pred_mask in best_images:
        overlay = overlay_colored_mask(
            unpad_array(image),
            unpad_array(pred_mask),
            alpha=0.45,
            dataset_name=dataset_name,
            class_names=class_names,
        )
        Image.fromarray(overlay).save(Path(results_dir) / f"{Path(file_name).stem}.png")


__all__ = [
    "CLASS_OVERLAY_COLORS",
    "INSTANCE_OVERLAY_COLORS",
    "overlay_colored_mask",
    "overlay_mask",
    "overlay_semantic_instances",
    "plot_segmentation_comparison",
    "save_overlay_comparison",
    "save_validation_overlays",
    "tensor_to_numpy_image",
]
