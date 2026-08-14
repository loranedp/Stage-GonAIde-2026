"""Opérations pures utilisées par l'éditeur manuel de masques."""

import numpy as np
from PIL import Image


def extract_stroke_mask(image_data: np.ndarray, mode: str) -> np.ndarray:
    """Isole le tracé vert (ajout) ou rouge (effacement) d'une image RGBA/RGB."""
    data = np.asarray(image_data)
    if data.ndim != 3 or data.shape[2] not in (3, 4):
        raise ValueError(f"Image du canvas invalide : forme reçue {data.shape}.")
    if mode not in {"Ajouter", "Effacer"}:
        raise ValueError(f"Mode d'édition inconnu : {mode!r}.")

    r = data[..., 0].astype(np.int16)
    g = data[..., 1].astype(np.int16)
    b = data[..., 2].astype(np.int16)
    visible = data[..., 3] > 0 if data.shape[2] == 4 else np.ones(data.shape[:2], dtype=bool)
    if mode == "Ajouter":
        color_match = (g > 180) & (r < 120) & (b < 120)
    else:
        color_match = (r > 180) & (g < 120) & (b < 120)
    return visible & color_match


def apply_stroke_to_masks(
    masks: dict[str, np.ndarray],
    class_name: str,
    stroke_small: np.ndarray,
    mode: str,
    crop_params: tuple[int, int, int, int] | list[int],
) -> None:
    """Applique en place un tracé du canvas au masque à pleine résolution."""
    if class_name not in masks:
        raise ValueError(f"Classe absente des masques : {class_name!r}.")
    if mode not in {"Ajouter", "Effacer"}:
        raise ValueError(f"Mode d'édition inconnu : {mode!r}.")

    stroke = np.asarray(stroke_small, dtype=bool)
    if stroke.ndim != 2 or not all(value > 0 for value in stroke.shape):
        raise ValueError(f"Tracé invalide : forme reçue {stroke.shape}.")
    mask = np.asarray(masks[class_name], dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Masque {class_name!r} invalide : forme reçue {mask.shape}.")

    x, y, width, height = crop_params
    if (
        min(x, y) < 0
        or width <= 0
        or height <= 0
        or x + width > mask.shape[1]
        or y + height > mask.shape[0]
    ):
        raise ValueError(
            f"Crop {(x, y, width, height)} hors du masque {mask.shape}."
        )

    stroke_image = Image.fromarray((stroke * 255).astype(np.uint8)).resize(
        (width, height), Image.Resampling.NEAREST
    )
    full_stroke = np.zeros_like(mask)
    full_stroke[y : y + height, x : x + width] = np.asarray(stroke_image) > 0
    masks[class_name] = mask | full_stroke if mode == "Ajouter" else mask & ~full_stroke


def apply_pending_stroke(
    edit_state: dict, crop_params: tuple[int, int, int, int] | list[int]
) -> bool:
    """Applique puis consomme le tracé en attente d'une session d'édition."""
    pending = edit_state.get("pending")
    if pending is None:
        return False
    apply_stroke_to_masks(
        edit_state["masks"],
        pending["class_name"],
        pending["stroke"],
        pending["mode"],
        crop_params,
    )
    edit_state["pending"] = None
    return True
