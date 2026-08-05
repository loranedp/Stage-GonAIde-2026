"""Chargement de l'ensemble de modèles (2 folds) et inférence sur une image."""

import numpy as np
import streamlit as st
import torch
from PIL import Image

import config
import metadata_utils
from utils.post_traitement import fill_holes, keep_largest_components

import segmentation_models_pytorch as smp


def _build_model() -> torch.nn.Module:
    return smp.Unet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights="imagenet",
        in_channels=3,
        classes=config.NUM_CLASSES,
        activation=None,
    )


@st.cache_resource
def load_models() -> list:
    """Charge les 2 modèles de l'ensemble (state_dict, strict). Mis en cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in config.FOLD_MODEL_PATHS:
        model = _build_model()
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def _preprocess(image: Image.Image, id_image: str, device: torch.device) -> torch.Tensor:
    x, y, w, h = config.CROP_PARAMS
    cropped = image.convert("RGB").crop((x, y, x + w, y + h))
    resized = cropped.resize(config.IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.array(resized, dtype=np.float32) / 255.0
    image_tensor = torch.as_tensor(array).permute(2, 0, 1).contiguous()

    meta = metadata_utils.lookup(id_image)
    if meta is None:
        raise ValueError(
            f"Aucune métadonnée trouvée pour id_image='{id_image}' dans "
            f"{config.METADATA_XLSX_PATH.name} (colonne image_id)."
        )
    categorie = meta["categorie"]
    if categorie not in metadata_utils.CME_CATEGORIES:
        raise ValueError(f"Catégorie inconnue '{categorie}' pour id_image='{id_image}'.")

    return image_tensor.unsqueeze(0).to(device)


def predict(models: list, image: Image.Image, id_image: str) -> dict:
    """Prédit les masques multi-classes (Cavite/Gonade/Intestin) d'une image.

    Retourne un dict :
      - masks: dict {class_name: np.ndarray bool (H, W)} à la résolution d'origine
      - confidence_gonad: float (probabilité moyenne de l'ensemble sur les pixels
        prédits Gonade) ou None si aucun pixel Gonade n'est prédit.
    """
    device = next(models[0].parameters()).device
    original_size = image.size  # (W, H) de l'image complète uploadée
    input_tensor = _preprocess(image, id_image, device)

    with torch.no_grad():
        ensemble_probs = torch.zeros(
            (config.NUM_CLASSES, *config.IMAGE_SIZE[::-1]), device=device
        )
        for model in models:
            logits = model(input_tensor)
            ensemble_probs += torch.sigmoid(logits)[0]
        ensemble_probs /= len(models)

    preds_bool_tensor = (ensemble_probs > config.PRED_THRESHOLD).unsqueeze(0)  # (1, C, H, W)
    preds_bool_tensor = keep_largest_components(preds_bool_tensor)
    preds_bool_tensor = fill_holes(preds_bool_tensor)
    preds_bool = preds_bool_tensor.squeeze(0).cpu().numpy()

    # Les masques sont prédits sur la zone cropée (CROP_PARAMS) ; on les replace
    # dans le référentiel de l'image d'origine avant de retourner.
    x, y, w, h = config.CROP_PARAMS
    masks = {}
    for idx, class_name in enumerate(config.CLASS_NAMES):
        mask_small = Image.fromarray(preds_bool[idx].astype(np.uint8) * 255).resize(
            (w, h), Image.Resampling.NEAREST
        )
        full_mask = Image.new("L", original_size, 0)
        full_mask.paste(mask_small, (x, y))
        masks[class_name] = np.array(full_mask) > 0

    confidences = {}
    for idx, class_name in enumerate(config.CLASS_NAMES):
        class_probs_small = ensemble_probs[idx].cpu().numpy()
        class_mask_small = preds_bool[idx]
        if class_mask_small.any():
            confidences[class_name] = float(class_probs_small[class_mask_small].mean())
        else:
            confidences[class_name] = None

    return {"masks": masks, "confidences": confidences}
