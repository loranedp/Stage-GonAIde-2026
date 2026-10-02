"""Chargement des modèles cavité et œufs et inférence sur une image."""

import numpy as np
import streamlit as st
import torch
from PIL import Image

import config
import metadata_utils
from utils.post_traitement import fill_holes, keep_largest_components, split_eggs

import segmentation_models_pytorch as smp


def _build_model(num_classes: int = config.NUM_CLASSES) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


@st.cache_resource
def load_models() -> list:
    """Charge les modèles cavité de l'ensemble (state_dict, strict)."""
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


@st.cache_resource
def load_egg_model():
    """Charge le modèle YOLO instance entraîné sur l'ensemble des données œufs."""
    from ultralytics import YOLO

    if not config.EGG_YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint YOLO œufs introuvable : {config.EGG_YOLO_MODEL_PATH}"
        )
    return YOLO(str(config.EGG_YOLO_MODEL_PATH))


def _preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    x, y, w, h = config.CROP_PARAMS
    cropped = image.convert("RGB").crop((x, y, x + w, y + h))
    resized = cropped.resize(config.IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.array(resized, dtype=np.float32) / 255.0
    return torch.as_tensor(array).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)


def _preprocess(image: Image.Image, id_image: str, device: torch.device) -> torch.Tensor:
    meta = metadata_utils.lookup(id_image)
    if meta is None:
        raise ValueError(
            f"Aucune métadonnée trouvée pour id_image='{id_image}' dans "
            f"{config.METADATA_XLSX_PATH.name} (colonne image_id)."
        )
    categorie = meta["categorie"]
    if categorie not in metadata_utils.CME_CATEGORIES:
        raise ValueError(f"Catégorie inconnue '{categorie}' pour id_image='{id_image}'.")

    return _preprocess_image(image, device)


def predict(models: list, image: Image.Image, id_image: str) -> dict:
    """Prédit les masques Cavite/Gonade d'une image.

    Retourne un dict :
      - masks: dict {class_name: np.ndarray bool (H, W)} à la résolution d'origine
      - confidences: score moyen de l'ensemble sur les pixels prédits pour chaque classe.
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


def predict_eggs(
    model,
    image: Image.Image,
) -> dict:
    """Prédit les instances d'œufs avec YOLO et les replace dans l'image source."""
    if model is None:
        raise ValueError("Aucun modèle YOLO œufs n'est chargé.")
    original_size = image.size
    x, y, w, h = config.CROP_PARAMS
    cropped = image.convert("RGB").crop((x, y, x + w, y + h))
    device = "0" if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source=np.asarray(cropped),
        imgsz=config.IMAGE_SIZE[0],
        conf=config.YOLO_CONFIDENCE,
        retina_masks=True,
        verbose=False,
        device=device,
    )

    instances_crop = np.zeros((h, w), dtype=np.int32)
    egg_areas_px = []
    confidences = []
    if results and results[0].masks is not None:
        instance_masks = results[0].masks.data.detach().cpu().numpy() > 0.5
        boxes = results[0].boxes
        if boxes is not None:
            confidences = boxes.conf.detach().cpu().numpy().astype(float).tolist()
        paste_w = max(0, min(w, original_size[0] - x))
        paste_h = max(0, min(h, original_size[1] - y))
        for instance_mask in instance_masks:
            if instance_mask.shape != instances_crop.shape:
                instance_mask = np.asarray(
                    Image.fromarray(instance_mask.astype(np.uint8)).resize(
                        (w, h), Image.Resampling.NEAREST
                    ),
                    dtype=bool,
                )
            visible_mask = instance_mask[:paste_h, :paste_w]
            area_px = int(np.count_nonzero(visible_mask))
            if area_px == 0:
                continue
            egg_areas_px.append(area_px)
            instances_crop[:paste_h, :paste_w][visible_mask] = len(egg_areas_px)
    semantic_crop = instances_crop > 0

    full_semantic = Image.new("L", original_size, 0)
    full_semantic.paste(Image.fromarray(semantic_crop.astype(np.uint8) * 255), (x, y))
    full_instances = np.zeros((original_size[1], original_size[0]), dtype=np.int32)
    paste_w = min(w, original_size[0] - x)
    paste_h = min(h, original_size[1] - y)
    if paste_w > 0 and paste_h > 0:
        full_instances[y : y + paste_h, x : x + paste_w] = instances_crop[:paste_h, :paste_w]

    confidence = float(np.mean(confidences)) if confidences else None
    return {
        "masks": {"Oeuf": np.asarray(full_semantic) > 0},
        "instances": full_instances,
        "egg_areas_px": egg_areas_px,
        "confidences": {"Oeuf": confidence},
    }


def split_egg_mask(
    mask: np.ndarray,
    min_distance: int = config.EGG_MIN_DISTANCE,
    min_area: int = config.EGG_MIN_AREA,
) -> np.ndarray:
    """Recalcule les instances après une correction manuelle du masque Oeuf."""
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2:
        raise ValueError("Le masque Oeuf doit être bidimensionnel.")
    tensor = torch.as_tensor(mask_array).unsqueeze(0).unsqueeze(0)
    return (
        split_eggs(tensor, min_distance=min_distance, min_area=min_area)[0]
        .cpu()
        .numpy()
        .astype(np.int32)
    )
