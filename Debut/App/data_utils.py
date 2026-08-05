"""Gestion des images validées "bonne", ajoutées de façon permanente aux
données d'entraînement dans data/train_added/ (séparé des données d'origine
pour pouvoir les différencier si besoin)."""

import json

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image

import config
import dataset

_EMPTY_ANN = {"images": [], "annotations": []}


def _ensure_dirs() -> None:
    config.TRAIN_ADDED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.LABELLISATION_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.CORRECTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_added_annotations(path=config.TRAIN_ADDED_ANN_PATH) -> dict:
    _ensure_dirs()
    if not path.exists():
        return {"images": [], "annotations": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_added_annotations(data: dict, path=config.TRAIN_ADDED_ANN_PATH) -> None:
    _ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_added_filenames(path=config.TRAIN_ADDED_ANN_PATH) -> list:
    data = load_added_annotations(path)
    return [img["file_name"] for img in data["images"]]


def list_labellisation_filenames() -> list:
    return list_added_filenames(config.LABELLISATION_ANN_PATH)


def list_corrected_filenames() -> list:
    return list_added_filenames(config.CORRECTED_ANN_PATH)


def get_disk_status(filename: str) -> str | None:
    """Statut déduit de la présence de filename dans train_added ("bonne") ou
    A_labelliser ("mauvaise"), déjà validé lors d'une session précédente.
    None si jamais traité."""
    if filename in list_added_filenames():
        return "bonne"
    if filename in list_labellisation_filenames():
        return "mauvaise"
    return None


def _encode_masks(masks: dict, image_id: int, start_ann_id: int) -> list:
    """Encode chaque masque de classe (dict class_name -> np.ndarray bool) en annotation COCO RLE.
    category_id = index de la classe dans config.CLASS_NAMES + 1 (pas de classe fond)."""
    annotations = []
    ann_id = start_ann_id
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        binary_mask = masks.get(class_name)
        if binary_mask is None or not binary_mask.any():
            continue
        rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_idx + 1,
                "segmentation": rle,
                "area": float(mask_util.area(rle)),
                "bbox": mask_util.toBbox(rle).tolist(),
            }
        )
        ann_id += 1
    return annotations


def add_image_to_training_set(filename: str, image: Image.Image, masks: dict) -> None:
    """Ajoute (ou remplace) une image validée "bonne" et ses masques prédits par classe."""
    remove_image_from_training_set(filename)
    _ensure_dirs()

    data = load_added_annotations()
    image_id = (max((img["id"] for img in data["images"]), default=0)) + 1
    next_ann_id = (max((ann["id"] for ann in data["annotations"]), default=0)) + 1

    width, height = image.size
    image.convert("RGB").save(config.TRAIN_ADDED_IMAGES_DIR / filename)

    data["images"].append(
        {"id": image_id, "file_name": filename, "width": width, "height": height}
    )
    data["annotations"].extend(_encode_masks(masks, image_id, next_ann_id))
    _save_added_annotations(data)


def add_image_to_labellisation_set(filename: str, image: Image.Image, masks: dict) -> None:
    """Ajoute (ou remplace) une image validée "mauvaise" et ses masques prédits par classe."""
    remove_image_from_labellisation_set(filename)
    _ensure_dirs()

    data = load_added_annotations(config.LABELLISATION_ANN_PATH)
    image_id = (max((img["id"] for img in data["images"]), default=0)) + 1
    next_ann_id = (max((ann["id"] for ann in data["annotations"]), default=0)) + 1

    width, height = image.size
    image.convert("RGB").save(config.LABELLISATION_IMAGES_DIR / filename)

    data["images"].append(
        {"id": image_id, "file_name": filename, "width": width, "height": height}
    )
    data["annotations"].extend(_encode_masks(masks, image_id, next_ann_id))
    _save_added_annotations(data, config.LABELLISATION_ANN_PATH)


def add_image_to_corrected_set(filename: str, image: Image.Image, masks: dict) -> None:
    """Ajoute (ou remplace) une image dont les masques ont été corrigés manuellement
    (bouton "Re-labellisation")."""
    remove_image_from_corrected_set(filename)
    _ensure_dirs()

    data = load_added_annotations(config.CORRECTED_ANN_PATH)
    image_id = (max((img["id"] for img in data["images"]), default=0)) + 1
    next_ann_id = (max((ann["id"] for ann in data["annotations"]), default=0)) + 1

    width, height = image.size
    image.convert("RGB").save(config.CORRECTED_IMAGES_DIR / filename)

    data["images"].append(
        {"id": image_id, "file_name": filename, "width": width, "height": height}
    )
    data["annotations"].extend(_encode_masks(masks, image_id, next_ann_id))
    _save_added_annotations(data, config.CORRECTED_ANN_PATH)


def load_corrected_image_and_masks(filename: str) -> dict | None:
    """Recharge (image, masks) déjà corrigés manuellement pour filename depuis
    Corrigees, si présent. Inverse de add_image_to_corrected_set. None sinon."""
    data = load_added_annotations(config.CORRECTED_ANN_PATH)
    image_entry = next((img for img in data["images"] if img["file_name"] == filename), None)
    if image_entry is None:
        return None

    image_id = image_entry["id"]
    height, width = image_entry["height"], image_entry["width"]
    masks = {name: np.zeros((height, width), dtype=bool) for name in config.CLASS_NAMES}
    for ann in data["annotations"]:
        if ann["image_id"] != image_id:
            continue
        class_name = config.CLASS_NAMES[ann["category_id"] - 1]
        rle = ann["segmentation"]
        if isinstance(rle["counts"], str):
            rle = {"counts": rle["counts"].encode("utf-8"), "size": rle["size"]}
        masks[class_name] = mask_util.decode(rle).astype(bool)

    image = Image.open(config.CORRECTED_IMAGES_DIR / filename).convert("RGB")
    return {"image": image, "masks": masks}


def load_coco_ground_truth_masks(filename: str) -> dict | None:
    """Charge les masques de vérité terrain (dict class_name -> np.ndarray bool) pour
    filename depuis data/COCO/annotations.json, si l'image y est présente. None sinon."""
    if not config.COCO_ANN_PATH.exists():
        return None
    with open(config.COCO_ANN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_entry = next((img for img in data["images"] if img["file_name"] == filename), None)
    if image_entry is None:
        return None

    image_id = image_entry["id"]
    height, width = image_entry["height"], image_entry["width"]
    masks = {name: np.zeros((height, width), dtype=bool) for name in config.CLASS_NAMES}
    for ann in data["annotations"]:
        if ann["image_id"] != image_id:
            continue
        class_idx = ann["category_id"] - 1
        if not (0 <= class_idx < config.NUM_CLASSES):
            continue
        decoded = dataset.decode_segmentation(ann["segmentation"], height, width)
        class_name = config.CLASS_NAMES[class_idx]
        masks[class_name] = np.maximum(masks[class_name], decoded).astype(bool)

    return masks


def remove_image_from_training_set(filename: str) -> None:
    """Retire une image (et ses annotations) précédemment ajoutée, si présente."""
    data = load_added_annotations()
    image_entry = next((img for img in data["images"] if img["file_name"] == filename), None)
    if image_entry is None:
        return

    image_id = image_entry["id"]
    data["images"] = [img for img in data["images"] if img["id"] != image_id]
    data["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] != image_id]
    _save_added_annotations(data)

    image_path = config.TRAIN_ADDED_IMAGES_DIR / filename
    if image_path.exists():
        image_path.unlink()


def remove_image_from_labellisation_set(filename: str) -> None:
    """Retire une image (et ses annotations) précédemment ajoutée, si présente."""
    data = load_added_annotations(config.LABELLISATION_ANN_PATH)
    image_entry = next((img for img in data["images"] if img["file_name"] == filename), None)
    if image_entry is None:
        return

    image_id = image_entry["id"]
    data["images"] = [img for img in data["images"] if img["id"] != image_id]
    data["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] != image_id]
    _save_added_annotations(data, config.LABELLISATION_ANN_PATH)

    image_path = config.LABELLISATION_IMAGES_DIR / filename
    if image_path.exists():
        image_path.unlink()


def remove_image_from_corrected_set(filename: str) -> None:
    """Retire une image (et ses annotations) précédemment ajoutée, si présente."""
    data = load_added_annotations(config.CORRECTED_ANN_PATH)
    image_entry = next((img for img in data["images"] if img["file_name"] == filename), None)
    if image_entry is None:
        return

    image_id = image_entry["id"]
    data["images"] = [img for img in data["images"] if img["id"] != image_id]
    data["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] != image_id]
    _save_added_annotations(data, config.CORRECTED_ANN_PATH)

    image_path = config.CORRECTED_IMAGES_DIR / filename
    if image_path.exists():
        image_path.unlink()
