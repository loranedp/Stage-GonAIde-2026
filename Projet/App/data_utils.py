"""Persistance robuste des images et annotations ajoutées depuis l'application."""

import io
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image

import config
from utils.dataset_unet import decode_segmentation

logger = logging.getLogger(__name__)


class DataStoreError(RuntimeError):
    """Erreur contextualisée de lecture ou d'écriture des annotations de l'app."""


def _ensure_dirs() -> None:
    for directory in (
        config.TRAIN_ADDED_IMAGES_DIR,
        config.LABELLISATION_IMAGES_DIR,
        config.CORRECTED_IMAGES_DIR,
        config.EGG_TRAIN_ADDED_IMAGES_DIR,
        config.EGG_LABELLISATION_IMAGES_DIR,
        config.EGG_CORRECTED_IMAGES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _safe_filename(filename: str) -> str:
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"Nom de fichier non sûr : {filename!r}.")
    return filename


def _validate_store(
    data: object, path: Path, num_classes: int = config.LEGACY_CAVITY_NUM_CLASSES
) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("images"), list) or not isinstance(
        data.get("annotations"), list
    ):
        raise DataStoreError(f"Structure JSON invalide dans {path}.")

    image_ids: set[int] = set()
    for image in data["images"]:
        try:
            image_id = image["id"]
            _safe_filename(image["file_name"])
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataStoreError(f"Entrée image invalide dans {path}: {image!r}.") from exc
        if image_id in image_ids or width <= 0 or height <= 0:
            raise DataStoreError(f"Identifiant ou dimensions d'image invalides dans {path}.")
        image_ids.add(image_id)

    annotation_ids: set[int] = set()
    for annotation in data["annotations"]:
        try:
            annotation_id = annotation["id"]
            image_id = annotation["image_id"]
            category_id = int(annotation["category_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataStoreError(f"Annotation invalide dans {path}: {annotation!r}.") from exc
        if annotation_id in annotation_ids:
            raise DataStoreError(f"Identifiant d'annotation dupliqué dans {path}.")
        if image_id not in image_ids:
            raise DataStoreError(f"Annotation orpheline (image_id={image_id}) dans {path}.")
        if not 1 <= category_id <= num_classes:
            raise DataStoreError(f"Catégorie {category_id} inconnue dans {path}.")
        annotation_ids.add(annotation_id)
    return data


def load_added_annotations(
    path: Path | None = None, num_classes: int = config.LEGACY_CAVITY_NUM_CLASSES
) -> dict:
    """Charge et valide un magasin d'annotations géré par l'application."""
    path = Path(path or config.TRAIN_ADDED_ANN_PATH)
    _ensure_dirs()
    if not path.exists():
        return {"images": [], "annotations": []}
    try:
        with path.open("r", encoding="utf-8") as stream:
            return _validate_store(json.load(stream), path, num_classes=num_classes)
    except DataStoreError:
        logger.exception("Magasin d'annotations invalide", extra={"path": str(path)})
        raise
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Impossible de lire les annotations", extra={"path": str(path)})
        raise DataStoreError(f"Impossible de lire {path}: {exc}.") from exc


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except OSError:
        logger.exception("Écriture atomique impossible", extra={"path": str(path)})
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _save_added_annotations(
    data: dict,
    path: Path | None = None,
    num_classes: int = config.LEGACY_CAVITY_NUM_CLASSES,
) -> None:
    path = Path(path or config.TRAIN_ADDED_ANN_PATH)
    _validate_store(data, path, num_classes=num_classes)
    serialized = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(path, serialized)


def list_added_filenames(path: Path | None = None) -> list:
    return [image["file_name"] for image in load_added_annotations(path)["images"]]


def list_labellisation_filenames() -> list:
    return list_added_filenames(config.LABELLISATION_ANN_PATH)


def list_corrected_filenames() -> list:
    return list_added_filenames(config.CORRECTED_ANN_PATH)


def list_egg_added_filenames() -> list:
    return list_added_filenames(config.EGG_TRAIN_ADDED_ANN_PATH)


def get_disk_status(filename: str, image_type: str = "cavite") -> str | None:
    """Retourne le statut persistant de l'image, ou ``None`` si elle est nouvelle."""
    _safe_filename(filename)
    if image_type not in {"cavite", "oeufs"}:
        raise ValueError(f"Type d'image inconnu : {image_type!r}.")
    train_path = (
        config.EGG_TRAIN_ADDED_ANN_PATH
        if image_type == "oeufs"
        else config.TRAIN_ADDED_ANN_PATH
    )
    label_path = (
        config.EGG_LABELLISATION_ANN_PATH
        if image_type == "oeufs"
        else config.LABELLISATION_ANN_PATH
    )
    if filename in list_added_filenames(train_path):
        return "bonne"
    if filename in list_added_filenames(label_path):
        return "mauvaise"
    return None


def _normalize_masks(
    masks: dict, image_size: tuple[int, int], class_names: list[str] | None = None
) -> dict[str, np.ndarray]:
    width, height = image_size
    class_names = class_names or config.CLASS_NAMES
    normalized: dict[str, np.ndarray] = {}
    for class_name in class_names:
        if class_name not in masks:
            raise ValueError(f"Masque manquant pour la classe {class_name!r}.")
        mask = np.asarray(masks[class_name], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                f"Masque {class_name!r} de forme {mask.shape}, attendu {(height, width)}."
            )
        normalized[class_name] = mask
    return normalized


def _encode_masks(
    masks: dict, image_id: int, start_ann_id: int, class_names: list[str] | None = None
) -> list:
    class_names = class_names or config.CLASS_NAMES
    annotations = []
    ann_id = start_ann_id
    for class_idx, class_name in enumerate(class_names):
        binary_mask = masks[class_name]
        if not binary_mask.any():
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


def _image_bytes(image: Image.Image, filename: str) -> bytes:
    image_format = Image.registered_extensions().get(Path(filename).suffix.lower())
    if image_format is None:
        raise ValueError(f"Extension d'image non prise en charge : {Path(filename).suffix!r}.")
    output = io.BytesIO()
    image.convert("RGB").save(output, format=image_format)
    return output.getvalue()


def _add_image(
    filename: str,
    image: Image.Image,
    masks: dict,
    image_dir: Path,
    ann_path: Path,
    class_names: list[str] | None = None,
) -> None:
    filename = _safe_filename(filename)
    class_names = class_names or config.CLASS_NAMES
    store_num_classes = (
        config.LEGACY_CAVITY_NUM_CLASSES
        if class_names == config.CLASS_NAMES
        else len(class_names)
    )
    normalized_masks = _normalize_masks(masks, image.size, class_names)
    image_path = image_dir / filename
    try:
        data = load_added_annotations(ann_path, num_classes=store_num_classes)
        previous = next((item for item in data["images"] if item["file_name"] == filename), None)
        if previous is not None:
            previous_id = previous["id"]
            data["images"] = [item for item in data["images"] if item["id"] != previous_id]
            data["annotations"] = [
                item for item in data["annotations"] if item["image_id"] != previous_id
            ]

        image_id = max((item["id"] for item in data["images"]), default=0) + 1
        next_ann_id = max((item["id"] for item in data["annotations"]), default=0) + 1
        width, height = image.size
        data["images"].append(
            {"id": image_id, "file_name": filename, "width": width, "height": height}
        )
        data["annotations"].extend(
            _encode_masks(normalized_masks, image_id, next_ann_id, class_names)
        )

        # Préparer et valider les deux contenus avant le premier remplacement.
        encoded_image = _image_bytes(image, filename)
        _validate_store(data, ann_path, num_classes=store_num_classes)
        previous_image = image_path.read_bytes() if image_path.exists() else None
        _atomic_write_bytes(image_path, encoded_image)
        try:
            _save_added_annotations(data, ann_path, num_classes=store_num_classes)
        except Exception:
            # Le JSON encore en place référence l'ancienne image. On la restaure
            # donc si le second remplacement échoue.
            if previous_image is None:
                image_path.unlink(missing_ok=True)
            else:
                _atomic_write_bytes(image_path, previous_image)
            raise
    except (OSError, ValueError, KeyError, DataStoreError) as exc:
        logger.exception(
            "Impossible d'enregistrer une image annotée",
            extra={"image_name": filename, "annotation_path": str(ann_path)},
        )
        if isinstance(exc, (ValueError, DataStoreError)):
            raise
        raise DataStoreError(f"Impossible d'enregistrer {filename}: {exc}.") from exc


def add_image_to_training_set(
    filename: str, image: Image.Image, masks: dict, image_type: str = "cavite"
) -> None:
    if image_type == "oeufs":
        _add_image(
            filename,
            image,
            masks,
            config.EGG_TRAIN_ADDED_IMAGES_DIR,
            config.EGG_TRAIN_ADDED_ANN_PATH,
            config.EGG_CLASS_NAMES,
        )
    elif image_type == "cavite":
        _add_image(
            filename, image, masks, config.TRAIN_ADDED_IMAGES_DIR, config.TRAIN_ADDED_ANN_PATH
        )
    else:
        raise ValueError(f"Type d'image inconnu : {image_type!r}.")


def add_image_to_labellisation_set(
    filename: str, image: Image.Image, masks: dict, image_type: str = "cavite"
) -> None:
    if image_type == "oeufs":
        _add_image(
            filename,
            image,
            masks,
            config.EGG_LABELLISATION_IMAGES_DIR,
            config.EGG_LABELLISATION_ANN_PATH,
            config.EGG_CLASS_NAMES,
        )
    elif image_type == "cavite":
        _add_image(
            filename, image, masks, config.LABELLISATION_IMAGES_DIR, config.LABELLISATION_ANN_PATH
        )
    else:
        raise ValueError(f"Type d'image inconnu : {image_type!r}.")


def add_image_to_corrected_set(
    filename: str, image: Image.Image, masks: dict, image_type: str = "cavite"
) -> None:
    if image_type == "oeufs":
        _add_image(
            filename,
            image,
            masks,
            config.EGG_CORRECTED_IMAGES_DIR,
            config.EGG_CORRECTED_ANN_PATH,
            config.EGG_CLASS_NAMES,
        )
    elif image_type == "cavite":
        _add_image(filename, image, masks, config.CORRECTED_IMAGES_DIR, config.CORRECTED_ANN_PATH)
    else:
        raise ValueError(f"Type d'image inconnu : {image_type!r}.")


def load_corrected_image_and_masks(filename: str, image_type: str = "cavite") -> dict | None:
    """Recharge une correction et vérifie l'image ainsi que tous ses RLE."""
    filename = _safe_filename(filename)
    if image_type == "oeufs":
        ann_path = config.EGG_CORRECTED_ANN_PATH
        image_dir = config.EGG_CORRECTED_IMAGES_DIR
        class_names = config.EGG_CLASS_NAMES
    elif image_type == "cavite":
        ann_path = config.CORRECTED_ANN_PATH
        image_dir = config.CORRECTED_IMAGES_DIR
        class_names = config.CLASS_NAMES
    else:
        raise ValueError(f"Type d'image inconnu : {image_type!r}.")
    store_num_classes = (
        config.LEGACY_CAVITY_NUM_CLASSES
        if class_names == config.CLASS_NAMES
        else len(class_names)
    )
    data = load_added_annotations(ann_path, num_classes=store_num_classes)
    image_entry = next((item for item in data["images"] if item["file_name"] == filename), None)
    if image_entry is None:
        return None

    image_path = image_dir / filename
    try:
        if not image_path.is_file():
            raise DataStoreError(f"Image corrigée manquante : {image_path}.")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            image.load()
        expected_size = (int(image_entry["width"]), int(image_entry["height"]))
        if image.size != expected_size:
            raise DataStoreError(
                f"Dimensions de {filename} incohérentes : {image.size}, attendu {expected_size}."
            )

        width, height = expected_size
        masks = {name: np.zeros((height, width), dtype=bool) for name in class_names}
        for annotation in data["annotations"]:
            if annotation["image_id"] != image_entry["id"]:
                continue
            class_idx = int(annotation["category_id"]) - 1
            if class_idx >= len(class_names):
                continue
            class_name = class_names[class_idx]
            segmentation = annotation.get("segmentation")
            if (
                not isinstance(segmentation, dict)
                or "counts" not in segmentation
                or "size" not in segmentation
            ):
                raise DataStoreError(f"RLE invalide pour {filename}, classe {class_name}.")
            rle = dict(segmentation)
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")
            decoded = np.asarray(mask_util.decode(rle), dtype=bool)
            if decoded.shape != (height, width):
                raise DataStoreError(
                    f"RLE de forme {decoded.shape} pour {filename}, attendu {(height, width)}."
                )
            masks[class_name] |= decoded
        return {"image": image, "masks": masks}
    except DataStoreError:
        logger.exception("Correction invalide", extra={"image_name": filename})
        raise
    except (OSError, TypeError, ValueError, KeyError) as exc:
        logger.exception("Impossible de charger la correction", extra={"image_name": filename})
        raise DataStoreError(f"Impossible de charger la correction de {filename}: {exc}.") from exc


def load_coco_ground_truth_masks(
    filename: str, image_type: str = "cavite"
) -> dict | None:
    """Charge les masques COCO d'origine du type demandé, si l'image est présente."""
    filename = _safe_filename(filename)
    if image_type == "oeufs":
        ann_path = config.EGG_COCO_ANN_PATH
        class_names = config.EGG_CLASS_NAMES
    elif image_type == "cavite":
        ann_path = config.COCO_ANN_PATH
        class_names = config.CLASS_NAMES
    else:
        raise ValueError(f"Type d'image inconnu : {image_type!r}.")

    if not ann_path.exists():
        return None
    with ann_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    image_entry = next((item for item in data["images"] if item["file_name"] == filename), None)
    if image_entry is None:
        return None

    image_id = image_entry["id"]
    height, width = image_entry["height"], image_entry["width"]
    masks = {name: np.zeros((height, width), dtype=bool) for name in class_names}
    for annotation in data["annotations"]:
        if annotation["image_id"] != image_id:
            continue
        class_idx = annotation["category_id"] - 1
        if not 0 <= class_idx < len(class_names):
            continue
        decoded = decode_segmentation(annotation["segmentation"], height, width)
        masks[class_names[class_idx]] |= np.asarray(decoded, dtype=bool)
    return masks


def _remove_image(filename: str, image_dir: Path, ann_path: Path) -> None:
    filename = _safe_filename(filename)
    try:
        data = load_added_annotations(ann_path)
        image_entry = next((item for item in data["images"] if item["file_name"] == filename), None)
        if image_entry is None:
            return
        image_id = image_entry["id"]
        data["images"] = [item for item in data["images"] if item["id"] != image_id]
        data["annotations"] = [
            item for item in data["annotations"] if item["image_id"] != image_id
        ]
        _save_added_annotations(data, ann_path)
        image_path = image_dir / filename
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            # Un fichier orphelin n'est plus consommé puisque le JSON fait foi.
            logger.warning(
                "Annotation supprimée mais image orpheline impossible à retirer",
                exc_info=True,
                extra={"path": str(image_path)},
            )
    except (OSError, ValueError, KeyError, DataStoreError) as exc:
        logger.exception(
            "Impossible de supprimer une image annotée",
            extra={"image_name": filename, "annotation_path": str(ann_path)},
        )
        if isinstance(exc, (ValueError, DataStoreError)):
            raise
        raise DataStoreError(f"Impossible de supprimer {filename}: {exc}.") from exc


def remove_image_from_training_set(filename: str, image_type: str = "cavite") -> None:
    if image_type == "oeufs":
        _remove_image(
            filename, config.EGG_TRAIN_ADDED_IMAGES_DIR, config.EGG_TRAIN_ADDED_ANN_PATH
        )
    elif image_type == "cavite":
        _remove_image(filename, config.TRAIN_ADDED_IMAGES_DIR, config.TRAIN_ADDED_ANN_PATH)
    else:
        raise ValueError(f"Type d'image inconnu : {image_type!r}.")


def remove_image_from_labellisation_set(filename: str, image_type: str = "cavite") -> None:
    if image_type == "oeufs":
        _remove_image(
            filename, config.EGG_LABELLISATION_IMAGES_DIR, config.EGG_LABELLISATION_ANN_PATH
        )
    elif image_type == "cavite":
        _remove_image(filename, config.LABELLISATION_IMAGES_DIR, config.LABELLISATION_ANN_PATH)
    else:
        raise ValueError(f"Type d'image inconnu : {image_type!r}.")


def remove_image_from_corrected_set(filename: str, image_type: str = "cavite") -> None:
    if image_type == "oeufs":
        _remove_image(filename, config.EGG_CORRECTED_IMAGES_DIR, config.EGG_CORRECTED_ANN_PATH)
    elif image_type == "cavite":
        _remove_image(filename, config.CORRECTED_IMAGES_DIR, config.CORRECTED_ANN_PATH)
    else:
        raise ValueError(f"Type d'image inconnu : {image_type!r}.")
