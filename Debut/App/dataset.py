"""Dataset multi-classes + CME pour le ré-entraînement.

Combine toutes les images de data/COCO/images/cavite (via les annotations COCO) et
les images validées "bonne" ajoutées via l'application (data/train_added/).
Les images sans correspondance dans les métadonnées (pour les canaux CME)
sont ignorées."""

import json
import zlib
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config
import metadata_utils
from filename_utils import parse_filename
from utils.dataset_unet import decode_segmentation


def assign_fold(id_poisson: str, num_folds: int) -> int:
    """Attribue le poisson à un groupe de fold de façon déterministe et stable (hash CRC32
    modulo num_folds) : un même poisson (donc toutes ses images) retombe toujours dans le
    même groupe, run après run, ce qui est nécessaire car le ré-entraînement est incrémental
    (nouvelles images ajoutées au fil du temps)."""
    return zlib.crc32(id_poisson.encode("utf-8")) % num_folds


class MultiClassCMEDataset(Dataset):
    def __init__(self, image_size=config.IMAGE_SIZE, crop_params=config.CROP_PARAMS):
        self.image_size = image_size
        self.crop_params = crop_params
        self.samples = []
        self._load_coco_samples()
        self._load_added_samples()
        self._load_corrected_samples()

    def _add_sample(self, image_path, filename, annotations, width, height):
        try:
            parsed = parse_filename(filename)
        except ValueError:
            return
        if metadata_utils.lookup(parsed.id_image) is None:
            return
        self.samples.append(
            {
                "image_path": image_path,
                "id_image": parsed.id_image,
                "id_poisson": parsed.id_poisson,
                "annotations": annotations,
                "width": width,
                "height": height,
            }
        )

    def _load_coco_samples(self):
        if not config.COCO_ANN_PATH.exists():
            return
        with open(config.COCO_ANN_PATH, "r", encoding="utf-8") as f:
            coco = json.load(f)

        anns_by_image = defaultdict(list)
        for ann in coco.get("annotations", []):
            anns_by_image[ann["image_id"]].append(ann)

        for img in coco.get("images", []):
            image_path = config.COCO_IMAGES_DIR / img["file_name"]
            annotations = anns_by_image.get(img["id"], [])
            if not image_path.exists() or not annotations:
                continue
            self._add_sample(image_path, img["file_name"], annotations, img["width"], img["height"])

    def _load_added_samples(self):
        if not config.TRAIN_ADDED_ANN_PATH.exists():
            return
        with open(config.TRAIN_ADDED_ANN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        anns_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_image[ann["image_id"]].append(ann)

        for img in data.get("images", []):
            image_path = config.TRAIN_ADDED_IMAGES_DIR / img["file_name"]
            if not image_path.exists():
                continue
            self._add_sample(
                image_path,
                img["file_name"],
                anns_by_image.get(img["id"], []),
                img["width"],
                img["height"],
            )

    def _load_corrected_samples(self):
        if not config.CORRECTED_ANN_PATH.exists():
            return
        with open(config.CORRECTED_ANN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        anns_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_image[ann["image_id"]].append(ann)

        for img in data.get("images", []):
            image_path = config.CORRECTED_IMAGES_DIR / img["file_name"]
            if not image_path.exists():
                continue
            self._add_sample(
                image_path,
                img["file_name"],
                anns_by_image.get(img["id"], []),
                img["width"],
                img["height"],
            )

    def fold_indices(self, fold_idx: int, num_folds: int) -> tuple:
        """Retourne (indices_train, indices_val) pour le fold fold_idx (0-based) : les
        échantillons sont groupés par id_poisson (cf. assign_fold) afin qu'aucun poisson ne
        se retrouve à la fois en train et en val pour ce fold. Le groupe fold_idx constitue
        la partie held-out (validation) ; tous les autres groupes forment le train."""
        if not (0 <= fold_idx < num_folds):
            raise ValueError(f"fold_idx doit être dans [0, {num_folds}), reçu {fold_idx}.")
        train_indices, val_indices = [], []
        for i, sample in enumerate(self.samples):
            group = assign_fold(sample["id_poisson"], num_folds)
            (val_indices if group == fold_idx else train_indices).append(i)
        return train_indices, val_indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        mask_np = np.zeros((config.NUM_CLASSES, sample["height"], sample["width"]), dtype=np.uint8)
        for ann in sample["annotations"]:
            category_id = ann["category_id"]
            class_idx = category_id - 1  # category_id 1/2/3 -> index 0/1/2
            if not (0 <= class_idx < config.NUM_CLASSES):
                continue
            decoded = decode_segmentation(ann["segmentation"], sample["height"], sample["width"])
            mask_np[class_idx] = np.maximum(mask_np[class_idx], decoded)

        x, y, w, h = self.crop_params
        image = image.crop((x, y, x + w, y + h))
        mask_np = mask_np[:, y : y + h, x : x + w]

        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.as_tensor(image_np).permute(2, 0, 1).contiguous()

        processed_masks = []
        for c in range(config.NUM_CLASSES):
            mask_pil = Image.fromarray(mask_np[c])
            mask_pil = mask_pil.resize(self.image_size, Image.Resampling.NEAREST)
            processed_masks.append(np.array(mask_pil, dtype=np.float32))
        mask_tensor = torch.as_tensor(np.stack(processed_masks, axis=0))

        return image_tensor, mask_tensor
