#================ Création et transformation du dataset ================

import sys
import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mask_util
from pathlib import Path
import pandas as pd

sys.path.append(str(Path.cwd().parent)) # Ajoute le dossier parent au chemin de recherche de Python

from utils.data_augmentation_fct import data_augmentation
from utils.conditional_metadata_embedding import CME
from UNet.models.EggSegmentationHVUNet import generate_hv_targets


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), str):
            rle = {"counts": rle["counts"].encode("utf-8"), "size": rle["size"]}
        return mask_util.decode(rle)
    rles = mask_util.frPyObjects(segmentation, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle)


class RoboflowUNetDataset(torch.utils.data.Dataset):
    def __init__(self, samples, image_dir, df, image_size, num_classes=3, class_ids=[1, 2, 3], crop_params=[85, 33, 510, 380], is_train=False, is_cme = False, seed=42,
                 ignore_category_id=None, ignore_target_class_idx=None, ignore_value=-100.0):
        """
        Args:
            samples (list): Liste des dictionnaires d'échantillons contenant l'image et ses annotations.
            image_dir (str): Chemin vers le dossier unique contenant toutes les images.
            image_size (tuple): Taille finale (H, W).
            num_classes (int): Nombre de classes.
            class_ids (list): Liste des IDs de catégories COCO valides.
            crop_params (list): Paramètres de crop [x, y, w, h] pour recadrer les images et masques.
            ignore_category_id (int, optionnel): ID COCO de la catégorie d'annotation "ignore"
                (ex: Gonade_ignore). N'est jamais un canal du masque : sert uniquement à marquer
                des pixels à exclure (loss/métriques) sur le canal `ignore_target_class_idx`.
            ignore_target_class_idx (int, optionnel): index du canal (parmi `class_ids`) sur lequel
                appliquer la valeur sentinelle `ignore_value` aux pixels couverts par `ignore_category_id`.
            ignore_value (float): valeur sentinelle écrite dans le masque pour les pixels ignorés
                (doit correspondre à l'`ignore_index` de la loss utilisée, ex. `BCE_DiceLoss_ignore`).
        """
        self.samples = samples
        self.image_dir = image_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_ids = class_ids
        self.crop_params = crop_params
        self.df = df
        self.is_train = is_train
        self.is_cme = is_cme
        self.seed = seed  # seed dédié à l'augmentation, indépendant du modèle testé
        self.epoch = 0    # mis à jour par la boucle d'entraînement à chaque epoch
        self.ignore_category_id = ignore_category_id
        self.ignore_target_class_idx = ignore_target_class_idx
        self.ignore_value = ignore_value

        self.one_hot_df = pd.get_dummies(self.df["categorie"], dtype=int)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # -------- 1. Selection d'une image --------
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")

        original_size = image.size
        w_orig, h_orig = original_size

        # -------- 2. Création d'un masque --------
        # Initialiser un masque multi-canal vide
        mask_np = np.zeros((self.num_classes, h_orig, w_orig), dtype=np.uint8)

        # --- Transformer les annotations en masques binaires pour chaque classe ---
        for ann in sample['annotations']:
            cat_id = ann['category_id']
            if cat_id not in self.class_ids: # Vérifie que appartient à une des classes
                continue
            class_idx = self.class_ids.index(cat_id)
            seg = ann['segmentation'] # Extrait la segmentation de l'objet

            # Transformation de la segmentation en masque binaire (RLE ou polygone)
            ann_mask = decode_segmentation(seg, h_orig, w_orig)

            # Fusion des masques qui ont la même classe
            mask_np[class_idx] = np.maximum(mask_np[class_idx], ann_mask)

        # --- Masque binaire "ignore" (ex: Gonade_ignore) : n'est jamais un canal du masque,
        # sert seulement à marquer des pixels à exclure sur `ignore_target_class_idx` ---
        ignore_np = np.zeros((h_orig, w_orig), dtype=np.uint8)
        if self.ignore_category_id is not None:
            for ann in sample['annotations']:
                if ann['category_id'] != self.ignore_category_id:
                    continue
                ann_mask = decode_segmentation(ann['segmentation'], h_orig, w_orig)
                ignore_np = np.maximum(ignore_np, ann_mask)

        # -------- 3. Pré-traitement des données --------
        # Application du crop
        if self.crop_params is not None:
            x, y, w, h = self.crop_params
            image = image.crop((x, y, x + w, y + h))
            mask_np = mask_np[:, y:y+h, x:x+w]
            ignore_np = ignore_np[y:y+h, x:x+w]

        # Redimensionnement de l'image
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)

        # Normalisation de l'image
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_torch = torch.as_tensor(image_np).permute(2, 0, 1).contiguous()

        # Redimensionnement des masques
        processed_masks = []
        for c in range(self.num_classes):
            mask_pil = Image.fromarray(mask_np[c])
            mask_pil = mask_pil.resize(self.image_size, Image.Resampling.NEAREST)
            processed_masks.append(np.array(mask_pil, dtype=np.float32))

        mask_torch = torch.as_tensor(np.stack(processed_masks, axis=0))

        # Redimensionnement du masque "ignore" (même traitement NEAREST que les masques de classe)
        ignore_pil = Image.fromarray(ignore_np)
        ignore_pil = ignore_pil.resize(self.image_size, Image.Resampling.NEAREST)
        ignore_torch = torch.as_tensor(np.array(ignore_pil, dtype=np.float32)).unsqueeze(0)

        # CME : Ajout de canaux RBG
        if self.is_cme:
            image_torch = CME(sample['image_path'], image_torch, self.df, self.one_hot_df)

        # Ajout de la Data Augmentation
        # Seed déterministe par (seed du dataset, epoch, idx) : l'augmentation appliquée
        # à un échantillon donné à une epoch donnée est ainsi indépendante de tout ce qui
        # a pu consommer le générateur aléatoire global avant (init du modèle, dropout, ...),
        # ce qui garantit la même augmentation quel que soit le modèle comparé.
        if self.is_train:
            img = image_torch
            # Le canal "ignore" est empilé avec les masques de classe le temps de l'augmentation,
            # pour subir exactement la même rotation/translation/crop (data_augmentation est
            # générique sur le nombre de canaux, aucune modification nécessaire côté augmentation).
            msk = torch.cat([mask_torch, ignore_torch], dim=0)

            item_seed = (self.seed * 1_000_003 + self.epoch * 10_007 + idx) % (2**31 - 1)
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(item_seed)
            augmented_img, augmented_msk = data_augmentation(img, msk)
            torch.random.set_rng_state(rng_state)

            image_torch = augmented_img
            mask_torch = augmented_msk[:self.num_classes]
            ignore_torch = augmented_msk[self.num_classes:]

        # Applique la sentinelle "ignore" uniquement sur le canal ciblé (ex: Gonade),
        # après tout redimensionnement/augmentation pour rester alignée spatialement.
        if self.ignore_category_id is not None and self.ignore_target_class_idx is not None:
            ignore_bool = ignore_torch[0] > 0.5
            target = mask_torch[self.ignore_target_class_idx]
            mask_torch[self.ignore_target_class_idx] = torch.where(
                ignore_bool, torch.full_like(target, self.ignore_value), target
            )

        return {
            'image': image_torch,
            'mask': mask_torch,
            'original_size': original_size,
            'crop_params': self.crop_params,
            'image_path': sample['image_path']
        }


class EggHVDataset(torch.utils.data.Dataset):
    """Dataset COCO d'instances d'œufs pour la sortie masque/H/V."""
    def __init__(self, samples, image_size, crop_params=(85, 33, 510, 380),
                 is_train=False, seed=42):
        self.samples = samples
        self.image_size = image_size
        self.crop_params = crop_params
        self.is_train = is_train
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        from utils.dataset_unet import decode_segmentation
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        width, height = image.size
        instances = []
        for ann in sample['annotations']:
            if ann.get('category_id') != 1:
                continue
            instances.append(decode_segmentation(ann['segmentation'], height, width))
        mask, horizontal, vertical = generate_hv_targets(instances, height, width)
        x, y, w, h = self.crop_params
        image = image.crop((x, y, x + w, y + h))
        targets = np.stack([mask[y:y+h, x:x+w], horizontal[y:y+h, x:x+w], vertical[y:y+h, x:x+w]])
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        targets_tensor = torch.from_numpy(np.stack([np.asarray(Image.fromarray(channel).resize(self.image_size, Image.Resampling.BILINEAR), dtype=np.float32) for channel in targets]))
        targets_tensor[0].clamp_(0, 1)
        targets_tensor[1:].clamp_(-1, 1)

        if self.is_train:
            item_seed = (
                self.seed * 1_000_003 + self.epoch * 10_007 + idx
            ) % (2**31 - 1)
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(item_seed)
            image_tensor, targets_tensor = data_augmentation(
                image_tensor, targets_tensor, model="egg_hv"
            )
            torch.random.set_rng_state(rng_state)

        return {'image': image_tensor, 'mask': targets_tensor, 'image_path': sample['image_path'], 'original_size': (width, height), 'crop_params': self.crop_params}
