### Dataset séquentiel par poisson (`SliceSequenceUNetDataset`)

#Regroupe les coupes de `RoboflowUNetDataset` par poisson (`id_poisson`) et les trie par
#position (`position_ratio`, repli sur `position`), pour produire des séquences
#`(T, C, H, W)` / `(T, num_classes, H, W)` consommables par `CSANet` / `xLSTMUNet`.

#- Augmentation appliquée au niveau de la séquence entière (`data_augmentation_sequence`),
#  pas coupe par coupe, pour ne pas casser la correspondance spatiale entre coupes.


import sys
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import segmentation_models_pytorch as smp
from pathlib import Path
import pandas as pd

sys.path.append(str(Path.cwd().parent)) # Ajoute le dossier parent au chemin de recherche de Python

from utils.dataset_unet import RoboflowUNetDataset, data_augmentation_sequence

class SliceSequenceUNetDataset(torch.utils.data.Dataset):
    """
    Regroupe les coupes de RoboflowUNetDataset par poisson (id_poisson), triées par
    position dans la séquence, pour produire des séquences (T, C, H, W) consommées par
    CSA-Net / xLSTM-UNet (_PerSliceUNetBase, cellule 19).
    """

    def __init__(self, samples, image_dir, df, image_size, num_classes=3,
                 class_ids=[1, 2, 3], crop_params=[85, 33, 510, 380],
                 is_train=False, seed=42,
                 ignore_category_id=None, ignore_target_class_idx=None, ignore_value=-100.0):
        # is_train=False forcé : l'augmentation par coupe de RoboflowUNetDataset ne
        # doit jamais s'appliquer ici, l'augmentation est gérée au niveau séquence.
        self._slice_dataset = RoboflowUNetDataset(
            samples=samples, image_dir=image_dir, df=df, image_size=image_size,
            num_classes=num_classes, class_ids=class_ids, crop_params=crop_params,
            is_train=False, seed=seed,
            ignore_category_id=ignore_category_id,
            ignore_target_class_idx=ignore_target_class_idx,
            ignore_value=ignore_value,
        )
        self.samples = samples
        self.is_train = is_train
        self.seed = seed
        self.epoch = 0  # mis à jour manuellement par la boucle d'entraînement, comme RoboflowUNetDataset.epoch

        def _is_missing(v):
            return v is None or (isinstance(v, float) and pd.isna(v))

        # Regroupement par poisson. Edge case : un échantillon sans métadonnées
        # trouvées (cellule 6, branche `else: op=None; op_ratio=None`) n'a aucune
        # position connue : on ne peut pas l'ordonner, donc on l'exclut (avec
        # avertissement) plutôt que de faire échouer tout le pipeline.
        groups = {}
        excluded = 0
        for idx, sample in enumerate(samples):
            if _is_missing(sample.get('position_ratio')) and _is_missing(sample.get('position')):
                excluded += 1
                continue
            groups.setdefault(sample['id_poisson'], []).append(idx)

        if excluded > 0:
            print(f"[SliceSequenceUNetDataset] {excluded} échantillon(s) sans position "
                  f"connue exclu(s) du regroupement par séquence.")

        def _sort_key(idx):
            ratio = samples[idx].get('position_ratio')
            if not _is_missing(ratio):
                return float(ratio)
            return float(samples[idx].get('position'))

        self.fish_ids = sorted(groups.keys())
        self.sequences = [sorted(groups[fid], key=_sort_key) for fid in self.fish_ids]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        slice_indices = self.sequences[idx]

        images, masks = [], []
        original_sizes, crop_params_list, image_paths = [], [], []
        for slice_idx in slice_indices:
            item = self._slice_dataset[slice_idx]
            images.append(item['image'])
            masks.append(item['mask'])
            original_sizes.append(item['original_size'])
            crop_params_list.append(item['crop_params'])
            image_paths.append(item['image_path'])

        images_seq = torch.stack(images, dim=0)  # (T, C, H, W)
        masks_seq = torch.stack(masks, dim=0)    # (T, num_classes, H, W)

        if self.is_train:
            item_seed = (self.seed * 1_000_003 + self.epoch * 10_007 + idx) % (2**31 - 1)
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(item_seed)
            images_seq, masks_seq = data_augmentation_sequence(images_seq, masks_seq)
            torch.random.set_rng_state(rng_state)

        return {
            'image': images_seq,
            'mask': masks_seq,
            'original_size': original_sizes,
            'crop_params': crop_params_list,
            'image_path': image_paths,
            'id_poisson': self.fish_ids[idx],
        }

    