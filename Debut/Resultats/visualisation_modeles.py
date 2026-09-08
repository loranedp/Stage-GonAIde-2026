# SCRIPT POUR LANCER LA VISUALISATION SIMULTANEE DES RESULTATS DES DIFFERENTS MODELES

# =========================== 1. Importation des bibliothèques ============================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import cv2
import sys
from PIL import Image
import torch
import re
from collections import defaultdict
import torchvision.transforms.functional as TF

# Résout la racine du projet à partir de ce fichier, indépendamment du
# répertoire depuis lequel le script est lancé.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.stats_fct import overlay_mask


# ============================ 2. Définition des chemins et paramètres ============================
EVAL_ROOT = PROJECT_ROOT / 'UNet' / 'masks' / 'eval'
IMAGE_ROOT = PROJECT_ROOT / 'data' / 'images'
OUTPUT_ROOT = PROJECT_ROOT / 'Resultats' / 'visualisation'

TASKS = {
    'oeufs': {'num_classes': 1, 'image_dir': 'oeufs'},
    '3classes': {'num_classes': 3, 'image_dir': 'cavite'},
}

# ============================ 3. Visualisation des résultats ============================
COLS = 3
CROP = (85, 33, 510, 380)

def image_id_from_prediction(mask_path):
    match = re.fullmatch(r'pred_fold_\d+_(.+)\.txt', mask_path.name)
    if match is None:
        raise ValueError(f'Nom de prédiction inattendu : {mask_path.name}')
    return match.group(1)

def mask_from_yolo_file(mask_path, num_classes, width, height):
    mask = np.zeros((num_classes, height, width), dtype=np.float32)
    with mask_path.open('r') as file:
        for line in file:
            if not line.strip():
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 3 or len(parts[1:]) % 2:
                raise ValueError(f'Polygone YOLO invalide dans {mask_path}')

            class_id = int(parts[0])
            if not 0 <= class_id < num_classes:
                continue

            coords = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= width
            coords[:, 1] *= height
            cv2.fillPoly(mask[class_id], [coords.astype(np.int32)], 1.0)
    return torch.from_numpy(mask)

for task_name, task_config in TASKS.items():
    num_classes = task_config['num_classes']
    image_dir = IMAGE_ROOT / task_config['image_dir']
    output_dir = OUTPUT_ROOT / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Indexe toutes les prédictions par image et par modèle.
    predictions_by_image = defaultdict(dict)
    model_dirs = sorted(
        path for path in EVAL_ROOT.iterdir()
        if path.is_dir() and (path / task_name).is_dir()
    )
    for model_dir in model_dirs:
        for mask_path in sorted((model_dir / task_name).glob('pred_*.txt')):
            image_id = image_id_from_prediction(mask_path)
            predictions_by_image[image_id][model_dir.name] = mask_path

    # La vérité terrain est commune aux modèles : on ne conserve qu'un
    # fichier par image pour l'afficher une seule fois dans la grille.
    true_masks_by_image = {}
    for model_dir in model_dirs:
        for mask_path in sorted((model_dir / task_name).glob('true_*.txt')):
            match = re.fullmatch(r'true_fold_\d+_(.+)\.txt', mask_path.name)
            if match is None:
                raise ValueError(f'Nom de vérité terrain inattendu : {mask_path.name}')
            true_masks_by_image.setdefault(match.group(1), mask_path)

    print(f'Sauvegarde des résultats pour {task_name}: {len(model_dirs)} modèles, '
          f'{len(predictions_by_image)} images')

    for image_id in sorted(predictions_by_image):
        image_path = image_dir / f'{image_id}.jpg'
        if not image_path.exists():
            print(f'Image source absente, visualisation ignorée : {image_path}')
            continue

        x, y, width, height = CROP
        original_image = Image.open(image_path).convert('RGB')
        image = original_image.crop((x, y, x + width, y + height))
        image_tensor = TF.to_tensor(image)

        true_display = None
        true_mask_path = true_masks_by_image.get(image_id)
        if true_mask_path is not None:
            true_mask_tensor = mask_from_yolo_file(
                true_mask_path, num_classes, width, height
            )
            true_display = overlay_mask(
                image_tensor, true_mask_tensor, alpha=0.4,
                target_size=image.size
            )

        displays = []
        model_names = []
        for model_dir in model_dirs:
            mask_path = predictions_by_image[image_id].get(model_dir.name)
            if mask_path is None:
                continue
            mask_tensor = mask_from_yolo_file(
                mask_path, num_classes, width, height
            )
            displays.append(
                overlay_mask(image_tensor, mask_tensor, alpha=0.4,
                             target_size=image.size)
            )
            model_names.append(model_dir.name)

        num_panels = len(displays) + 1 + (true_display is not None)
        rows = (num_panels + COLS - 1) // COLS
        fig, axes = plt.subplots(rows, COLS, figsize=(5 * COLS, 5 * rows))
        axes_flat = np.asarray(axes).reshape(-1)

        axes_flat[0].imshow(image)
        axes_flat[0].set_title(f'{image_id}_original',
                              fontsize=11, fontweight='bold', pad=8)
        axes_flat[0].axis('off')

        first_model_index = 1
        if true_display is not None:
            axes_flat[1].imshow(true_display)
            axes_flat[1].set_title(
                f'{image_id}_verite_terrain', fontsize=11,
                fontweight='bold', pad=8
            )
            axes_flat[1].axis('off')
            first_model_index = 2

        for index, (display_image, model_name) in enumerate(
            zip(displays, model_names), start=first_model_index
        ):
            axes_flat[index].imshow(display_image)
            axes_flat[index].set_title(
                f'{image_id}_{model_name}', fontsize=11,
                fontweight='bold', pad=8
            )
            axes_flat[index].axis('off')

        for axis in axes_flat[num_panels:]:
            axis.axis('off')

        fig.tight_layout()
        fig.savefig(output_dir / f'combined_{image_id}.png',
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
