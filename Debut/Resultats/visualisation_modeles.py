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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.stats_fct import overlay_colored_mask, overlay_mask


# ============================ 2. Définition des chemins et paramètres ============================
UNET_ROOT = PROJECT_ROOT / 'UNet' / 'masks' / 'eval'
YOLO_ROOT = PROJECT_ROOT / 'YOLO26' / 'masks' / 'eval'
IMAGE_ROOT = PROJECT_ROOT / 'data' / 'images'
LABEL_ROOT = PROJECT_ROOT / 'data' / 'labels' / 'YOLO'
OUTPUT_ROOT = PROJECT_ROOT / 'Resultats' / 'visualisation'

TASKS = {
    'oeufs': {'num_classes': 1, 'image_dir': 'oeufs'},
    '3classes': {'num_classes': 3, 'image_dir': 'cavite'},
}

# ============================ 3. Visualisation des résultats ============================
# Forme de la grille de visualisation. Les images et labels ont déjà été cropés.
COLS = 3
IMAGE_SIZE = (510, 380)

def image_id_from_prediction(mask_path):
    match = re.fullmatch(r'pred_fold_\d+_(.+)\.txt', mask_path.name)
    if match is not None:
        return match.group(1)
    match = re.fullmatch(r'pred_(.+)\.txt', mask_path.name)
    if match is not None:
        return match.group(1)
    raise ValueError(f'Nom de prédiction inattendu : {mask_path.name}')

def mask_from_yolo_file(mask_path, num_classes, width, height,
                        preserve_instances=False):
    """Rasterise une prédiction normalisée dans le repère du crop."""
    mask_shape = (height, width) if preserve_instances else (
        num_classes, height, width
    )
    mask_dtype = np.int32 if preserve_instances else np.float32
    mask = np.zeros(mask_shape, dtype=mask_dtype)
    instance_id = 0
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
            if np.any((coords < 0) | (coords > 1)):
                raise ValueError(
                    f'Coordonnées YOLO hors de [0, 1] dans {mask_path}'
                )
            coords[:, 0] *= width
            coords[:, 1] *= height
            if preserve_instances:
                instance_id += 1
                cv2.fillPoly(
                    mask, [np.rint(coords).astype(np.int32)], instance_id
                )
                continue
            cv2.fillPoly(
                mask[class_id], [np.rint(coords).astype(np.int32)], 1.0
            )
    return torch.from_numpy(mask)

def mask_from_source_label(label_path, num_classes, image_size,
                           preserve_instances=False):
    """Charge un label YOLO déjà exprimé dans le repère de l'image cropée."""
    width, height = image_size
    mask_shape = (height, width) if preserve_instances else (
        num_classes, height, width
    )
    mask_dtype = np.int32 if preserve_instances else np.float32
    mask = np.zeros(mask_shape, dtype=mask_dtype)
    instance_id = 0

    if not label_path.exists():
        return torch.from_numpy(mask)

    with label_path.open('r') as file:
        for line in file:
            if not line.strip():
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 3 or len(parts[1:]) % 2:
                raise ValueError(f'Polygone YOLO invalide dans {label_path}')

            class_id = int(parts[0])
            if not 0 <= class_id < num_classes:
                continue

            coords = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= width
            coords[:, 1] *= height
            if preserve_instances:
                instance_id += 1
                cv2.fillPoly(
                    mask, [np.rint(coords).astype(np.int32)], instance_id
                )
                continue
            cv2.fillPoly(mask[class_id], [np.rint(coords).astype(np.int32)], 1.0)

    return torch.from_numpy(mask)


def overlay_egg_instances(image_tensor, instance_mask, alpha=0.4,
                          target_size=None):
    """Colore chaque œuf sans dessiner de contour."""
    return overlay_colored_mask(
        image_tensor, instance_mask, alpha=alpha, dataset_name='oeufs',
        target_size=target_size
    )


def prediction_sources(task_name):
    """Retourne les dossiers de prédictions à comparer pour une tâche.

    Les sorties UNet sont organisées comme ``eval/<modele>/<tache>`` tandis
    que les sorties YOLO sont organisées comme ``eval/<type>/<tache>``.
    """
    sources = []

    if UNET_ROOT.is_dir():
        for model_dir in sorted(UNET_ROOT.iterdir()):
            task_dir = model_dir / task_name
            if model_dir.is_dir() and task_dir.is_dir():
                sources.append((model_dir.name, task_dir))

    for yolo_type in ('instance', 'semantique'):
        task_dir = YOLO_ROOT / yolo_type / task_name
        if task_dir.is_dir():
            sources.append((f'YOLO_{yolo_type}', task_dir))

    return sources

for task_name, task_config in TASKS.items():
    num_classes = task_config['num_classes']
    preserve_instances = task_name == 'oeufs'
    image_dir = IMAGE_ROOT / task_config['image_dir']
    output_dir = OUTPUT_ROOT / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Indexe toutes les prédictions par image et par modèle.
    predictions_by_image = defaultdict(dict)
    sources = prediction_sources(task_name)
    for source_name, source_dir in sources:
        for mask_path in sorted(source_dir.glob('pred_*.txt')):
            image_id = image_id_from_prediction(mask_path)
            predictions_by_image[image_id][source_name] = mask_path

    print(f'Sauvegarde des résultats pour {task_name}: {len(sources)} modèles, '
          f'{len(predictions_by_image)} images')

    for image_id in sorted(predictions_by_image):
        image_path = image_dir / f'{image_id}.jpg'
        if not image_path.exists():
            print(f'Image source absente, visualisation ignorée : {image_path}')
            continue

        image = Image.open(image_path).convert('RGB')
        if image.size != IMAGE_SIZE:
            print(f'Image hors format cropé, visualisation ignorée : {image_path}')
            continue
        width, height = image.size
        image_tensor = TF.to_tensor(image)

        label_path = LABEL_ROOT / task_config['image_dir'] / f'{image_id}.txt'
        true_mask_tensor = mask_from_source_label(
            label_path, num_classes, image.size,
            preserve_instances=preserve_instances
        )
        if label_path.exists():
            overlay_function = (
                overlay_egg_instances if preserve_instances else overlay_mask
            )
            true_display = overlay_function(
                image_tensor, true_mask_tensor, alpha=0.4,
                target_size=image.size
            )
        else:
            true_display = None

        displays = []
        model_names = []
        for source_name, _ in sources:
            mask_path = predictions_by_image[image_id].get(source_name)
            if mask_path is None:
                continue
            mask_tensor = mask_from_yolo_file(
                mask_path, num_classes, width, height,
                preserve_instances=preserve_instances
            )
            overlay_function = (
                overlay_egg_instances if preserve_instances else overlay_mask
            )
            displays.append(
                overlay_function(image_tensor, mask_tensor, alpha=0.4,
                                 target_size=image.size)
            )
            model_names.append(source_name)

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
                f'verite_terrain', fontsize=11,
                fontweight='bold', pad=8
            )
            axes_flat[1].axis('off')
            first_model_index = 2

        for index, (display_image, model_name) in enumerate(
            zip(displays, model_names), start=first_model_index
        ):
            axes_flat[index].imshow(display_image)
            axes_flat[index].set_title(
                f'{model_name}', fontsize=11,
                fontweight='bold', pad=8
            )
            axes_flat[index].axis('off')

        for axis in axes_flat[num_panels:]:
            axis.axis('off')

        fig.tight_layout()
        fig.savefig(output_dir / f'combined_{image_id}.png',
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
