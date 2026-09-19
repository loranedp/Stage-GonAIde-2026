# SCRIPT POUR LANCER LA VISUALISATION SIMULTANEE DES RESULTATS DES DIFFERENTS MODELES

# =========================== 1. Importation des bibliothèques ============================
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import cv2
import sys
from PIL import Image
import re
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================ 2. Définition des chemins et paramètres ============================
UNET_ROOT = PROJECT_ROOT / 'UNet' / 'masks' / 'eval'
YOLO_ROOT = PROJECT_ROOT / 'YOLO26' / 'masks' / 'eval'
OPENUS_CV_ROOT = (PROJECT_ROOT / 'Ex1' / 'OpenUS-multiclass' / 'OpenUS'
                  / 'output' / 'custom_seg_cv5')
IMAGE_ROOT = PROJECT_ROOT / 'data' / 'YOLO' / 'images'
LABEL_ROOT = PROJECT_ROOT / 'data' / 'YOLO' / 'labels'
OUTPUT_ROOT = PROJECT_ROOT / 'Resultats' / 'visualisation'

TASKS = {
    'oeufs': {'num_classes': 1, 'image_dir': 'oeufs'},
    '3classes': {'num_classes': 3, 'image_dir': 'cavite'},
}

# ============================ 3. Visualisation des résultats ============================
# Format des images visualisées. Les images et labels ont déjà été cropés.
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
    return mask

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
        return mask

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

    return mask


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

    if task_name == '3classes':
        # Les prédictions OpenUS à visualiser sont celles des validations
        # croisées, et non les prédictions de test de custom_seg_new.
        openus_dirs = []
        for fold_dir in sorted(OPENUS_CV_ROOT.glob('fold_*')):
            attempts = sorted(fold_dir.glob('attempt_*/eval/predicted_masks_teacher'))
            if attempts:
                openus_dirs.append(attempts[-1])
        if openus_dirs:
            sources.append(('OpenUS', openus_dirs))

    return sources

def run(task_names, dpi=150):
  from utils.stats_fct import overlay_colored_mask, overlay_mask
  for task_name in task_names:
    task_config = TASKS[task_name]
    num_classes = task_config['num_classes']
    preserve_instances = task_name == 'oeufs'
    image_dir = IMAGE_ROOT / task_config['image_dir']
    output_dir = OUTPUT_ROOT / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Indexe toutes les prédictions par image et par modèle.
    predictions_by_image = defaultdict(dict)
    sources = prediction_sources(task_name)
    for source_name, source_dir in sources:
        source_dirs = source_dir if isinstance(source_dir, list) else [source_dir]
        for source_dir in source_dirs:
            for mask_path in sorted(source_dir.glob('pred_*.txt')):
                image_id = image_id_from_prediction(mask_path)
                previous = predictions_by_image[image_id].get(source_name)
                if previous is not None:
                    print(
                        f'Avertissement : doublon {source_name} pour {image_id} '
                        f'({previous} et {mask_path}); dernière prédiction conservée'
                    )
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
        image_tensor = np.asarray(image)

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

        # Tous les panneaux sont affichés sur une seule ligne. La hauteur de
        # la figure est calculée à partir du ratio des images cropées afin de
        # conserver leur format rectangulaire sans les déformer.
        num_panels = len(displays) + 1 + (true_display is not None)
        panel_width = 5
        panel_height = panel_width * IMAGE_SIZE[1] / IMAGE_SIZE[0]
        fig, axes = plt.subplots(
            1, num_panels,
            figsize=(panel_width * num_panels, panel_height)
        )
        axes_flat = np.asarray(axes).reshape(-1)

        axes_flat[0].imshow(image)
        axes_flat[0].set_title(
            'original', fontsize=14, fontweight='bold', pad=8
        )
        axes_flat[0].axis('off')

        first_model_index = 1
        if true_display is not None:
            axes_flat[1].imshow(true_display)
            axes_flat[1].set_title(
                'verite_terrain', fontsize=14,
                fontweight='bold', pad=8
            )
            axes_flat[1].axis('off')
            first_model_index = 2

        for index, (display_image, model_name) in enumerate(
            zip(displays, model_names), start=first_model_index
        ):
            axes_flat[index].imshow(display_image)
            axes_flat[index].set_title(
                model_name, fontsize=14,
                fontweight='bold', pad=8
            )
            axes_flat[index].axis('off')

        for axis in axes_flat[num_panels:]:
            axis.axis('off')

        # Réduit l'espace blanc entre les panneaux et autour de la figure,
        # tout en gardant une marge suffisante pour les titres.
        fig.tight_layout(pad=0.2, w_pad=0.02, h_pad=0.2)
        fig.savefig(output_dir / f'combined_{image_id}.png',
                    bbox_inches='tight', pad_inches=0.02, dpi=dpi)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('oeufs', 'cavite', 'both'), default='both')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args(argv)
    if args.dpi <= 0: parser.error('--dpi doit être strictement positif')
    selected = ('oeufs', '3classes') if args.task == 'both' else (('3classes',) if args.task == 'cavite' else ('oeufs',))
    run(selected, args.dpi)

if __name__ == '__main__':
    main()
