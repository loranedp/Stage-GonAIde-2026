import os
import numpy as np
import torch
from torch.nn import functional as F
from pathlib import Path
from PIL import Image
import sys
import cv2
import json
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.append(str(Path.cwd().parent)) # Ajoute le dossier parent au chemin de recherche de Python

# Importation des datasets personnalisés pour U-Net : 2D et 2,5D
from utils.dataset_unet import RoboflowUNetDataset
from utils.dataset_2_5D import SliceSequenceUNetDataset
from utils.post_traitement import split_eggs

from utils.stats_fct import (
    plot_segmentation_comparison,
    mask_to_yolo_polygons,
    overlay_colored_mask,
)

# Zones où l'annotateur n'est pas certain de la présence d'une gonade (pixels à ignorer en loss/métriques).
IGNORE_CATEGORY_NAME = "ignore"

# ----------- Regroupement des images et annotations dans un dictionnaire -----------
def dataset_to_dict(df, annotations_json, all_images_dir):
    """
    Transforme le dataset COCO en un dictionnaire contenant les informations nécessaires pour l'entraînement et l'évaluation.
    """
    # Fichier d'annotation JSON
    with open(annotations_json, 'r') as f:
        coco_data = json.load(f)

    # -------- 1. Création de l'échantillon de données --------
    # --- Récupération de toutes les classes ---
    raw_categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', []) if cat['id'] > 0} # Associe les ID des catégories à leurs noms

    # --- Suppression de la catégorie "ignore" des classes réelles ---
    ignore_category_id = next((cid for cid, name in raw_categories.items() if name == IGNORE_CATEGORY_NAME), None)
    categories = {cid: name for cid, name in raw_categories.items() if cid != ignore_category_id}

    # --- Tri et affichage des classes détectées ---
    class_ids = sorted(list(categories.keys()))
    num_classes = len(class_ids)
    print(f"Classes détectées ({num_classes}) : {categories}")
    if ignore_category_id is not None:
        print(f"Catégorie ignore détectée : '{IGNORE_CATEGORY_NAME}' (id={ignore_category_id})")

    # --- Regroupement des annotations par ID d'image ---
    img_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # --- construction d'un dictionnaire avec toutes les images et leurs annotations associées ---
    all_samples = []
    for img_info in coco_data.get('images', []):
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_path = os.path.join(all_images_dir, img_name)
        
        # Vérifie si le fichier image existe
        if not os.path.exists(img_path):
            img_path = os.path.join(all_images_dir, os.path.basename(img_name))
            if not os.path.exists(img_path):
                continue

        id_image = img_name.split("_")[0]
        id_poisson = img_name.split("_")[2]

        # Récupère la catégorie et la position de l'Image
        matches_cat = df.loc[df["image_id"] == id_image, "categorie"]
        matches_op = df.loc[df["image_id"] == id_image, "position"]
        matches_op_ratio = df.loc[df["image_id"] == id_image, "position_ratio"]

        if not matches_cat.empty:
            cat = matches_cat.values[0]
            op = matches_op.values[0]
            op_ratio = matches_op_ratio.values[0]
        else:
            print(f"Image {id_image} introuvable dans les métadonnées !")
            cat = None
            op = None
            op_ratio = None

        # On ajoute toutes les informations dans le dictionnaire de l'échantillon
        all_samples.append({
            'image_path': img_path,
            'image_info': img_info,
            'annotations': img_to_anns.get(img_id, []),
            'id_poisson': id_poisson,
            'category': cat,
            'position': op,
            'position_ratio' : op_ratio
        })

    # Trier l'échantillon par nom de fichier
    all_samples.sort(key=lambda x: os.path.basename(x['image_path']))

    return all_samples, categories, num_classes, class_ids, ignore_category_id

# -------- Réorganisation des echantillons grace à la fonction collate_fn --------
def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    original_sizes = [item['original_size'] for item in batch]
    crop_params = [item['crop_params'] for item in batch]
    file_names = [os.path.basename(item['image_path']) for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'original_size': original_sizes,
        'crop_params': crop_params,
        'file_names': file_names
    }

def collate_fn_sequence(batch):
    assert len(batch) == 1, (
        "SliceSequenceUNetDataset impose batch_size=1 : chaque batch doit correspondre "
        "à une séquence complète (un seul poisson), car T varie (1 à 7 coupes) et "
        "aucun padding n'est utilisé."
    )
    item = batch[0]
    return {
        'image': item['image'].unsqueeze(0),   # (1, T, C, H, W)
        'mask': item['mask'].unsqueeze(0),     # (1, T, num_classes, H, W)
        'original_size': [item['original_size']],
        'crop_params': [item['crop_params']],
        'image_path': [item['image_path']],
        'id_poisson': [item['id_poisson']],
    }

def collate_fn_egg_hv(batch):
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'image_path': [item['image_path'] for item in batch],
        'original_size': [item['original_size'] for item in batch],
        'crop_params': [item['crop_params'] for item in batch],
    }


def split_dataset(all_samples):
    # Récupérer la liste de toutes les catégories
    categories_all = [sample['category'] for sample in all_samples]

    X = np.array(all_samples, dtype=object) # Les images
    y = np.array(categories_all) # Les classes
    groups = np.array([sample["image_info"]["file_name"].split("_")[2] for sample in all_samples]) # L'ID du poisson pour chaque image

    # Les données sont divisées par StratifiedGroupKFold (par groupe et par poisson) excepté pour les oeufs, uniquement divisés par groupes
    metadata_available = all(value is not None and not (isinstance(value, float) and np.isnan(value))
                             for value in categories_all)
    if metadata_available and len(np.unique(y)) > 1:
        kf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
        split_iterator = kf.split(X, y, groups) # Divise selon la distribution des classes et les groupes (poissons)
    else:
        kf = GroupKFold(n_splits=6, shuffle=True, random_state=42)
        split_iterator = kf.split(X, groups=groups) # Divise uniquement selon les groupes (poissons)

    # --- Division des données en ensembles d'entraînement/validation et de test ---
    for cv_idx, test_idx in split_iterator:
        cv_samples, test_samples = X[cv_idx], X[test_idx]
        cv_categories, test_categories = y[cv_idx], y[test_idx]
        groups_train, groups_test = groups[cv_idx], groups[test_idx]
        break # On ne garde que le premier split

    print(f"Nombre total d'images valides : {len(all_samples)}")
    print(f"Échantillons train/val        : {len(cv_samples)}")
    print(f"Échantillons de test          : {len(test_samples)}")

    return cv_samples, test_samples, cv_categories, test_categories, groups_train, groups_test


def _mean_top_half_instance_area_px(instance_labels):
    """Surface moyenne en pixels des 50 % plus grandes instances, par image."""
    if instance_labels.ndim == 2:
        instance_labels = instance_labels.unsqueeze(0)
    if instance_labels.ndim != 3:
        raise ValueError("Les labels d'instances d'œufs doivent avoir la forme (B, H, W).")

    mean_areas = []
    for labels in instance_labels:
        label_ids, counts = torch.unique(labels, return_counts=True)
        areas = counts[label_ids > 0].to(dtype=torch.float32)
        if areas.numel() == 0:
            mean_areas.append(torch.zeros((), device=labels.device))
            continue
        selected_count = max(1, int(0.5 * areas.numel()))
        mean_areas.append(torch.topk(areas, selected_count).values.mean())
    return torch.stack(mean_areas)


# Calcul des métriques par image et par classe avec prise en compte de l'échelle et du dataset
def metrics_by_class(true, preds, echelle, dataset_name, egg_instance_labels=None) :
    intersection = (preds & true).float().sum((2, 3))
    union = (preds | true).float().sum((2, 3))   

    # --- Calcul de l'IoU ---
    iou_per_img = intersection / (union + 1e-6)  
    iou_per_img[union==0] = 1.0 # Si prédiction et vérité sont toutes deux vides, IoU = 1.0

    # --- Calcul du Dice Score ---
    pred_area = preds.float().sum((2, 3))
    true_area = true.float().sum((2, 3))
    dice_per_img = (2 * intersection) / (pred_area + true_area + 1e-6)
    both_empty = (pred_area == 0) & (true_area == 0)
    dice_per_img[both_empty] = 1.0

    # --- Calcul de la précision et du rappel ---
    false_positive = (preds.float().sum((2, 3)) - intersection)
    false_negative = (true.float().sum((2, 3)) - intersection)
    precision_per_img = intersection / (intersection + false_positive + 1e-6)
    recall_per_img = intersection / (intersection + false_negative + 1e-6)

    #--- Calcul des surfaces ---
    pred_surf_pixels = pred_area
    true_surf_pixels = true_area

    diff_surf_pixels = (true_surf_pixels - pred_surf_pixels).abs()
    if dataset_name != "oeufs":
        echelle = echelle.to(
            device=diff_surf_pixels.device,
            dtype=diff_surf_pixels.dtype
        ).reshape(-1, 1)

        # ``echelle`` est la hauteur physique totale de l'image en cm.
        # Un pixel représente donc (echelle / hauteur_px) ** 2 cm².
        diff_surface = diff_surf_pixels * (echelle / true.shape[2]) ** 2
    else:
        true_instance_labels = split_eggs(
            true, min_distance=8, min_area=100, ouverture_size=3, ouverture_iterations=1
        )
        if egg_instance_labels is None:
            egg_instance_labels = split_eggs(
                preds, min_distance=8, min_area=100, ouverture_size=3, ouverture_iterations=1
            )
        else:
            egg_instance_labels = torch.as_tensor(egg_instance_labels, device=true.device)
        predicted_mean_area = _mean_top_half_instance_area_px(egg_instance_labels)
        true_mean_area = _mean_top_half_instance_area_px(true_instance_labels)
        echelle = echelle.to(device=true.device, dtype=torch.float32).reshape(-1)

        # ``echelle`` est exprimée en cm ; conversion finale en mm².
        diff_surface = (true_mean_area - predicted_mean_area).abs().unsqueeze(1)
        diff_surface *= ((echelle * 10) / true.shape[2]).unsqueeze(1) ** 2
        missing_surface = (true_mean_area == 0) | (predicted_mean_area == 0)
        diff_surface[missing_surface.unsqueeze(1)] = torch.nan

    return intersection, union,iou_per_img, dice_per_img, precision_per_img, recall_per_img, diff_surface

def save_validation_masks(masks_dir, best_masks):
    """Exporte les prédictions du meilleur epoch au format YOLO."""
    for file_name, _true_mask, pred_mask in best_masks:
        stem = Path(file_name).stem
        pred_path = Path(masks_dir) / f"pred_{stem}.txt"

        with pred_path.open("w") as f_pred:
            for class_idx in range(pred_mask.shape[0]):
                polygons = mask_to_yolo_polygons(
                    pred_mask[class_idx], class_idx, target_size=(510, 380)
                )
                if polygons:
                    f_pred.write("\n".join(polygons) + "\n")


def save_validation_overlays(results_dir, best_images, dataset_name=None,
                             class_names=None):
    """Sauvegarde les prédictions avec une couleur par instance ou classe."""
    for file_name, image, pred_mask in best_images:
        stem = Path(file_name).stem
        overlay = overlay_colored_mask(
            image,
            pred_mask,
            alpha=0.45,
            dataset_name=dataset_name,
            class_names=class_names,
        )
        output_path = Path(results_dir) / f"pred_overlay_{stem}.png"
        Image.fromarray(overlay).save(output_path)
