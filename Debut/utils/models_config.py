import os
import numpy as np
import torch
from pathlib import Path
import sys
import json

sys.path.append(str(Path.cwd().parent)) # Ajoute le dossier parent au chemin de recherche de Python

# Importation des datasets personnalisés pour U-Net : 2D et 2,5D
from utils.dataset_unet import RoboflowUNetDataset
from utils.unet_spatial import unpad_array
from utils.dataset_2_5D import SliceSequenceUNetDataset
from utils.post_traitement import split_eggs  # Compatibilité avec les anciens imports/tests.
from utils.segmentation_metrics import (
    _median_areas_px,
    _median_instance_area_px,
    metrics_by_class,
)
from utils.segmentation_visualization import save_validation_overlays
from utils.yolo_metrics import mask_to_yolo_polygons

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
    file_names = [os.path.basename(item['image_path']) for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'original_size': original_sizes,
        'egg_annotation_areas_px': [item.get('egg_annotation_areas_px') for item in batch],
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
        'image_path': [item['image_path']],
        'id_poisson': [item['id_poisson']],
    }

def collate_fn_egg_hv(batch):
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'image_path': [item['image_path'] for item in batch],
        'original_size': [item['original_size'] for item in batch],
        'egg_annotation_areas_px': [item.get('egg_annotation_areas_px') for item in batch],
    }


def split_dataset(all_samples):
    # Récupérer la liste de toutes les catégories
    categories_all = [sample['category'] for sample in all_samples]

    X = np.array(all_samples, dtype=object) # Les images
    y = np.array(categories_all) # Les classes
    groups = np.array([sample["image_info"]["file_name"].split("_")[2] for sample in all_samples]) # L'ID du poisson pour chaque image

    # Liste figée après un tirage simple de ~1/6 des poissons communs cavité/œufs (graine 42).
    common_test_fish = json.loads((Path(__file__).resolve().parent / "common_test_fish.json").read_text())
    test_mask = np.isin(groups, list(common_test_fish))
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise ValueError("Aucun poisson de common_test_fish.json n'est présent dans ce dataset.")
    cv_idx = np.flatnonzero(~test_mask)
    cv_samples, test_samples = X[cv_idx], X[test_idx]
    cv_categories, test_categories = y[cv_idx], y[test_idx]
    groups_train, groups_test = groups[cv_idx], groups[test_idx]

    print(f"Nombre total d'images valides : {len(all_samples)}")
    print(f"Échantillons train/val        : {len(cv_samples)}")
    print(f"Échantillons de test          : {len(test_samples)}")

    return cv_samples, test_samples, cv_categories, test_categories, groups_train, groups_test


def save_validation_masks(masks_dir, best_masks, dataset_name=None):
    """Exporte les prédictions du meilleur epoch au format YOLO.

    Pour les œufs, ``pred_mask`` est une carte 2D d'identifiants d'instances.
    Chaque identifiant positif est exporté séparément afin que deux œufs
    adjacents ne soient pas refusionnés lors de l'extraction des contours.
    """
    for file_name, _true_mask, pred_mask in best_masks:
        stem = Path(file_name).stem
        pred_path = Path(masks_dir) / f"pred_{stem}.txt"

        with pred_path.open("w") as f_pred:
            if dataset_name == "oeufs":
                instance_labels = unpad_array(pred_mask)
                if instance_labels.ndim == 3 and instance_labels.shape[0] == 1:
                    instance_labels = instance_labels[0]
                if instance_labels.ndim != 2:
                    raise ValueError(
                        "La carte d'instances d'œufs doit avoir la forme "
                        "(H, W) ou (1, H, W)."
                    )
                for instance_id in torch.unique(instance_labels):
                    if int(instance_id) <= 0:
                        continue
                    polygons = mask_to_yolo_polygons(
                        instance_labels == instance_id,
                        class_id=0,
                        target_size=(510, 380),
                        largest_only=True,
                    )
                    if polygons:
                        f_pred.write(polygons[0] + "\n")
                continue

            for class_idx in range(pred_mask.shape[0]):
                polygons = mask_to_yolo_polygons(
                    unpad_array(pred_mask[class_idx]), class_idx, target_size=(510, 380)
                )
                if polygons:
                    f_pred.write("\n".join(polygons) + "\n")
