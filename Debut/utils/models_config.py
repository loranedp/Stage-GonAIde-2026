import os
import numpy as np
import torch
from torch.nn import functional as F
from pathlib import Path
import sys
import cv2
import json
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


sys.path.append(str(Path.cwd().parent)) # Ajoute le dossier parent au chemin de recherche de Python

# Importation des datasets personnalisés pour U-Net : 2D et 2,5D
from utils.dataset_unet import RoboflowUNetDataset
from utils.dataset_2_5D import SliceSequenceUNetDataset

# Catégorie d'annotation spéciale : ne devient jamais un canal de sortie du modèle.
# Sert uniquement à marquer, sur le canal Gonade, des zones où l'annotateur n'est
# pas certain de la présence d'une gonade (pixels à ignorer en loss/métriques).
IGNORE_CATEGORY_NAME = "ignore"


# ----------- Regroupement des images et annotations dans un dictionnaire -----------
def dataset_to_dict(df, annotations_json, all_images_dir):
    # Fichier d'annotation JSON
    with open(annotations_json, 'r') as f:
        coco_data = json.load(f)

    # -------- 1. Création de l'échantillon de données --------
    # Récupération des classes
    raw_categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', []) if cat['id'] > 0} # Associe les ID des catégories à leurs noms

    # La catégorie "ignore" (id=4 dans les exports COCO) est retirée des classes réelles :
    # elle ne doit jamais devenir un canal de sortie du modèle, seulement marquer des pixels à exclure.
    ignore_category_id = next((cid for cid, name in raw_categories.items() if name == IGNORE_CATEGORY_NAME), None)
    categories = {cid: name for cid, name in raw_categories.items() if cid != ignore_category_id}

    class_ids = sorted(list(categories.keys()))
    num_classes = len(class_ids)
    print(f"Classes détectées ({num_classes}) : {categories}")
    if ignore_category_id is not None:
        print(f"Catégorie ignore détectée : '{IGNORE_CATEGORY_NAME}' (id={ignore_category_id})")

    # Regrouper les annotations par ID d'image
    img_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Construire d'un dictionnaire avec toutes les images et leurs annotations associées
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
            print(f"Image {id_image} introuvable dans les métadonnées Excel !")
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


def split_dataset(all_samples, all_images_dir, df, IMAGE_SIZE, num_classes, class_ids, dim="2",
                   ignore_category_id=None, ignore_target_class_idx=None, ignore_value=-100.0):
    # Récupérer la liste de toutes les catégories
    categories_all = [sample['category'] for sample in all_samples]

    X = np.array(all_samples, dtype=object) # Les images
    y = np.array(categories_all) # Les classes
    groups = np.array([sample["image_info"]["file_name"].split("_")[2] for sample in all_samples]) # L'ID du poisson pour chaque image

    # Les exports qui ne contiennent pas les métadonnées d'échographie (ex: oeufs)
    # ne peuvent pas être traités par StratifiedGroupKFold avec des NaN. On conserve
    # l'absence de stratification tout en garantissant qu'un même poisson reste dans
    # un seul sous-ensemble.
    metadata_available = all(value is not None and not (isinstance(value, float) and np.isnan(value))
                             for value in categories_all)
    if metadata_available and len(np.unique(y)) > 1:
        kf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
        split_iterator = kf.split(X, y, groups)
    else:
        kf = GroupKFold(n_splits=6)
        split_iterator = kf.split(X, groups=groups)

    for cv_idx, test_idx in split_iterator:
        cv_samples, test_samples = X[cv_idx], X[test_idx]
        cv_categories, test_categories = y[cv_idx], y[test_idx]
        groups_train, groups_test = groups[cv_idx], groups[test_idx]
        break # On ne garde que le premier split

    print(f"Nombre total d'images valides : {len(all_samples)}")
    print(f"Échantillons train/val        : {len(cv_samples)}")
    print(f"Échantillons de test          : {len(test_samples)}")

    # -------- 2. Création des datasets --------
    if dim == "2":
        cv_dataset = RoboflowUNetDataset(
        samples=cv_samples,
        image_dir=all_images_dir,
        df = df,
        image_size=IMAGE_SIZE,
        num_classes=num_classes,
        class_ids=class_ids,
        is_train = False,
        ignore_category_id=ignore_category_id,
        ignore_target_class_idx=ignore_target_class_idx,
        ignore_value=ignore_value
        )
        test_dataset = RoboflowUNetDataset(
            samples=test_samples,
            image_dir=all_images_dir,
            df = df,
            image_size=IMAGE_SIZE,
            num_classes=num_classes,
            class_ids=class_ids,
            is_train = False,
            ignore_category_id=ignore_category_id,
            ignore_target_class_idx=ignore_target_class_idx,
            ignore_value=ignore_value
        )
    else:
        cv_dataset = SliceSequenceUNetDataset(
        samples=cv_samples,
        image_dir=all_images_dir,
        df = df,
        image_size=IMAGE_SIZE,
        num_classes=num_classes,
        class_ids=class_ids,
        is_train = False,
        ignore_category_id=ignore_category_id,
        ignore_target_class_idx=ignore_target_class_idx,
        ignore_value=ignore_value
    )
        test_dataset = SliceSequenceUNetDataset(
            samples=test_samples,
            image_dir=all_images_dir,
            df = df,
            image_size=IMAGE_SIZE,
            num_classes=num_classes,
            class_ids=class_ids,
            is_train = False,
            ignore_category_id=ignore_category_id,
            ignore_target_class_idx=ignore_target_class_idx,
            ignore_value=ignore_value
        )

    return cv_samples, test_samples, cv_categories, test_categories, groups_train, groups_test, cv_dataset, test_dataset

# --------------- Définition des fonctions de perte personnalisées pour l'entraînement du modèle ---------------
# Dice Loss
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs) # Transforme les prédictions en probabilités
        intersection = (inputs * targets).sum(dim=(2, 3)) # Intersection entre les prédictions et les cibles
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        return 1. - dice.mean()  # Moyenne sur le batch et les classes
    
# Binary Cross-Entropy Loss + Dice Loss
class BCE_DiceLoss(torch.nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# --------------- Fonctions de pertes avec des pixels ignorés (artefacts) pour l'entraînement du modèle ---------------
# 1. Dice Loss avec prise en compte de ignore_index
class DiceLoss_ignore(torch.nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs) # Transforme les prédictions en probabilités

        # --- Ignore des pixels ---
        valid_mask = (targets != self.ignore_index).float() # Masque binaire des pixels valides
        targets_clean = torch.where(targets == self.ignore_index, torch.tensor(0.0, device=targets.device), targets) # Remplace les -100 par 0 pour le calcul de la perte
        inputs_masked = inputs * valid_mask # Neutralise les pixels d'artefacts dans les prédictions
        targets_masked = targets_clean * valid_mask # Neutralise les pixels d'artefacts dans les cibles
        
        # Calcul de l'intersection et de la somme sur les zones valides uniquement
        intersection = (inputs_masked * targets_masked).sum(dim=(2, 3))
        total = inputs_masked.sum(dim=(2, 3)) + targets_masked.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (total + self.smooth)
        return 1. - dice.mean()

# 2. BCE + Dice Loss combinée avec ignore_index
class BCE_DiceLoss_ignore(torch.nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, ignore_index=-100):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss_ignore(ignore_index=ignore_index)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # --- Ignore des pixels ---
        valid_mask = (targets != self.ignore_index).float() # masque binaire des pixels valides
        targets_clean = torch.where(targets == self.ignore_index, torch.tensor(0.0, device=targets.device), targets) # Remplace les -100 par 0 pour le calcul de la BCE
        
        # Calcul de la BCE pixel par pixel puis moyennage uniquement sur les pixels valides
        bce_raw = self.bce(inputs, targets_clean)
        bce_loss = (bce_raw * valid_mask).sum() / valid_mask.sum().clamp(min=1e-6)
        
        dice_loss = self.dice(inputs, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# --------- Calcul des métriques par image et par classe ---------
def get_metrics(true_masks, pred_masks):
            # Calcul TP, FP et FN par image et par classe
            tp = (pred_masks & true_masks).float().sum((2, 3))
            fp = (pred_masks & ~true_masks).float().sum((2, 3))
            fn = (~pred_masks & true_masks).float().sum((2, 3))

            # Calcul des surfaces et leurs différences
            true_surface = true_masks.float().sum((2, 3))
            pred_surface = pred_masks.float().sum((2, 3))
            
            diff_surface = (true_surface - pred_surface).abs()
            ratio_surface = diff_surface / (true_surface + 1e-6)

            # Calculs métriques par images
            intersection = tp
            union = (pred_masks | true_masks).float().sum((2, 3))   
            
            iou_per_img = intersection / (union + 1e-6)  
            precision_per_img = tp / (tp + fp + 1e-6)
            recall_per_img = tp / (tp + fn + 1e-6)
            f1_per_img = 2 * tp / (2 * tp + fp + fn + 1e-6)

            iou_per_img[union==0] = 1.0

            return tp, fp, fn,iou_per_img, precision_per_img, recall_per_img, f1_per_img, diff_surface, ratio_surface
