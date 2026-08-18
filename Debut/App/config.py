"""Chemins et constantes partagés par l'application Streamlit."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- Modèle : ensemble des 5 folds (U-NET) ---
FOLD_MODEL_PATHS = [
    ROOT_DIR / "App" / "checkpoints" / "unet_finetuned_fold_1.pth",
    ROOT_DIR / "App" / "checkpoints" / "unet_finetuned_fold_2.pth",
    ROOT_DIR / "App" / "checkpoints" / "unet_finetuned_fold_3.pth",
    ROOT_DIR / "App" / "checkpoints" / "unet_finetuned_fold_4.pth",
    ROOT_DIR / "App" / "checkpoints" / "unet_finetuned_fold_5.pth",
    ]
FOLD_EGG_MODEL_PATHS = [
    ROOT_DIR / "App" / "checkpoints" / f"unet_finetuned_fold_{fold}_UNet_oeufs.pth"
    for fold in range(1, 6)
]
IMAGE_SIZE = (512, 512)  # (W, H) taille d'entrée du modèle, après crop
CROP_PARAMS = [85, 33, 510, 380]  # x, y, w, h appliqué avant resize (RoboflowUNetDataset)
OCR_SCALE_CROP = (530, 350, 250, 200)  # x, y, w, h ; zone d'affichage de l'échelle à l'écran (cf. utils/pre-traitement.ipynb)
ENCODER_NAME = "resnet34"

# Classes de segmentation (sortie sigmoïde multi-label, pas de classe fond).
# Index 0/1/2 = category_id 1/2/3 de data/COCO/annotations.json.
CLASS_NAMES = ["Cavite", "Gonade", "Intestin"]
EGG_CLASS_NAMES = ["Oeuf"]
NUM_CLASSES = len(CLASS_NAMES)
GONAD_CLASS_INDEX = CLASS_NAMES.index("Gonade")
PRED_THRESHOLD = 0.5

# Réglages de séparation des instances d'œufs transmis au watershed.
EGG_MIN_DISTANCE = 6
EGG_MIN_AREA = 150

# --- Métadonnées (catégorie CME, position sonde, longueur de gonade) ---
METADATA_XLSX_PATH = ROOT_DIR / "utils" / "correspondance_echo_final.xlsx"

# --- Données d'entraînement ---
COCO_ANN_PATH = DATA_DIR / "COCO" / "annotations.json"
EGG_COCO_ANN_PATH = DATA_DIR / "COCO" / "annotations.oeufs.json"
COCO_IMAGES_DIR = DATA_DIR / "COCO" / "images"

# Dossier des images validées "bonne" depuis l'app, ajoutées de façon permanente
# et conservées séparément des données d'origine pour pouvoir les différencier.
TRAIN_ADDED_DIR = DATA_DIR / "App" / "train_added"
TRAIN_ADDED_IMAGES_DIR = TRAIN_ADDED_DIR / "images"
TRAIN_ADDED_ANN_PATH = TRAIN_ADDED_DIR / "annotations.json"

# Dossier des images validées "mauvaise" depuis l'app, mise de coté pour labellisation ultérieure
LABELLISATION_DIR = DATA_DIR / "App" / "A_labelliser"
LABELLISATION_IMAGES_DIR = LABELLISATION_DIR / "images"
LABELLISATION_ANN_PATH = LABELLISATION_DIR / "annotations.json"

# Dossier des images dont les masques ont été corrigés manuellement dans l'app
# (bouton "Re-labellisation"), utilisées elles aussi pour le ré-entraînement.
CORRECTED_DIR = DATA_DIR / "App" / "Corrigees"
CORRECTED_IMAGES_DIR = CORRECTED_DIR / "images"
CORRECTED_ANN_PATH = CORRECTED_DIR / "annotations.json"

# Magasins séparés pour les images d'œufs. Ils reprennent le même format COCO
# minimal que les magasins cavité, avec une unique catégorie technique Oeuf.
EGG_TRAIN_ADDED_DIR = DATA_DIR / "App" / "train_added_oeufs"
EGG_TRAIN_ADDED_IMAGES_DIR = EGG_TRAIN_ADDED_DIR / "images"
EGG_TRAIN_ADDED_ANN_PATH = EGG_TRAIN_ADDED_DIR / "annotations.json"

EGG_LABELLISATION_DIR = DATA_DIR / "App" / "A_labelliser_oeufs"
EGG_LABELLISATION_IMAGES_DIR = EGG_LABELLISATION_DIR / "images"
EGG_LABELLISATION_ANN_PATH = EGG_LABELLISATION_DIR / "annotations.json"

EGG_CORRECTED_DIR = DATA_DIR / "App" / "Corrigees_oeufs"
EGG_CORRECTED_IMAGES_DIR = EGG_CORRECTED_DIR / "images"
EGG_CORRECTED_ANN_PATH = EGG_CORRECTED_DIR / "annotations.json"

# --- Export des résultats de calcul de volume ---
RESULTATS_DIR = ROOT_DIR / "Résultats"
RESULTATS_CSV_PATH = RESULTATS_DIR / "volumes_poissons.csv"

# --- Statuts de validation ---
STATUS_OPTIONS = ["bonne", "ok", "mauvaise"]
DEFAULT_STATUS = "ok"
