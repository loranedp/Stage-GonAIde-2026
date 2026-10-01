# Stage-GonAIde-2026
GonAIde : Application de l'IA à des images d'échographies de gonades pour estimer la fécondité chez la truite commune

# Consignes pour lancer le projet :

## Cloner le dépot github :
git clone le dépôt dans un dossier au choix : git clone https://github.com/loranedp/Stage-GonAIde-2026.git
cd Stage-GonAIde-2026/

## Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh

## Créer un environnement virtuel (python 3.14)
uv venv mon_venv --python 3.14

## Installer les dépendances
uv pip install --python mon_venv/bin/python -r requirements.txt

## Activer l'environnement
source mon_venv/bin/activate

## Téléchargement des données et des poids
Les données et les poids des modèles sont disponibles sur next cloud : https://nextcloud.inrae.fr/s/6o4xxWFft2fap59?dir=/Github
- Données : mettre le dossier 'data' dans 'Projet/' pour récupérer les données.
- App : mettre le dossier 'checkpoints' dans 'Projet/App/' pour pouvoir lancer les prédictions de l'interface.

**Facultatif :**
- UNet : mettre le dossier 'saves' dans 'Projet/UNet/' pour récupérer les poids des modèles UNet.
- YOLO : mettre le dossier 'saves' dans 'Projet/YOLO26/' pour récupérer les poids du modèle YOLO.

## Lancer le pre-traitement des données
python3 Projet/utils/pretraitement.py

## Lancement de l'application
streamlit run Projet/App/app.py

## En cas d'ajout de nouvelles données :
(2 cas : avant et après 2025)
- Telecharger depuis Roboflow les données au format "YOLO26" et "COCO segmentation"
- Mettre dans le dossier "data" avec la même architecture que présentement (les images et les labels)
- Lancer le pretraitement des nouvelles données : utils/python3 pretraitement.py

L'application est alors accessible à l'adresse http://localhost:8501.

## Résultats
Les résultats du calcul des volumes et de la fécondité pour les poissons prédits dans l'interface sont disponibles dans 'Projet/Résultats/volumes_poissons.csv'.
