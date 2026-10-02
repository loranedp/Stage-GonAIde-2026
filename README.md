# Stage-GonAIde-2026
GonAIde : Application de l'IA à des images d'échographies de gonades pour estimer la fécondité chez la truite commune

# Consignes pour lancer le projet

## Cloner le dépot github
git clone https://github.com/loranedp/Stage-GonAIde-2026.git

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

## Lancer l'application
streamlit run Projet/App/app.py

L'application est alors accessible à l'adresse http://localhost:8501

Note : si les données entrées sont après 2025, il faut également re-télécharger le fichier csv mis à jours

## Résultats
Les résultats du calcul des volumes et de la fécondité pour les poissons prédits dans l'interface sont disponibles dans 'Projet/Résultats/volumes_poissons.csv'.

## En cas d'ajout de nouvelles données annotées
- Telecharger depuis Roboflow les données au format "YOLO26" et "COCO segmentation"
- Mettre dans le dossier "data" avec les données non prétraitée sur OpenCloud
- Remplacer toutes les images dans data (anciennes et nouvelles)
- Lancer le pretraitement des données : python3 utils/pretraitement.py