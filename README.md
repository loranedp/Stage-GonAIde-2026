# Stage-GonAIde-2026
GonAIde : Application de l'IA à des images d'échographies de gonades pour estimer la fécondité chez la truite commune

## Création d'un environnement virtuel (python 3.14)
cd Stage-GonAIde-2026

python3.14 -m venv mon_venv

source mon_venv/bin/activate

## Installation des packages nécessaires
pip install -r requirements.txt

## Téléchargement des données et des poids
Les données et les poids des modèles sont disponibles sur next cloud : https://nextcloud.inrae.fr/s/6o4xxWFft2fap59?dir=/Github
- Données : mettre le dossier 'data' dans 'Debut/' pour récupérer les données.
- App : mettre le dossier 'checkpoints' dans 'Debut/App/' pour pouvoir lancer les prédictions de l'interface.
- UNet : mettre le dossier 'saves' dans 'Debut/UNet/' pour récupérer les poids de tous les modèles UNet.

## Ajout de nouvelles données :
- Telecharger depuis Roboflow les données au format "YOLO26" et "COCO segmentation"
- Mettre dans le dossier "data" avec la même architecture que présentement (les images et les labels)
- Lancer le pretraitement des nouvelles données : utils/python3 pretraitement.py

## Lancement de l'application
streamlit run Debut/App/app.py

L'application est alors accessible à l'adresse http://localhost:8501.

## Résultats
Les résultats du calcul des volumes et de la fécondité sont disponibles dans 'Debut/Résultats/volumes_poissons.csv'.

## UltraSam
Pour relancer le modèle, suivre le .README présent dans le dossier.
