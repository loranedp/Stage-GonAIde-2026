# Stage-GonAIde-2026
GonAIde : Application de l'IA à des images d'échographies de gonades pour estimer la fécondité chez la truite commune

## Installation des packages nécessaires
pip install -r requirements.txt

## Installation terminal pour l'OCR
sudo apt install tesseract-ocr

## Téléchargement des données et des poids
Les données et les poids des modèles sont disponibles sur next cloud : https://nextcloud.inrae.fr/s/6o4xxWFft2fap59?dir=/Github
- Données : mettre le dossier 'data' dans 'Debut/' pour récupérer les données.
- App : mettre le dossier 'checkpoints' dans 'Debut/App/' pour pouvoir lancer les prédictions de l'interface.
- UNet : mettre le dossier 'saves' dans 'Debut/UNet/' pour récupérer les poids de tous les modèles UNet.


## Quand nouvelles données :
- Telecharger depuis Roboflow les données au format "YOLO26" et "COCO segmentation"
- Mettre dans le dossier "data" avec la même architecture que présentement
- Relancer le script "pre-traitement.ipynb"
- Télécharger les poids associés aux modèles voulu (nextcloud)

## Lancement de l'application
streamlit run Debut/App/app.py