# Stage-GonAIde-2026
GonAIde : Application de l'IA à des images d'échographies de gonades pour estimer la fécondité chez la truite commune

## Installation des packages nécessaires
pip install -r requirements.txt

## Installation terminal pour l'OCR
sudo apt install tesseract-ocr

## Lancement de l'application
streamlit run Debut/App/app.py

## Quand nouvelles données :
- Telecharger depuis Roboflow les données au format "YOLO26" et "COCO segmentation"
- Mettre dans le dossier "data" avec la même architecture que présentement
- Relancer le script "pre-traitement.ipynb"
- Télécharger les poids associés aux modèles voulu (nextcloud)