# Stage-GonAIde-2026
GonAIde : Application de l'IA à des images d'échographies de gonades pour estimer la fécondité chez la truite commune

# Consignes pour lancer le projet

## Cloner le dépot github
```bash
git clone https://github.com/loranedp/Stage-GonAIde-2026.git
cd Stage-GonAIde-2026/
```

## Installer uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Créer un environnement virtuel (python 3.14)
```bash
uv venv mon_venv --python 3.14
```

## Installer les dépendances
```bash
uv pip install --python mon_venv/bin/python -r requirements.txt
```

## Activer l'environnement
```bash
source mon_venv/bin/activate
```

## Téléchargement des données et des poids
Les données et les poids des modèles sont disponibles sur next cloud : https://nextcloud.inrae.fr/s/6o4xxWFft2fap59?dir=/Github
- Données : mettre le dossier 'data' dans 'Projet/' pour récupérer les données.
- App : mettre le dossier 'checkpoints' dans 'Projet/App/' pour pouvoir lancer les prédictions de l'interface.

**Facultatif :**
- UNet : mettre le dossier 'saves' dans 'Projet/UNet/' pour récupérer les poids des modèles UNet.
- YOLO : mettre le dossier 'saves' dans 'Projet/YOLO26/' pour récupérer les poids du modèle YOLO.

## Lancer le pre-traitement des données

Installer préalablement **Tesseract OCR** si nécessaire :

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```
Puis lancer le pretraitement :
```bash
python Projet/utils/pretraitement.py
```

Utiliser les images et annotations originales : le prétraitement ne doit pas être relancé sur des images déjà recadrées.

## Lancer l'application
```bash
streamlit run Projet/App/app.py
```

L'application est alors accessible à l'adresse http://localhost:8501

Note : si les données entrées sont après 2025, il faut également re-télécharger le fichier csv mis à jours et renommer les images. Un script R pour renommer les images est disponible sur Nextcloud.

## Résultats
Les résultats du calcul des volumes et de la fécondité pour les poissons prédits dans l'interface sont disponibles dans 'Projet/Résultats/volumes_poissons.csv'.

Les résultats sur l'échantillon de validation et de tests sont également disponibles en .html dans 'Projet/Résultats/rapports'.


## En cas d'ajout de nouvelles données annotées
- Telecharger depuis Roboflow les données au format "YOLO26" et "COCO segmentation"
- Telecharger le fichier csv et le sauvegarder sous Projet/utils/correspondance_echo_final.xlsx
- Mettre dans le dossier "data" avec les données non prétraitée sur OpenCloud
- Remplacer toutes les images dans data (anciennes et nouvelles)
- Lancer le pretraitement des données : python3 utils/pretraitement.py
