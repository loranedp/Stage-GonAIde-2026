# OpenUS : segmentation des échographies

Ce dossier adapte [OpenUS](https://www.arxiv.org/abs/2511.11510) à la segmentation multi-label du jeu de données COCO de `Projet/data`. Le parcours principal entraîne et évalue **Cavite** et **Gonade** par validation croisée à cinq folds. Une variante à trois classes ajoute **Intestin**.

Les commandes ci-dessous partent de la racine du dépôt `Stage-GonAIde-2026`.

## 1. Créer `OpenUs_venv`

```bash
cd Projet/OpenUS-multiclass
python3.10 -m venv OpenUs_venv
source OpenUs_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install triton==2.2.0 pytest chardet yacs termcolor fvcore seaborn packaging ninja einops numpy==1.24.4 timm==0.4.12 wandb pyiqa clean-fid pandas openpyxl scikit-image
```

Le noyau CUDA `selective_scan` accélère VMamba. Sa compilation demande `nvcc`, un compilateur C++ et les en-têtes Python 3.10 ; installez-le dans le même venv si ces outils sont disponibles :

```bash
nvcc --version
python -m pip install --no-build-isolation ./OpenUS/vmamba_models/selective_scan
python -c "import selective_scan_cuda_core; print('selective_scan OK')"
```

Sans ce noyau, le repli PyTorch fonctionne mais l'entraînement est beaucoup plus lent. Le poids OpenUS `OpenUS/checkpoint/openus_cpt0150.pth` est requis. Le poids VMamba optionnel `OpenUS/pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth` peut être absent : le code affiche alors un avertissement et charge le poids OpenUS.

Pour les sessions suivantes :

```bash
cd Projet/OpenUS-multiclass
source OpenUs_venv/bin/activate
cd OpenUS
```

## 2. Préparer les données

Depuis `Projet/OpenUS-multiclass`, entrer dans `OpenUS`, vérifier les entrées, puis créer les fichiers de splits :

```bash
cd OpenUS
python sync_custom_data.py --dry-run
python sync_custom_data.py
```

Le script lit les images et `annotations.json` dans `Projet/data/COCO`, ainsi que les poissons de test dans `Projet/utils/common_test_fish.json`. Il écrit `data/splits.json` et `data/splits_smoke.json` sans recopier les images. Le test est fixé par poisson et reste indépendant des cinq folds. Le fichier `Projet/utils/correspondance_echo_final.xlsx` est aussi nécessaire pour les mesures de surface en cm².

Si les données partagées se trouvent dans un autre répertoire `Projet`, passer son chemin à `sync_custom_data.py --projet-root /chemin/vers/Projet`. Pour la validation croisée, indiquer ensuite les chemins correspondants avec `--coco_json`, `--images_root` et `--metadata_file`.

## 3. Lancer la validation croisée à deux classes

Toujours depuis `Projet/OpenUS-multiclass/OpenUS`, avec `OpenUs_venv` activé :

```bash
python run_cross_validation.py \
  --coco_json ../../data/COCO/labels/cavite/annotations_2_classes.json \
  --output_dir output/custom_seg_cv5_2classes_v2 --dry-run

python run_cross_validation.py \
  --coco_json ../../data/COCO/labels/cavite/annotations_2_classes.json \
  --output_dir output/custom_seg_cv5_2classes_v2
```

`--dry-run` contrôle les entrées et affiche les commandes des cinq folds sans écrire de résultats ni entraîner de modèle. Le lancement normal exige un dossier de sortie encore inexistant. Si `output/custom_seg_cv5_2classes_v2` existe déjà, choisir un nouveau nom ou reprendre la même expérience :

```bash
python run_cross_validation.py \
  --coco_json ../../data/COCO/labels/cavite/annotations_2_classes.json \
  --output_dir output/custom_seg_cv5_2classes_v2 --resume
```

Après une modification du calcul des métriques, recalculer validation et test à partir des checkpoints existants, sans réentraînement :

```bash
python run_cross_validation.py \
  --coco_json ../../data/COCO/labels/cavite/annotations_2_classes.json \
  --output_dir output/custom_seg_cv5_2classes_v2 --recompute-metrics
```

Le lanceur partage les poissons du train et de la validation entre cinq folds (graine 42). Chaque fold entraîne un décodeur neuf sur un backbone gelé, choisit son checkpoint par Dice de validation, exporte les prédictions de validation hors fold, puis évalue le test indépendant. Le test n'intervient pas dans le choix du checkpoint. Les poids ne sont pas entraînés sur les images de leur fold de validation, mais celles-ci servent à choisir l'époque.

Par défaut : 100 époques, taux d'apprentissage `0.001`, batch de 4 images, 4 workers, images de 512 × 512 et clé de checkpoint `teacher`. Options utiles : `--epochs`, `--lr`, `--batch_size_per_gpu`, `--num_workers`, `--img_size`, `--val_freq`, `--seed`, `--pretrained_weights`, `--checkpoint_key`, `--split_file` et `--metadata_file`. Les chemins relatifs sont résolus depuis `OpenUS`. Le lanceur emploie le même interpréteur Python pour les scripts d'entraînement et de test. Les options `--no-save_predictions`, `--no-save_masks` et `--no-save_overlays` désactivent les exports de test correspondants ; les TXT hors fold restent produits.

## 4. Lire les résultats

Dans `output/custom_seg_cv5_2classes_v2` :

| Chemin | Contenu |
| --- | --- |
| `splits/` | Cinq splits et leur manifeste avec paramètres et empreintes des entrées. |
| `fold_N/attempt_M/` | Checkpoint retenu, métriques, journaux et prédictions du fold. |
| `fold_N/attempt_M/eval/predicted_masks_teacher/` | Un TXT de prédiction hors fold par image de validation. |
| `summary.json`, `summary.csv` | Métriques de validation et de test par fold ; le JSON contient leur moyenne non pondérée et l'écart-type d'échantillon. |
| `oof_validation_metrics.csv` | Métriques par image de validation hors fold. |
| `results_OpenUS_2classes.pkl` | Résultats par fold dans un format compatible avec l'analyse UNet. |

Les scores sont calculés sur les masques remis à la résolution originale. Les différences de surface sont en cm² ; si une échelle manque, la valeur correspondante est indisponible, sans supprimer les autres métriques. `summary.json` indique explicitement l'ordre des classes. Les TXT hors fold codent les contours externes par classe avec des coordonnées normalisées ; ils ne représentent pas les trous internes.

`--resume` vérifie le manifeste, ignore les folds terminés et reprend l'évaluation si l'entraînement est achevé. Un entraînement interrompu recommence dans une nouvelle tentative `attempt_M` ; les anciens fichiers restent dans leur dossier. Une erreur dans un fold arrête le lanceur.

## Variante à trois classes

Pour Cavite, Gonade et Intestin, employer `annotations.json` et **un autre dossier de sortie** :

```bash
python run_cross_validation.py \
  --coco_json ../../data/COCO/labels/cavite/annotations.json \
  --output_dir output/custom_seg_cv5_3classes --dry-run

python run_cross_validation.py \
  --coco_json ../../data/COCO/labels/cavite/annotations.json \
  --output_dir output/custom_seg_cv5_3classes
```

Le nombre de canaux du décodeur est déduit des catégories COCO : les checkpoints à deux et trois classes ne sont pas interchangeables. Le fichier d'analyse produit pour cette variante s'appelle `results_OpenUS_3classes.pkl`.
