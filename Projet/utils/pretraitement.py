# SCRIPT POUR LE PRETRAITEMENT DES DONNEES AVANT L'ENTRAINEMENT DES MODELES
# A lancer si ajout de nouvelles données dans le dataset


# ================== 1. Importation des packages ==================
import pandas as pd
import os
import json
import numpy as np
import pytesseract
import glob
from pytesseract import Output
import cv2
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from stats_fct import overlay_colored_mask
from yolo_metrics import load_yolo_polygon_masks


# ================== 2. Gestion des metadata ==================
# Importation du fichier avec les metadatas
df = pd.read_excel('Correspondance_Capture-Echographe.xlsx', sheet_name = 2, usecols=[0,1,4,5,6])
df = df.rename({"id_image": "name_file"}, axis = 1)

print("Importation de toutes les métadonnées :")
df_cap = pd.read_excel('Correspondance_Capture-Echographe.xlsx', sheet_name = 1, usecols=[4,10,11,20])

# Filtre sur les lignes sans identifiants et sans nom
df = df[df.cap_id != "?"]
df = df[df.name_file != "na"]

print(f"Nombre de poissons : {len(df['cap_id'].value_counts())}")
print(f"Nombre d'échographies totales : {len(df)}")
print(f"Nombre d'échographies de gonades : {len(df[df.type_image != 'œufs'])}")
print(f"Nombre d'échographies d'œufs : {len(df[df.type_image == 'œufs'])}")
print("---------------------------------")


# ================== 3. Ajout de l'identifiant des échographies ==================
names = []
for name in df.name_file :
    if len(name.split("[")) > 1:
        ref_csv = name.split("[")[1]
        ref_csv = ref_csv.split("]")[0]
        names.append(ref_csv)
    else :
        names.append("NA")

df["image_id"] = names


# ================== 4. Transformation des masques et images ==================
# -------- 4.1 Ordonne les classes des masques YOLO --------
def sort_classes(label_dir):
    for file_name in os.listdir(label_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(label_dir, file_name)
            with open(file_path, "r") as f: # Ouverture du fichier .txt
                lines = f.readlines()

            # Filtre sur les lignes vides
            lines = [line.strip() for line in lines if line.strip()]

            # Ordonne les lignes selon leur classe (premier élément de chaque ligne)
            lines.sort(key=lambda line: int(line.split()[0]))

            # Ajoute un saut de ligne à la fin de chaque ligne
            lines = [line + "\n" for line in lines]

            # Écrire ligne par ligne les nouvelles lignes dans le fichier
            with open(file_path, "w") as f:
                f.writelines(lines)

sort_classes("../data/YOLO/labels/oeufs")
sort_classes("../data/YOLO/labels/cavite")

print(f"Nombre d'images COCO importées de la cavite : {len(os.listdir('../data/COCO/images/cavite/'))}")
print(f"Nombre d'images COCO importées des oeufs : {len(os.listdir('../data/COCO/images/oeufs/'))}")
print(f"Nombre d'images YOLO importées de la cavite : {len(os.listdir('../data/YOLO/images/cavite/'))}")
print(f"Nombre d'images YOLO importées des oeufs : {len(os.listdir('../data/YOLO/images/oeufs/'))}")

# -------- 4.2 Renomage des images et des masques YOLO --------
def rename(path, format):
    for file_name in os.listdir(path):
        if Path(file_name).suffix.lower() != f".{format.lower()}":
            continue
        if "-" in file_name : # Vérifie si l'image n'a pas déjà été renomée
            split_name = file_name.split("_")

            if len(split_name) == 4:
                # Récupère du nom l'id de l'image et son heure
                heures = split_name[0]
                cap_id = split_name[2]

                cap_id = cap_id.split("-")[1]
                heures = heures.replace("-",".")
                heures = heures.split(" ")[0]

                try:
                    # Récupère l'année
                    annee = str(df.loc[df.image_id == cap_id, 'annee'].item())
                    # Récupère l'identifiant de la capture (poisson)
                    poisson = str(df.loc[df.image_id == cap_id, 'cap_id'].item())

                    # Nouveau nom
                    new_file_name = f"{cap_id}_{heures}_{poisson}_{annee}.{format}"

                except ValueError:
                    # Si cap_id non trouvé ou non unique
                    print(f"L'année n'a pas été trouvée. Le nom n'a pas été changé.")

            # Si déjà renommé avant l'annotation
            if len(split_name) == 5:
                new_file_name = "_".join(split_name[0:4])
                new_file_name = f"{new_file_name.replace("-",".")}.{format}"

            os.rename(path + file_name, path + new_file_name)

    return True

# Renommages des images
rename("../data/COCO/images/cavite/", "jpg")
rename("../data/COCO/images/oeufs/", "jpg")
rename("../data/YOLO/images/cavite/", "jpg")
rename("../data/YOLO/images/oeufs/", "jpg")

# Renommages des labels YOLO
rename("../data/YOLO/labels/cavite/", "txt")
rename("../data/YOLO/labels/oeufs/", "txt")

# Prépare les noms finaux avant le crop afin de lire l'échelle sur les images
# d'origine, dont la zone OCR est supprimée par le recadrage.
df["new_name_file"] = ""
for index, row in df.iterrows():
    heures = row["name_file"].split("_")[0].replace("-", ".").split(" ")[0]
    df.at[index, "new_name_file"] = (
        f"{row['image_id']}_{heures}_{row['cap_id']}_{row['annee']}"
    )


# -------- 4.3 Récupère l'échelle de l'échographie avant le crop --------
ocr_crop_warning_shown = False

def OCR(image_path):
    """Lit l'échelle dans une image originale non recadrée.

    L'échelle est affichée au même endroit sur les échographies de gonade et
    d'œufs. Cette fonction doit donc être appelée avant le crop, quel que soit
    le type d'image.
    """
    global ocr_crop_warning_shown
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image OCR illisible : {image_path}")

    image_height, image_width = image.shape[:2]
    if (image_width, image_height) != (640, 480): # Si image non conforme, on ne fait pas d'OCR
        if not ocr_crop_warning_shown:
            print(
                "------------------ Attention ------------------\n"
                "L'échelle ne peut pas être récupérée sur des images déjà crop. "
                "Veuillez utiliser les images originales en 640x480."
                "-----------------------------------------\n"
            )
            ocr_crop_warning_shown = True
        return None

    x, y, w, h = 530, 350, 250, 200
    cropped_img = image[y:y+h, x:x+w]
    resized_img = cv2.resize(cropped_img, (280, 300))
    return pytesseract.image_to_string(resized_img, lang='eng').strip()


echelles_par_poisson = {}
for poisson, echos in df.groupby("cap_id"):
    echelle = 0
    image_originale_trouvee = False
    for file_name in echos["new_name_file"]:
        # Les deux types d'échographie proviennent du même affichage et
        # portent donc la même échelle. Les cavités servent de repli lorsque
        # le poisson n'a pas d'image d'œufs.
        for image_type in ("oeufs", "cavite"):
            image_path = f"../data/COCO/images/{image_type}/{file_name}.jpg"
            if not os.path.exists(image_path):
                continue

            ocr = OCR(image_path)
            if ocr is None:
                continue

            image_originale_trouvee = True
            if ocr == "3,8":
                echelle = 3.8
            elif ocr == "4.7":
                echelle = 4.7
            elif echelle not in (3.8, 4.7):
                echelle = 3.1

    if image_originale_trouvee:
        echelles_par_poisson[poisson] = echelle

# Les noms renommés suivent le format : image_id_heure_cap_id_annee.jpg
poissons_importes = {
    fichier.stem.split("_")[2]
    for dossier in (Path("../data/COCO/images/cavite"), Path("../data/COCO/images/oeufs"))
    for fichier in dossier.iterdir()
    if fichier.is_file() and len(fichier.stem.split("_")) >= 4
}
print(f"Nombre de poissons importés : {len(poissons_importes)}")
print("---------------------------------")


# -------- 4.3 Renomage des masques COCO --------
def rename_json(json_path, output_json_path):
    if os.path.exists(json_path) :
        # Ouvrir le fichier JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for image in data['images']:
            file_name = image['file_name']
            
            if "-" in file_name:
                split_name = file_name.split("_")

                if len(split_name) == 4:
                    heures = split_name[0]
                    cap_id = split_name[2]
                    
                    cap_id = cap_id.split("-")[1]
                    heures = heures.replace("-", ".")
                    heures = heures.split(" ")[0]
                        
                    try:
                        # Récupère l'année
                        annee = str(df.loc[df.image_id == cap_id, 'annee'].item())
                        # Récupère l'identifiant de la capture (poisson)
                        poisson = str(df.loc[df.image_id == cap_id, 'cap_id'].item())
                        
                        # Nouveau nom
                        new_file_name = f"{cap_id}_{heures}_{poisson}_{annee}.jpg"
                        
                        # Mise à jour de la valeur dans le JSON
                        image['file_name'] = new_file_name
                        
                    except ValueError:
                        # Si cap_id non trouvé ou non unique
                        print(f"L'année n'a pas été trouvée. Le nom n'a pas été changé.")

                # Si déjà renommé avant l'annotation
                if len(split_name) == 5:
                    new_file_name = "_".join(split_name[0:4])
                    new_file_name = f"{new_file_name.replace("-",".")}.jpg"
                    image['file_name'] = new_file_name

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    return True

rename_json("../data/COCO/labels/cavite/_annotations.coco.json", "../data/COCO/labels/cavite/annotations.json")
rename_json("../data/COCO/labels/oeufs/_annotations.coco.json", "../data/COCO/labels/oeufs/annotations.oeufs_2_classes.json")

# -------- 4.4 Cropping des images et des masques YOLO --------
def crop(image_paths, yolo_paths, coco_path=None, coco_image_paths=None):
    """Croppe les paires image/masque YOLO et leurs annotations COCO associées."""
    x, y, w, h = 85, 33, 510, 380

    # Vérifie que les images et les masques ont des noms uniques et correspondent
    def paths_by_stem(paths, file_type):
        files = {}
        duplicates = []
        for path in map(Path, paths):
            if path.stem in files:
                duplicates.append(path.stem)
            files[path.stem] = path
        if duplicates:
            raise ValueError(
                f"Plusieurs {file_type} portent le même nom : {sorted(set(duplicates))}"
            )
        return files

    image_files = paths_by_stem(image_paths, "images")
    yolo_files = paths_by_stem(yolo_paths, "fichiers YOLO")
    coco_image_files = paths_by_stem(coco_image_paths or [], "images COCO")
    missing_yolo = sorted(image_files.keys() - yolo_files.keys())
    missing_images = sorted(yolo_files.keys() - image_files.keys())
    if missing_yolo or missing_images:
        raise ValueError(
            f"Paires image/masque incomplètes. Images sans masque : {missing_yolo}; "
            f"masques sans image : {missing_images}."
        )
    if coco_image_paths is not None:
        missing_coco = sorted(image_files.keys() - coco_image_files.keys())
        missing_yolo_images = sorted(coco_image_files.keys() - image_files.keys())
        if missing_coco or missing_yolo_images:
            raise ValueError(
                f"Copies COCO/YOLO incomplètes. Images COCO manquantes : {missing_coco}; "
                f"images YOLO manquantes : {missing_yolo_images}."
            )

    coco_data = None
    coco_images_by_filename = {}
    if coco_path is not None:
        coco_path = Path(coco_path)
        with coco_path.open('r', encoding='utf-8') as coco_file:
            coco_data = json.load(coco_file)

        for coco_image in coco_data['images']:
            filename = Path(coco_image['file_name']).name
            if filename in coco_images_by_filename:
                raise ValueError(f"Plusieurs images COCO portent le nom : {filename}")
            coco_images_by_filename[filename] = coco_image

        missing_coco_images = sorted(
            image_file.name
            for image_file in image_files.values()
            if image_file.name not in coco_images_by_filename
        )
        if missing_coco_images:
            raise ValueError(
                f"Images sans annotation COCO : {missing_coco_images}."
            )

        coco_image_ids = {image['id'] for image in coco_data['images']}
        orphan_annotations = [
            annotation['id']
            for annotation in coco_data['annotations']
            if annotation['image_id'] not in coco_image_ids
        ]
        if orphan_annotations:
            raise ValueError(
                f"Annotations COCO orphelines : {orphan_annotations}."
            )

    for stem, image_file in image_files.items():
        image = cv2.imread(str(image_file))
        if image is None:
            raise ValueError(f"Image YOLO illisible : {image_file}")
        image_height, image_width = image.shape[:2]
        coco_image_file = coco_image_files.get(stem)
        coco_image = None
        if coco_image_file is not None:
            coco_image = cv2.imread(str(coco_image_file))
            if coco_image is None:
                raise ValueError(f"Image COCO illisible : {coco_image_file}")
            if coco_image.shape[:2] != image.shape[:2]:
                raise ValueError(
                    f"Dimensions COCO/YOLO différentes pour {stem} : "
                    f"{coco_image.shape[:2]} contre {image.shape[:2]}."
                )

        # Si l'image est déjà à la bonne taille, on ne fait rien
        if (image_width, image_height) == (w, h):
            continue
        # Si l'image est plus petite que la taille de crop, on ne fait rien
        if x + w > image_width or y + h > image_height:
            raise ValueError(
                f"Le crop ({x}, {y}, {w}, {h}) dépasse les dimensions "
                f"de l'image {image_file} ({image_width}, {image_height})."
            )

        yolo_file = yolo_files[stem]
        lines = []
        # On lit le fichier YOLO et on recalcule les coordonnées des polygones
        with yolo_file.open('r', encoding='utf-8') as yolo:
            for line in yolo:
                parts = line.split()
                if not parts:
                    continue

                class_id = int(parts[0])
                points = list(map(float, parts[1:]))
                if len(points) % 2:
                    raise ValueError(f"Nombre impair de coordonnées dans : {yolo_file}")

                normalized_points = []
                for index in range(0, len(points), 2):
                    point_x = points[index] * image_width - x
                    point_y = points[index + 1] * image_height - y
                    if 0 <= point_x <= w and 0 <= point_y <= h:
                        normalized_points.extend([point_x / w, point_y / h])

                lines.append(
                    f"{class_id} " + " ".join(map(str, normalized_points)) + '\n'
                )

        cropped_image = image[y:y+h, x:x+w]
        if not cv2.imwrite(str(image_file), cropped_image):
            raise OSError(f"Impossible d'écrire l'image cropée : {image_file}")
        if coco_image_file is not None:
            cropped_coco_image = coco_image[y:y+h, x:x+w]
            if not cv2.imwrite(str(coco_image_file), cropped_coco_image):
                raise OSError(f"Impossible d'écrire l'image COCO cropée : {coco_image_file}")
        with yolo_file.open('w', encoding='utf-8') as yolo:
            yolo.writelines(lines)

    # --- Mise à jour des annotations COCO ---
    if coco_data is not None:
        cropped_image_ids = {
            coco_images_by_filename[image_file.name]['id']
            for image_file in image_files.values()
            if (
                coco_images_by_filename[image_file.name]['width'],
                coco_images_by_filename[image_file.name]['height'],
            ) != (w, h)
        }

        for coco_image in coco_images_by_filename.values():
            if coco_image['id'] in cropped_image_ids:
                coco_image['width'] = w
                coco_image['height'] = h

        for annotation in coco_data['annotations']:
            if annotation['image_id'] not in cropped_image_ids:
                continue

            for polygon in annotation['segmentation']:
                for index in range(0, len(polygon), 2):
                    polygon[index] -= x
                    polygon[index + 1] -= y
            annotation['bbox'][0] -= x
            annotation['bbox'][1] -= y

        with coco_path.open('w', encoding='utf-8') as coco_file:
            json.dump(coco_data, coco_file, indent=4)

crop(
    image_paths=glob.glob("../data/YOLO/images/cavite/*.jpg"),
    yolo_paths=glob.glob("../data/YOLO/labels/cavite/*.txt"),
    coco_path="../data/COCO/labels/cavite/annotations.json",
    coco_image_paths=glob.glob("../data/COCO/images/cavite/*.jpg"),
)

crop(
    image_paths=glob.glob("../data/YOLO/images/oeufs/*.jpg"),
    yolo_paths=glob.glob("../data/YOLO/labels/oeufs/*.txt"),
    coco_path="../data/COCO/labels/oeufs/annotations.oeufs_2_classes.json",
    coco_image_paths=glob.glob("../data/COCO/images/oeufs/*.jpg"),
)


# -------- 4.5 Suppression de classes COCO --------
def delete_classes_COCO(input_path, output_path, excluded_categories):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    excluded_ids = {c['id'] for c in data['categories'] if c['name'] in excluded_categories}

    data['categories'] = [c for c in data['categories'] if c['id'] not in excluded_ids]
    data['annotations'] = [a for a in data['annotations'] if a['category_id'] not in excluded_ids]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    return True

# Suppression de l'intestin
delete_classes_COCO(input_path = "../data/COCO/labels/cavite/annotations.json", output_path = "../data/COCO/labels/cavite/annotations_2_classes.json", excluded_categories=["Intestin"])
# Suppression de la gonade
delete_classes_COCO(input_path = "../data/COCO/labels/oeufs/annotations.oeufs_2_classes.json", output_path = "../data/COCO/labels/oeufs/annotations.oeufs.json", excluded_categories=["Gonade"])

# -------- 4.6 Suppression de classes YOLO --------
def delete_classes_YOLO(input_dir, output_dir, excluded_classes):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Filtre les lignes selon les classes exclues
            filtered_lines = [line for line in lines if int(line.split()[0]) not in excluded_classes]

            # Décrémente les classes restantes si nécessaire
            for i in range(len(filtered_lines)):
                parts = filtered_lines[i].split()
                class_id = int(parts[0])
                # Décrémente la classe si elle est supérieure à la plus grande classe exclue
                if class_id > max(excluded_classes):
                    parts[0] = str(class_id - len(excluded_classes))
                    filtered_lines[i] = " ".join(parts) + "\n"
            

            # Écrire les lignes filtrées dans le nouveau fichier
            output_file_path = os.path.join(output_dir, file_name)
            with open(output_file_path, "w") as f:
                f.writelines(filtered_lines)
    return True

# Crée les labels cavité à deux classes en supprimant l'intestin (classe 2).
cavite_2classes_dir = "../data/YOLO/labels/cavite2classes"
os.makedirs(cavite_2classes_dir, exist_ok=True)
delete_classes_YOLO(
    input_dir="../data/YOLO/labels/cavite",
    output_dir=cavite_2classes_dir,
    excluded_classes=[2],
)

# Copie les labels avec toutes les classes dans un sous-dossier dédié
source_dir = "../data/YOLO/labels/oeufs"
backup_dir = "../data/YOLO/labels/oeufsclasses"

# Remplace uniquement les labels polygonaux précédents.
os.makedirs(backup_dir, exist_ok=True)
for previous_label in Path(backup_dir).glob("*.txt"):
    previous_label.unlink()

# Copie le contenu source sans recopier le sous-dossier de destination
for item in os.listdir(source_dir):
    if item == "oeufs2classes" or not item.endswith(".txt"):
        continue

    source_item = os.path.join(source_dir, item)
    destination_item = os.path.join(backup_dir, item)
    if os.path.isdir(source_item):
        shutil.copytree(source_item, destination_item)
    else:
        shutil.copy2(source_item, destination_item)


# Copie les images de la variante à deux classes dans le dossier parallèle attendu par YOLO.
source_images_dir = "../data/YOLO/images/oeufs"
backup_images_dir = "../data/YOLO/images/oeufsclasses"
if os.path.exists(backup_images_dir):
    shutil.rmtree(backup_images_dir)
shutil.copytree(source_images_dir, backup_images_dir)

delete_classes_YOLO(input_dir = "../data/YOLO/labels/oeufs/", output_dir = "../data/YOLO/labels/oeufs/", excluded_classes=[0]) # Suppression la gonade (classe 0)



# ================== 6. Enrichissement du fichier metadata ==================
# -------- 5.1 Création de nouvelles colonne pour obtenir la position de l'échographie --------
df['position'] = df[df['type_image'] != "œufs"]['type_image'].str.replace('+', '', regex=False).str[2:].astype(float)
df["position_ratio"] = df.position / df.long_gonade

# -------- 5.3 Associer une catégorie à chaque échographie --------
# Trie les données par poisson et par position pour s'assurer que les échographies sont dans le bon ordre
df = df.sort_values(by=["cap_id", "position"]).reset_index(drop=True)

# Récupère les identifiants uniques des poissons
unique_id = df["cap_id"].unique()

for id in unique_id:
    # Récupère les lignes correspondant au poisson
    all_fish_rows = df["cap_id"] == id
    rows = all_fish_rows & (df['type_image'] != "œufs") # Ignore les écho d'oeufs

    if rows.any():
        first_row = df[rows].index[0] # Récupère la première échographie
        last_row = df[rows].index[-1] # Récupère la dernière échographie

        # 1. Associe une catégorie à chaque image selon la position de l'échographie
        df.loc[rows, "categorie"] = np.where(df.loc[rows, "position_ratio"] > 1, "erreur", 
                                             np.where(df.loc[rows, "position_ratio"] > 0.6, "fin", 
                                                      np.where(df.loc[rows, "position"] > 1, "milieu", "debut"))
                                             )
        
        # 2. Si deux erreurs consécutives détectées, on ne conserve pas le poisson
        if df.loc[rows, "categorie"].eq("erreur").sum() >= 2:
            df = df.drop(df[rows].index)
            continue # Passe au poisson suivant

        # 3. Gestion des petites erreurs
        df.loc[rows, "position"] = np.where(df.loc[rows, "position"] - df.loc[rows, "long_gonade"] < 0, df.loc[rows, "position"],
                                            np.where(df.loc[rows, "position"] - df.loc[rows, "long_gonade"] < 0.5, df.loc[rows, "long_gonade"], df.loc[rows, "position"])
                                            )
        df.loc[rows, "categorie"] = np.where(df.loc[rows, "position"] - df.loc[rows, "long_gonade"] < 0, df.loc[rows, "categorie"],
                                            np.where(df.loc[rows, "position"] - df.loc[rows, "long_gonade"] < 0.5, "fin", df.loc[rows, "categorie"])
                                            )
        
        # 4. Gestion des grosses erreurs (décalage important)
        if (df.loc[rows, "categorie"]=="erreur").any(): # Décale progressif de l'erreur pour chaque échographie (sauf la première) pour essayer d'être le plus proche de la réalité
            progressions = np.arange(0, len(df[rows])) / (len(df[rows])-1)  # Crée un tableau de progression pour chaque échographie
            total_error = df.loc[last_row, "position"] - df.loc[last_row, "long_gonade"] # L'erreur entre la dernière échographie et la longueur de la gonade
            df.loc[rows, "position"] = df.loc[rows, "position"] - (total_error * progressions) # Décale chaque échographie en fonction de sa position
            
            # Mise à jour la position de la dernière échographie pour qu'elle corresponde à la longueur de la gonade
            df.loc[last_row, "position"] = df.loc[last_row, "long_gonade"]

            # Recalcule le ratio après le décalage
            df.loc[rows, "position_ratio"] = df.loc[rows, "position"] / df.loc[rows, "long_gonade"]

            # Mise à jour des catégories après le décalage
            df.loc[rows, "categorie"] = np.where(df.loc[rows, "position_ratio"] > 1, "erreur", 
                                                 np.where(df.loc[rows, "position_ratio"] > 0.6, "fin", 
                                                          np.where(df.loc[rows, "position"] > 1, "milieu", "debut"))
                                                          )

        # 5. Supprime les doublons de positions pour un même poisson (sauf oeufs)
        duplicates = df.loc[rows].duplicated(subset=["cap_id", "position"], keep="last")
        df = df.drop(df[rows].index[duplicates])

        # 6. Regroupe les catégories dans une nouvelle colonne
        categories = str(list(df.loc[rows, "categorie"]))
        df.loc[rows, "categories"] = categories

# 7. Supprime les lignes vides
df = df.dropna(how = 'all')

print("Catégories de toutes les échographies de gonades :")
print(f"Nombre d'images au début : {len(df[df['categorie'] == 'debut'])}") 
print(f"Nombre d'images au milieu : {len(df[df['categorie'] == 'milieu'])}") 
print(f"Nombre d'images à la fin : {len(df[df['categorie'] == 'fin'])}") 
print(f"Nombre de combinaisons différentes par poisson : {len(df['categories'].value_counts())}")
print("---------------------------------")

# -------- 5.4 Ajoute les informations du poisson --------
df_unique = df[~df.duplicated(subset=["cap_id"], keep = 'first')] # Garde qu'une ligne par poisson
id_unique = df_unique["cap_id"]

# Les âges peuvent être des valeurs textuelles (ex. "2+") ou numériques.
df["age_poisson"] = pd.Series(index=df.index, dtype="object")

for id in id_unique:
    # Récupère la ligne correspondant au poisson
    row = df_cap[df_cap.cap_id == id]

    # Complète df avec le poids, la taile et l'age du poisson
    df.loc[df.cap_id == id, "poids_poisson"] = float(str(row["cap_poids"].values[0]).replace(",", "."))
    df.loc[df.cap_id == id, "long_poisson"] = float(row["cap_lf"].values[0])
    df.loc[df.cap_id == id, "age_poisson"] = row["age_referent"].values[0]


# -------- 5.4 Ajoute les échelles lues avant le crop --------
for id in id_unique :
    echelle = echelles_par_poisson.get(id)

    if echelle == 3.8:
        df.loc[df.cap_id == id, "echelle"] = 3.8
    if echelle == 4.7:
        df.loc[df.cap_id == id, "echelle"] = 4.7
    if echelle == 3.1:
        df.loc[df.cap_id == id, "echelle"] = 3.1
 
# -------- 5.6 Sauvegarde visuellement les images avec leurs annotations --------
def sauvegarder_visualisations(images_dir, labels_dir, output_dir, num_classes):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nb_sauvegardees = 0
    nb_sans_annotation = 0

    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Annotation absente : {label_path.name}")
            nb_sans_annotation += 1
            continue

        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image)
        height, width = image_array.shape[:2]
        mask = load_yolo_polygon_masks(
            label_path,
            (height, width),
            num_classes=num_classes,
        )
        visualisation = overlay_colored_mask(
            image_array,
            mask,
            alpha=0.45,
        )
        Image.fromarray(visualisation).save(output_dir / f"{image_path.stem}.png")
        nb_sauvegardees += 1

    print(f"{nb_sauvegardees} visualisations enregistrées dans {output_dir}")
    if nb_sans_annotation:
        print(f"{nb_sans_annotation} images ignorées car l'annotation est absente.")


# Sauvegarde des visualisations avec toutes les classes
visualisation_dir = Path("../data/visualisation")
sauvegarder_visualisations(
    images_dir="../data/YOLO/images/cavite",
    labels_dir="../data/YOLO/labels/cavite",
    output_dir=visualisation_dir / "images",
    num_classes=3,
)
sauvegarder_visualisations(
    images_dir="../data/YOLO/images/oeufsclasses",
    labels_dir="../data/YOLO/labels/oeufsclasses",
    output_dir=visualisation_dir / "oeufs",
    num_classes=2,
)

#-------- 5.7 Sauvegarde des jeux de données finaux --------
df.to_excel("correspondance_echo_final.xlsx", index=False)


# ================== 7. Division des données de test ==================
# Recalculé à chaque prétraitement : ajouter des poissons peut changer le test.
utils_dir = Path(__file__).resolve().parent
images_dir = utils_dir.parent / "data" / "COCO" / "images"
fish_by_dataset = [
    {
        path.name.split("_")[2]
        for path in (images_dir / dataset).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    for dataset in ("cavite", "oeufs")
]
common_fish = sorted(fish_by_dataset[0] & fish_by_dataset[1])
if not common_fish:
    raise ValueError("Aucun poisson commun aux datasets cavité et œufs pour créer le test.")

test_fish = sorted(np.random.default_rng(42).choice(
    common_fish, size=(len(common_fish) + 5) // 6, replace=False
).tolist())
test_json_path = utils_dir / "common_test_fish.json"
test_json_path.write_text(json.dumps(test_fish, indent=2) + "\n", encoding="utf-8")
print(f"Division du test : {len(test_fish)} poissons sélectionnés parmi {len(common_fish)} poissons communs.")
print(f"Liste enregistrée dans {test_json_path}")
