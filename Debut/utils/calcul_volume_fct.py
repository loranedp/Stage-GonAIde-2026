# ================ Calcul du volume des gonades ================
# ---- Importation packages ----
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os


# ---- Récupération du nombre de pixel (surface polygon) -----
def calculate_polygon_area(yolo_coords, image_width, image_height):
    """ La formule d'aire de Gauss est une méthode pour calculer
    l'aire d'un polygone quelconque à partir des coordonnées de ses sommets.
    """
    # Dénormaliser les coordonnées : coordonnées en pixels
    coords = []
    for i in range(0, len(yolo_coords), 2):
        x = yolo_coords[i] * image_width
        y = yolo_coords[i+1] * image_height
        coords.append((x, y))

    # Appliquer la formule du shoelace
    n = len(coords)
    area = 0.0
    for i in range(n):
        x_i, y_i = coords[i]
        x_j, y_j = coords[(i + 1) % n]
        area += (x_i * y_j) - (x_j * y_i)

    area = abs(area) / 2.0
    return area

# ---- Calcul de la surface : images transversales -----
def calculate_areas(mask_path, image_path, echelle):
    # Importer l'image
    image = cv2.imread(image_path)
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    # Importer le masque YOLO
    with open(mask_path, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]

    # --- Calculer l'aire de chaque instance et les sommer par classe ---
    # (une classe peut avoir plusieurs instances, ex. plusieurs gonades)
    area_gonade = 0.0
    area_cavite = 0.0
    for line in lines:
        values = [float(x) for x in line.split()]
        class_id = int(values[0])
        yolo_coords = values[1:]
        area = calculate_polygon_area(yolo_coords, image_width, image_height)

        if class_id == 0:
            area_cavite += area
        elif class_id == 1:
            area_gonade += area

    # ---- Transformation des pixels en mm ----
    surface_cm2_gonade = (echelle / image_height) ** 2 * area_gonade
    surface_cm2_cavite = (echelle / image_height) ** 2 * area_cavite

    return surface_cm2_gonade, surface_cm2_cavite

# ---- Calcul du volume : images transversales -----
def calculate_volumes(df, id, path):
    #---- Initialisation des variables ----
    resultats = {"id poisson": [], "image_id": [], "position echo": [], "surface_cavite": [], "surface_gonade": [], "volume_cavite": [], 
             "volume_gonade": [], "echelle": [], "augmentation_surface_gonade": [], "augmentation_surface_cavite": []}
    
    last_row = None
    last_rayon_gonade = None
    last_rayon_cavite = None
    last_surface_gonade = None
    last_surface_cavite = None
    volumes_gonade_list = []
    volumes_cavite_list = []
    count = 0

    df = df[df["type_image"] != "œufs"] # Filtrer les échos sans type "oeufs"

    # ---- On parcours chaque échographie du poisson ----
    for index, row in df.iterrows():
        file_path = row["new_name_file"]

        # --- Importation des images et des masques ---
        mask_path = f"../{path}{file_path}.txt"
        image_path = f"../data/images/cavite/{file_path}.jpg"

        if os.path.exists(mask_path) : # Vérifie si le fichier existe
            count += 1
            echelle = row["echelle"] # Récupère l'échelle de l'image
            position = row["position"] # Récupère la position de l'écho

            # --- Calcul de la surface des gonades et de la cavité ---
            surface_gonade, surface_cavite = calculate_areas(mask_path, image_path, echelle)

            # ---- Calcul du rayon du cercle à partir des surfaces ----
            rayon_gonade = np.sqrt(surface_gonade/np.pi)
            rayon_cavite = np.sqrt(surface_cavite/np.pi)

            # ---- Calcul de l'augmentation (%) de la surface de la gonade par rapport à l'écho précédente ----
            if last_row is not None :
                augmentation_surface_gonade = (
                    round((surface_gonade - last_surface_gonade) / last_surface_gonade * 100, 0)
                    if last_surface_gonade != 0 else np.nan
                )
                augmentation_surface_cavite = (
                    round((surface_cavite - last_surface_cavite) / last_surface_cavite * 100, 0)
                    if last_surface_cavite != 0 else np.nan
                )
            else :
                augmentation_surface_gonade = 0
                augmentation_surface_cavite = 0

            # ---- Calcul du volume des cônes tronqués ----
            if last_row is not None : # Si pas la première échographie
                distance = row["position"] - last_row["position"]

                volume_gonade = 1/3 * np.pi * (last_rayon_gonade**2 + rayon_gonade**2 + last_rayon_gonade * rayon_gonade) * distance
                volume_cavite = 1/3 * np.pi * (last_rayon_cavite**2 + rayon_cavite**2 + last_rayon_cavite * rayon_cavite) * distance 

                volumes_gonade_list.append(volume_gonade)
                volumes_cavite_list.append(volume_cavite)

            elif position > 0 : # Si première échographie mais pas à la position 0, alors on calcul le volume du cône depuis l'opercule
                volume_gonade = np.pi * rayon_gonade**2 * position / 3 # Formule d'un cône
                volume_cavite = np.pi * rayon_cavite**2 * position / 3 # Formule d'un cône

                volumes_gonade_list.append(volume_gonade)
                volumes_cavite_list.append(volume_cavite)

            last_row = row
            last_rayon_gonade = rayon_gonade
            last_rayon_cavite = rayon_cavite
            last_surface_gonade = surface_gonade
            last_surface_cavite = surface_cavite
            print(f"Position de l'écho : {row['position']}, surface gonades : {surface_gonade:.2f} cm2, rayon : {rayon_gonade:.2f} cm, surface cavité : {surface_cavite:.2f} cm2")

            resultats["id poisson"].append(str(id))
            resultats["image_id"].append(row["image_id"])
            resultats["position echo"].append(row["position"])
            resultats["surface_cavite"].append(surface_cavite)
            resultats["surface_gonade"].append(surface_gonade)
            resultats["echelle"].append(echelle)
            resultats["augmentation_surface_gonade"].append(augmentation_surface_gonade)
            resultats["augmentation_surface_cavite"].append(augmentation_surface_cavite)


    # --- Calcul du volume total des gonades et de la cavité ---
    if count > 1:
        # ---- Calcul du cône final jusqu'à la fin de la gonade----
        volume_gonade = np.pi * last_rayon_gonade**2 * abs(last_row["position"] - last_row["long_gonade"]) / 3 # Formule d'un cône
        volumes_gonade_list.append(volume_gonade)

        volume_cavite = np.pi * last_rayon_cavite**2 * abs(last_row["position"] - last_row["long_gonade"]) / 3 # Formule d'un cône
        volumes_cavite_list.append(volume_cavite)

        # --- calcul volume final ----
        volume_total_gonade = np.sum(volumes_gonade_list)
        volume_total_cavite = np.sum(volumes_cavite_list)

        print(f"Poisson n°: {id}, Nombre d'échos : {count}/{len(df)}, Volume total gonades: {volume_total_gonade:.2f} cm3, volume total cavité: {volume_total_cavite:.2f} cm3")
        print("-----------------------")

        resultats["volume_cavite"].extend([volume_total_cavite] * count) # Ajoute le volume total de la cavité pour chaque échographie
        resultats["volume_gonade"].extend([volume_total_gonade] * count)

    elif count == 1:
        print("Pas assez d'échos pour calculer le volume des gonades.")
        print(f"Poisson n°: {id}, Nombre d'échos : {count}/{len(df)}")
        print("-----------------------")
        volume_total_gonade = "NA"
        volume_total_cavite = "NA"

        resultats["volume_cavite"].append(0)
        resultats["volume_gonade"].append(0)

    return resultats


# ---- Calcul de la surface : images longitudinales -----
def calculate_eggs_areas(mask_path, image_path, echelle):
    """Calcule la surface moyenne des œufs en mm².

    ``echelle`` reste fournie en centimètres sur la hauteur de l'image.
    """
    # Importer l'image
    image = cv2.imread(image_path)
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    # Importer le masque YOLO
    with open(mask_path, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]

    # --- Calculer l'aire de chaque instance et faire la moyenne des 2 plus grandes surfaces ---
    area_eggs = []
    for line in lines:
        values = [float(x) for x in line.split()]
        yolo_coords = values[1:]
        area = calculate_polygon_area(yolo_coords, image_width, image_height)
        area_eggs.append(area)

    # --- Prendre les 50% plus grandes surfaces ---
    area_eggs.sort(reverse=True)
    area_eggs_max = area_eggs[:max(1, int(0.5 * len(area_eggs)))]

    # ---- Faire la moyenne des surfaces sélectionnées ----
    mean_area_eggs = np.mean(area_eggs_max) if area_eggs_max else 0

    # ---- Transformation des pixels en mm² ----
    surface_mm2_eggs = ((echelle * 10) / image_height) ** 2 * mean_area_eggs
    area_eggs = [((echelle * 10) / image_height) ** 2 * area for area in area_eggs]

    return surface_mm2_eggs, area_eggs


def calculate_eggs_area_from_instances(instance_labels, echelle, image_height):
    """Calcule en mm² la surface moyenne des 50 % plus grandes instances d'œufs.
    """
    labels = np.asarray(instance_labels)

    areas = [
        int(np.count_nonzero(labels == label_id))
        for label_id in np.unique(labels)
        if label_id > 0
    ]
    if not areas:
        return 0.0
    areas.sort(reverse=True)
    selected = areas[: max(1, int(0.5 * len(areas)))]
    mean_area_pixels = float(np.mean(selected))
    return ((float(echelle) * 10) / float(image_height)) ** 2 * mean_area_pixels


def count_egg_instances(instance_labels):
    """Compte les identifiants d'œufs distincts (strictement positifs)."""
    labels = np.asarray(instance_labels)
    if labels.ndim != 2:
        raise ValueError("La carte d'instances d'œufs doit être bidimensionnelle.")
    return int(np.count_nonzero(np.unique(labels) > 0))


def calculate_egg_volume_from_instances(instance_labels, echelle, image_height):
    """Volume sphérique en mm³ associé à la surface moyenne des œufs."""
    surface_mm2 = calculate_eggs_area_from_instances(
        instance_labels, echelle, image_height
    )
    egg_count = count_egg_instances(instance_labels)
    selected_egg_count = max(1, int(0.5 * egg_count)) if egg_count else 0
    radius_mm = np.sqrt(surface_mm2 / np.pi)
    volume_mm3 = 4 / 3 * np.pi * radius_mm**3
    return {
        "surface_moyenne_oeufs_mm2": float(surface_mm2),
        "volume_moyen_oeufs_mm3": float(volume_mm3),
        "nombre_oeufs_distincts": egg_count,
        "nombre_oeufs_utilises_pour_moyenne": selected_egg_count,
    }


def calculate_mean_egg_volume(images):
    """Agrège le volume moyen d'œuf sur plusieurs échographies.

    Chaque élément contient ``id_image``, ``instances``, ``echelle`` et
    ``image_height``. Les images sans instance donnent un volume nul dans le
    détail mais ne permettent pas à elles seules de produire une fécondité.
    """
    rows = []
    for item in images:
        values = calculate_egg_volume_from_instances(
            item["instances"], item["echelle"], item["image_height"]
        )
        rows.append({"id_image": item["id_image"], **values})
    volumes = [row["volume_moyen_oeufs_mm3"] for row in rows]
    mean_volume = float(np.mean(volumes)) if volumes else None
    return {"images": rows, "volume_moyen_oeufs_mm3": mean_volume}


# ---- Calcul du volume : images longitudinales -----
def calculate_eggs_volumes(df, id, path):
    #---- Initialisation des variables ----
    resultats = {
        "id poisson": [],
        "image_id": [],
        "position echo": [],
        "surface_moyenne_oeufs_mm2": [],
        "volume_moyen_oeufs_mm3": [],
        "echelle": [],
    }
    
    volumes_oeufs_list = []
    count = 0
    surfaces_list = []

    df = df[df["type_image"] == "œufs"] # Filtrer les écho de type "oeufs"

    # ---- On parcours chaque échographie du poisson ----
    for index, row in df.iterrows():
        file_path = row["new_name_file"]

        # --- Importation des images et des masques ---
        mask_path = f"../{path}{file_path}.txt"
        image_path = f"../data/images/oeufs/{file_path}.jpg"

        if os.path.exists(mask_path) : # Vérifie si le fichier existe
            count += 1
            echelle = row["echelle"] # Récupère l'échelle de l'image

            # --- Calcul de la surface moyenne des oeufs ---
            surface_oeufs_mm2, surfaces_list = calculate_eggs_areas(mask_path, image_path, echelle)

            # ---- Calcul du rayon en mm à partir de la surface en mm² ----
            rayon_oeufs_mm = np.sqrt(surface_oeufs_mm2 / np.pi)

            # ---- Calcul du volume des sphères en mm³ ----
            volume_oeufs_mm3 = 4/3 * np.pi * rayon_oeufs_mm**3

            volumes_oeufs_list.append(volume_oeufs_mm3)

            print(
                f"Echo n° : {count}, surface oeufs : {surface_oeufs_mm2:.3f} mm², "
                f"rayon : {rayon_oeufs_mm:.3f} mm, "
                f"volume : {volume_oeufs_mm3:.4f} mm³"
            )

            resultats["id poisson"].append(str(id))
            resultats["image_id"].append(row["image_id"])
            resultats["surface_moyenne_oeufs_mm2"].append(surface_oeufs_mm2)
            resultats["volume_moyen_oeufs_mm3"].append(volume_oeufs_mm3)
            resultats["echelle"].append(echelle)
            resultats["position echo"].append(row["type_image"])

    # --- Calcul du volume total des oeufs ---
    if np.sum(volumes_oeufs_list) != 0:
       # Faire la moyenne des volumes calculés pour chaque échographie dont le volume est non nul
        volume_total_oeufs = np.mean(np.array(volumes_oeufs_list)[np.array(volumes_oeufs_list) != 0])

        print(
            f"Poisson n°: {id}, Nombre d'échos : {count}/{len(df)}, "
            f"Volume moyen oeufs: {volume_total_oeufs:.4f} mm³"
        )
        print("-----------------------")

    return resultats, surfaces_list
