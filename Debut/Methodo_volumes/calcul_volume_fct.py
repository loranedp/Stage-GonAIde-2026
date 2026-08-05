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

# ---- Calcul de la surface -----
def calculate_areas(mask_path, image_path, echelle):
    # Importer l'image
    image = cv2.imread(image_path)
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    # Importer le masque YOLO
    with open(mask_path, "r") as f:
        lines = f.readlines()

    # --- Récupérer les coordonnées des gonades et de la cavité ---
    coords_gonade = [float(x) for x in lines[1].split()]
    yolo_coords_gonade = coords_gonade[1:]
    # Récupérer les coordonnées de la cavité
    coords_cavite = [float(x) for x in lines[0].split()]
    yolo_coords_cavite = coords_cavite[1:]

    # --- Calculer l'aire du polygon ---
    area_gonade = calculate_polygon_area(yolo_coords_gonade, image_width, image_height)
    area_cavite = calculate_polygon_area(yolo_coords_cavite, image_width, image_height)

    # ---- Transformation des pixels en mm ----
    surface_cm2_gonade = (echelle / image_height) ** 2 * area_gonade
    surface_cm2_cavite = (echelle / image_height) ** 2 * area_cavite

    return surface_cm2_gonade, surface_cm2_cavite


def calculate_volumes(df, id, path):
    #---- Initialisation des variables ----
    resultats = {"id poisson": [], "position echo": [], "surface_cavite": [], "surface_gonade": [], "volume_cavite": [], 
             "volume_gonade": [], "echelle": [], "augmentation_surface_gonade": [], "augmentation_surface_cavite": []}
    
    last_row = None
    last_rayon_gonade = None
    last_rayon_cavite = None
    last_surface_gonade = None
    last_surface_cavite = None
    volumes_gonade_list = []
    volumes_cavite_list = []
    count = 0

    # ---- On parcours chaque échographie du poisson ----
    for index, row in df.iterrows():
        file_path = row["new_name_file"]

        # --- Importation des images et des masques ---
        mask_path = f"../{path}{file_path}.txt"
        image_path = f"../data/COCO/images/{file_path}.jpg"

        if os.path.exists(mask_path) : # Vérifie si le fichier existe
            count += 1
            echelle = row["echelle"] # Récupère l'échelle de l'image

            # --- Calcul de la surface des gonades et de la cavité ---
            surface_gonade, surface_cavite = calculate_areas(mask_path, image_path, echelle)

            # ---- Calcul du rayon du cercle à partir des surfaces ----
            rayon_gonade = np.sqrt(surface_gonade/np.pi)
            rayon_cavite = np.sqrt(surface_cavite/np.pi)

            # ---- Calcul de l'augmentation (%) de la surface de la gonade par rapport à l'écho précédente ----
            if last_row is not None :
                augmentation_surface_gonade = round((surface_gonade - last_surface_gonade) / last_surface_gonade * 100, 0)
                augmentation_surface_cavite = round((surface_cavite - last_surface_cavite) / last_surface_cavite * 100, 0)
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

            last_row = row
            last_rayon_gonade = rayon_gonade
            last_rayon_cavite = rayon_cavite
            last_surface_gonade = surface_gonade
            last_surface_cavite = surface_cavite
            print(f"Position de l'écho : {row['position']}, surface gonades : {surface_gonade:.2f} cm2, rayon : {rayon_gonade:.2f} cm, surface cavité : {surface_cavite:.2f} cm2")

            resultats["id poisson"].append(id)
            resultats["position echo"].append(row["position"])
            resultats["surface_cavite"].append(surface_cavite)
            resultats["surface_gonade"].append(surface_gonade)
            resultats["echelle"].append(echelle)
            resultats["augmentation_surface_gonade"].append(augmentation_surface_gonade)
            resultats["augmentation_surface_cavite"].append(augmentation_surface_cavite)

    # --- Calcul du volume total des gonades et de la cavité ---
    if np.sum(volumes_gonade_list) != 0:
        # ---- Calcul du cône final ----
        volume_gonade = np.pi * rayon_gonade**2 * abs(row["position"] - row["long_gonade"]) / 3 # Formule d'un cône
        volumes_gonade_list.append(volume_gonade)

        volume_cavite = np.pi * rayon_cavite**2 * abs(row["position"] - row["long_gonade"]) / 3 # Formule d'un cône
        volumes_cavite_list.append(volume_cavite)

        # --- calcul volume final ----
        volume_total_gonade = np.sum(volumes_gonade_list)
        volume_total_cavite = np.sum(volumes_cavite_list)

        print(f"Poisson n°: {id}, Nombre d'échos : {count}/{len(df)}, Volume total gonades: {volume_total_gonade:.2f} cm3, volume total cavité: {volume_total_cavite:.2f} cm3")
        print("-----------------------")

        resultats["volume_cavite"].extend([volume_total_cavite] * count) # Ajoute le volume total de la cavité pour chaque échographie
        resultats["volume_gonade"].extend([volume_total_gonade] * count)

    if count == 1:
        print("Pas assez d'échos pour calculer le volume des gonades.")
        print(f"Poisson n°: {id}, Nombre d'échos : {count}/{len(df)}")
        print("-----------------------")
        volume_total_gonade = "NA"
        volume_total_cavite = "NA"

        resultats["volume_cavite"].append(0)
        resultats["volume_gonade"].append(0)

    return resultats