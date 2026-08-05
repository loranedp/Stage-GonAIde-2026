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
