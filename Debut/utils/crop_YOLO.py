import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

# Chemins des images et des masques YOLO
image_files = sorted(glob.glob("crop/Gonades/*.jpg"))
mask_files = sorted(glob.glob("crop/Gonades/*.txt"))

# Paramètres de crop
x, y, w, h = 85, 33, 510, 380  # crop left, crop top, size right, size bottom
new_width, new_height = 640, 480
original_width, original_height = 640, 480

# Crop and resize each image and each mask, then return the results
count = 0
for image_file in image_files:
    count += 1
    lst = []

    # 1. Crop et redimensionnement de l'image
    image = cv2.imread(image_file)
    cropped_img = image[y:y+h, x:x+w] # crop
    resized_img = cv2.resize(cropped_img, (new_width, new_height)) # Redimensionne
    cv2.imwrite(image_file, resized_img) # Save

    # 2. Lecture des points du masque
    if len(mask_files) > 0:
        mask_file = mask_files[count-1]
        with open(mask_file, 'r') as f:
            for line in f:
                parts = list(map(float, line.split()))

                # Le premier élément est l'ID de classe (0), le reste sont les coordonnées x,y
                class_id = int(parts[0])
                points = parts[1:]

                # Dénormaliser les points pour les convertir en pixels
                denormalized_points = []
                for i in range(0, len(points), 2):
                    px = points[i] * original_width
                    py = points[i+1] * original_height
                    denormalized_points.extend([px, py])

                # Appliquer le crop : soustraire x et y, et filtrer les points hors cadre
                cropped_points = []
                for i in range(0, len(denormalized_points), 2):
                    px = denormalized_points[i] - x
                    py = denormalized_points[i+1] - y
                    # Vérifier si le point est dans le crop
                    if 0 <= px <= w and 0 <= py <= h:
                        cropped_points.extend([px, py])

                # Normaliser à nouveau pour la nouvelle taille
                normalized_points = []
                for i in range(0, len(cropped_points), 2):
                    px = cropped_points[i] / w
                    py = cropped_points[i+1] / h
                    normalized_points.extend([px, py])

                lst.append(f"{class_id} " + " ".join(map(str, normalized_points)) + '\n')

        # Écrire le nouveau masque
        with open(mask_file, 'w') as f:
            f.writelines(lst)
            f.close()




