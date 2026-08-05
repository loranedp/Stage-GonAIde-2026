import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image
import pandas as pd
import os


# 2. Charger l'image depuis le disque
chemin_entree = '0004579_10.51.07_2019.jpg'
img = cv2.imread(chemin_entree)

x, y, w, h = 579, 393, 12, 18  # crop left, crop top, size right, size bottom

crop = img[y:y+h, x:x+w]
# Save l'image croped et redimensionnée
cv2.imwrite("cropped_image.png", crop)

text = pytesseract.image_to_string(crop,lang='eng').strip()
print(text)


