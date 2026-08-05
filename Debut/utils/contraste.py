import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import os

images = [cv2.imread(file) for file in glob.glob("../Methodo_volumes/data/*.jpg")]

# Image luminosité
def boundedPixelValue(color, brightnessFactor):
	scaledValue = float(color * (1 + brightnessFactor))
	if scaledValue < 0:
		return 0
	elif scaledValue > 255:
		return 255

	return int(scaledValue)

count = 0
for i in range(1, 108):
    count += 1
    im = Image.open(f"cropped_img_{count}.jpg")   
    out = Image.new('RGB', im.size, 0xffffff)

    brightnessFactor = 0.4

    width, height = im.size
    for x in range(width):
        for y in range(height):
            r,g,b = im.getpixel((x,y))

            updatedR = boundedPixelValue(r, brightnessFactor)
            updatedG = boundedPixelValue(g, brightnessFactor)
            updatedB = boundedPixelValue(b, brightnessFactor)

            out.putpixel((x,y), (updatedR, updatedG, updatedB))

    out.save(f'brightnessScaled_{count}.jpg')

# Normalization of the images
img = cv2.imread('brightnessScaled_1.jpg')

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
normalized_gray_image = cv2.normalize(
    gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
normalized_color_image = cv2.cvtColor(
    normalized_gray_image, cv2.COLOR_GRAY2BGR)
# Save the normalized image
cv2.imwrite(f"normalized_img.jpg", normalized_color_image) # Même image
