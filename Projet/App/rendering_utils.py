"""Fonctions de rendu d'images indépendantes de l'interface Streamlit."""

import cv2
import numpy as np
from PIL import Image


EGG_OUTLINE_COLOR = (0, 0, 0)
EGG_OUTLINE_THICKNESS = 1


def draw_instance_contours(
    image: Image.Image,
    instances: np.ndarray,
    color: tuple[int, int, int] = EGG_OUTLINE_COLOR,
    thickness: int = EGG_OUTLINE_THICKNESS,
) -> Image.Image:
    """Trace le contour de chaque instance non nulle sur une copie RGB de ``image``."""
    instance_array = np.asarray(instances)
    rendered = np.array(image.convert("RGB"), copy=True)

    if instance_array.ndim != 2:
        raise ValueError("La carte d'instances doit être bidimensionnelle.")
    if instance_array.shape != rendered.shape[:2]:
        raise ValueError(
            "La carte d'instances et l'image doivent avoir les mêmes dimensions."
        )
    if thickness < 1:
        raise ValueError("L'épaisseur du contour doit être strictement positive.")

    for instance_id in np.unique(instance_array):
        if instance_id <= 0:
            continue
        instance_mask = np.ascontiguousarray(
            instance_array == instance_id, dtype=np.uint8
        )
        contours, _ = cv2.findContours(
            instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            rendered,
            contours,
            contourIdx=-1,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )

    return Image.fromarray(rendered)
