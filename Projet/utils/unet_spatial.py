"""Contrat spatial commun aux pipelines U-Net.

Les données ont déjà été recadrées par le pré-traitement : elles font 510 x
380 pixels.  Les modèles reçoivent ce contenu sans déformation, centré dans
un canevas 512 x 512 noir.
"""

import numpy as np
import torch
from PIL import Image


CROPPED_IMAGE_SIZE = (510, 380)  # (W, H)
MODEL_IMAGE_SIZE = (512, 512)  # (W, H)
CONTENT_OFFSET = (1, 66)  # (X, Y) dans le canevas modèle
CONTENT_BOUNDS = (0, 0, *CROPPED_IMAGE_SIZE)


def validate_cropped_size(size: tuple[int, int]) -> None:
    if tuple(size) != CROPPED_IMAGE_SIZE:
        raise ValueError(
            "Les images U-Net doivent déjà être cropées en "
            f"{CROPPED_IMAGE_SIZE[0]}x{CROPPED_IMAGE_SIZE[1]} ; reçu "
            f"{size[0]}x{size[1]}."
        )


def pad_image(image: Image.Image, fill=0) -> Image.Image:
    """Centre une image cropée dans le canevas du modèle, sans resize."""
    validate_cropped_size(image.size)
    padded = Image.new(image.mode, MODEL_IMAGE_SIZE, color=fill)
    padded.paste(image, CONTENT_OFFSET)
    return padded


def pad_array(array: np.ndarray, fill=0) -> np.ndarray:
    """Padde un tableau (..., H, W) en conservant toutes ses dimensions de tête."""
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError("Le tableau à padder doit avoir au moins deux dimensions.")
    height, width = array.shape[-2:]
    validate_cropped_size((width, height))
    out = np.full((*array.shape[:-2], MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0]), fill, array.dtype)
    x, y = CONTENT_OFFSET
    out[..., y : y + height, x : x + width] = array
    return out


def unpad_array(array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Extrait le contenu 510 x 380 en préservant le type de l'entrée.

    Un tenseur PyTorch reste un tenseur afin de pouvoir être transmis aux
    consommateurs qui attendent encore ``.cpu().numpy()``. Les entrées NumPy
    (ou convertibles en NumPy) gardent le comportement historique.
    """
    is_tensor = isinstance(array, torch.Tensor)
    shape = tuple(array.shape)
    if len(shape) < 2 or shape[-2:] != MODEL_IMAGE_SIZE[::-1]:
        raise ValueError(
            "Le tableau à dépaddder doit finir par les dimensions "
            f"{MODEL_IMAGE_SIZE[::-1]} ; reçu {shape}."
        )
    x, y = CONTENT_OFFSET
    width, height = CROPPED_IMAGE_SIZE
    if is_tensor:
        return array[..., y : y + height, x : x + width]

    array = np.asarray(array)
    return array[..., y : y + height, x : x + width]
