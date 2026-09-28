"""Calcul du volume des gonades à partir des masques prédits par le modèle.
"""

import numpy as np
import pytesseract
from PIL import Image

import config


def surface_cm2(gonad_mask: np.ndarray, echelle_cm: float, image_height_px: int) -> float:
    """Surface (cm²) du masque gonade, à partir de l'échelle OCR (cm sur la hauteur de l'image)."""
    return (echelle_cm / image_height_px) ** 2 * float(np.count_nonzero(gonad_mask))


def ocr_echelle(image: Image.Image) -> float | None:
    """Lit par OCR la valeur d'échelle (cm) affichée à l'écran sur une image brute uploadée
    (zone fixe config.OCR_SCALE_CROP). Retourne None si l'OCR échoue ou ne renvoie pas un nombre."""
    x, y, w, h = config.OCR_SCALE_CROP
    array = np.array(image.convert("RGB"))
    cropped = array[y : y + h, x : x + w]
    if cropped.size == 0:
        return None
    resized = np.array(Image.fromarray(cropped).resize((280, 300), Image.Resampling.BILINEAR))
    text = pytesseract.image_to_string(resized, lang="eng").strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def determine_fish_echelle(images: list) -> float:
    """Détermine l'échelle (cm) d'un poisson par OCR sur l'ensemble de ses échographies
    (un poisson n'a qu'une seule échelle ; l'OCR échoue parfois sur une image isolée, d'où
    l'agrégation). Priorité : 3.8 si lue sur au moins une image, sinon 4.7 si lue sur au
    moins une image, sinon 3.1 par défaut."""
    ocr_values = [ocr_echelle(image) for image in images]
    if any(v is not None and abs(v - 3.8) < 1e-6 for v in ocr_values):
        return 3.8
    if any(v is not None and abs(v - 4.7) < 1e-6 for v in ocr_values):
        return 4.7
    return 3.1


def compute_fish_volume(images_ordered: list, surface_key: str = "surface_cm2") -> dict:
    """Calcule le volume total (troncs de cône + cône final) pour un poisson.

    images_ordered : liste de dicts triés par position croissante, chacun
    contenant {surface_key: float, "position": float, "id_image": str}.
    Le dernier élément doit aussi contenir "long_gonade" (constant par poisson).

    surface_key : clé de surface à utiliser (ex. "surface_cm2" pour la gonade,
    "surface_cavite_cm2" pour la cavité). Le terme de cône final réutilise
    systématiquement "long_gonade" comme approximation de la longueur totale,
    faute de métadonnée équivalente pour la cavité.

    Reprend la logique de troncs de cône successifs + cône final du script
    "Calcul volume/calcul_volume_gonade.py".
    """
    if len(images_ordered) < 2:
        return {"surfaces": images_ordered, "volume_total_cm3": None}

    volumes = []
    last_position = None
    last_rayon = None
    for item in images_ordered:
        rayon = np.sqrt(item[surface_key] / np.pi)
        if last_position is not None:
            distance = item["position"] - last_position
            volume = (
                1 / 3 * np.pi * (last_rayon**2 + rayon**2 + last_rayon * rayon) * distance
            )
            volumes.append(volume)
        last_position = item["position"]
        last_rayon = rayon

    last_item = images_ordered[-1]
    final_volume = (
        np.pi * last_rayon**2 * abs(last_item["position"] - last_item["long_gonade"]) / 3
    )
    volumes.append(final_volume)

    return {"surfaces": images_ordered, "volume_total_cm3": float(np.sum(volumes))}


def detect_surface_gonade_anomalies(images_ordered: list) -> set:
    """Détecte les id_image dont la surface de gonade est potentiellement incohérente.

    Pour un poisson, la surface de gonade ne doit pas diminuer entre Op1 et Op2, ni augmenter de plus de
    10% entre deux échos au-delà d'Op2. Nécessite au moins 3 images ; sinon set() vide.

    images_ordered : liste de dicts triés par position croissante, chacun contenant
    {"id_image": str, "surface_cm2": float, "position": float}.
    """
    if len(images_ordered) <= 2:
        return set()

    pb_list = []
    last_surface = None
    for item in images_ordered:
        surface = item["surface_cm2"]
        if last_surface in (None, 0):
            augmentation = 0.0
        else:
            augmentation = round((surface - last_surface) / last_surface * 100)
        last_surface = surface

        position = item["position"]
        if position == 2.0 or position == 3.0:
            pb_list.append(1 if augmentation < 0 else 0)
        elif position > 3.0:
            pb_list.append(1 if augmentation > 10 else 0)
        else:
            pb_list.append(0)

    flagged = set()
    for i in range(len(pb_list) - 1):
        if pb_list[i] == 1 and pb_list[i + 1] == 1:
            flagged.add(images_ordered[i]["id_image"])
            break
    if pb_list[1] == 1 and pb_list[0] == 0 and pb_list[2] == 0:
        flagged.add(images_ordered[0]["id_image"])

    return flagged
