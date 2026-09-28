"""Parsing des noms de fichiers uploadés : idimage_HH.MM.SS_idpoisson_annee.jpg"""

from pathlib import Path
from typing import NamedTuple


class ParsedFilename(NamedTuple):
    id_image: str
    heure: str
    id_poisson: str


def parse_filename(filename: str) -> ParsedFilename:
    """Extrait (id_image, heure, id_poisson) d'un nom au format idimage_HH.MM.SS_idpoisson_annee.jpg.

    Lève ValueError si le nom ne contient pas exactement 4 parties séparées par "_".
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"Nom de fichier invalide : '{filename}'. Format attendu : "
            "idimage_HH.MM.SS_idpoisson_annee.jpg"
        )
    id_image, heure, id_poisson, annee = parts
    return ParsedFilename(id_image=id_image, heure=heure, id_poisson=id_poisson)
