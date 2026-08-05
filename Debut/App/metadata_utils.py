"""Lecture des métadonnées par image depuis utils/correspondance_echo_final.xlsx.

Fournit, par id_image (identique à la colonne "image_id" du fichier) :
  - la catégorie CME (debut/fin/milieu) utilisée comme canaux additionnels du modèle,
  - la position de la sonde et la longueur de gonade utilisées par le calcul de volume.
"""

import streamlit as st
import pandas as pd

import config

# Ordre alphabétique, identique à celui produit par pd.get_dummies() lors de
# l'entraînement (vérifié empiriquement : ['debut', 'fin', 'milieu']).
CME_CATEGORIES = ["debut", "fin", "milieu"]


def _find_categorie_column(columns) -> str:
    """Le nom de colonne 'catégorie' est mal encodé dans le fichier Excel source
    (caractère de remplacement à la place de 'é') ; on le retrouve par motif plutôt
    que par comparaison exacte de chaîne."""
    for col in columns:
        if col.lower().startswith("cat") and col.lower().endswith("gorie"):
            return col
    raise KeyError("Colonne 'catégorie' introuvable dans utils/correspondance_echo_final.xlsx")


@st.cache_data
def _load() -> pd.DataFrame:
    df = pd.read_excel(
        config.METADATA_XLSX_PATH,
        dtype={"annee": str, "image_id": str, "new_name_file": str},
    )
    categorie_col = _find_categorie_column(df.columns)
    df = df.rename(columns={categorie_col: "categorie"})
    df = df.drop_duplicates(subset="image_id", keep="first").set_index("image_id")
    return df


def lookup(id_image: str) -> dict | None:
    """Retourne {cap_id, position, long_gonade, long_poisson, poids_poisson, age_poisson, categorie, annee, type_image} pour cet
    id_image, ou None si absent du fichier de métadonnées."""
    df = _load()
    if id_image not in df.index:
        return None
    row = df.loc[id_image]
    return {
        "cap_id": row["cap_id"],
        "position": float(row["position"]),
        "long_gonade": float(row["long_gonade"]),
        "long_poisson": float(row["long_poisson"]),
        "poids_poisson": float(row["poids_poisson"]),
        "age_poisson": row["age_poisson"],
        "categorie": row["categorie"],
        "annee": row["annee"],
        "type_image": row["type_image"],
        "echelle": float(row["echelle"]),
    }
