"""Test commun stratifié par longueur de gonade, au niveau du poisson."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


COMMON_TEST_PATH = Path(__file__).resolve().parent / "common_test_fish.json"


def compute_common_test_fish(metadata, cavity_fish, egg_fish):
    """Retourne le premier des six folds, sur trois quantiles de long_gonade."""
    fish = sorted(set(map(str, cavity_fish)) & set(map(str, egg_fish)))
    if not fish:
        raise ValueError("Aucun poisson commun aux datasets cavité et œufs.")
    rows = metadata.loc[metadata["cap_id"].astype(str).isin(fish)].copy()
    rows["cap_id"] = rows["cap_id"].astype(str)
    rows["long_gonade"] = pd.to_numeric(rows["long_gonade"], errors="coerce")
    lengths = []
    for fish_id in fish:
        values = rows.loc[rows["cap_id"] == fish_id, "long_gonade"]
        if values.empty or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Longueur de gonade absente ou invalide pour le poisson {fish_id}.")
        if values.nunique() != 1:
            raise ValueError(f"Longueurs de gonade incohérentes pour le poisson {fish_id}.")
        lengths.append(values.iloc[0])
    try:
        strata = pd.qcut(pd.Series(lengths), q=3, labels=False)
    except ValueError as error:
        raise ValueError("Impossible de former trois quantiles de longueur de gonade.") from error
    if strata.nunique() != 3 or strata.value_counts().min() < 6:
        raise ValueError("Chaque quantile de longueur de gonade doit contenir au moins six poissons.")
    splitter = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    _, test_indices = next(splitter.split(fish, strata))
    return [fish[index] for index in test_indices]


def load_common_test_fish():
    """Charge la liste commune ; sa génération est explicite pour figer le test."""
    return {str(value) for value in json.loads(COMMON_TEST_PATH.read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    debut = Path(__file__).resolve().parents[1]
    metadata = pd.read_excel(debut / "utils/correspondance_echo_final.xlsx", dtype={"cap_id": str})

    def fish_in(directory):
        return {path.name.split("_")[2] for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    test_fish = compute_common_test_fish(
        metadata, fish_in(debut / "data/images/cavite"), fish_in(debut / "data/images/oeufs")
    )
    COMMON_TEST_PATH.write_text(json.dumps(test_fish, indent=2) + "\n")
    print(f"Test commun : {len(test_fish)} poissons, enregistré dans {COMMON_TEST_PATH}")


if __name__ == "__main__":
    main()
