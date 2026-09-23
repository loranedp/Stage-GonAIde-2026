"""Validation helpers for cross-validation result files."""

from collections import Counter
from pathlib import Path


def _fold_number(key):
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.startswith("fold_"):
        suffix = key.removeprefix("fold_")
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"Clé de fold invalide : {key!r}")


def normalize_and_validate_cv_results(results, expected_folds=5, expected_images=None):
    """Return folds named ``fold_1`` ... and reject malformed CV results.

    Both result formats used by this repository are supported: a list of
    per-image YOLO dictionaries, or a U-Net dictionary containing ``name``.
    ``fold_times`` is preserved when present.
    """
    normalized = {}
    for key, fold_results in results.items():
        if key == "fold_times":
            continue
        fold_number = _fold_number(key)
        fold_name = f"fold_{fold_number}"
        if fold_name in normalized:
            raise ValueError(f"Fold dupliqué après normalisation : {fold_name}")
        normalized[fold_name] = fold_results

    expected_names = {f"fold_{index}" for index in range(1, expected_folds + 1)}
    actual_names = set(normalized)
    if actual_names != expected_names:
        raise ValueError(
            f"Folds invalides : attendu {sorted(expected_names)}, obtenu {sorted(actual_names)}"
        )

    image_names = []
    for fold_name in sorted(normalized, key=_fold_number):
        fold_results = normalized[fold_name]
        if isinstance(fold_results, dict):
            fold_images = fold_results.get("name")
            if fold_images is None:
                raise ValueError(f"Champ 'name' absent de {fold_name}")
        else:
            fold_images = [item.get("image") for item in fold_results]
            if any(name is None for name in fold_images):
                raise ValueError(f"Champ 'image' absent d'un résultat de {fold_name}")
        image_names.extend(Path(name).name for name in fold_images)

    duplicates = sorted(name for name, count in Counter(image_names).items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Images présentes dans plusieurs folds : {preview}")

    if expected_images is not None:
        expected = {Path(name).name for name in expected_images}
        actual = set(image_names)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise ValueError(
                "Partition de validation incorrecte : "
                f"{len(missing)} image(s) manquante(s), "
                f"{len(unexpected)} image(s) inattendue(s)"
            )

    if "fold_times" in results:
        time_folds = {_fold_number(key) for key in results["fold_times"]}
        if time_folds != set(range(1, expected_folds + 1)):
            raise ValueError(
                f"Temps de folds invalides : attendu {expected_folds}, obtenu {len(time_folds)}"
            )
        normalized["fold_times"] = results["fold_times"]
    return normalized


def repair_repeated_leading_folds(results, expected_folds=5):
    """Repair the known notebook rerun artifact, otherwise leave results intact.

    The historical YOLO pickle contains two stale copies of fold 1 followed by
    one complete five-fold run.  Recovery is deliberately narrow: the extra
    leading folds must have exactly the same image sequence as the first fold
    of the retained run.
    """
    fold_items = [(key, value) for key, value in results.items() if key != "fold_times"]
    fold_items.sort(key=lambda item: _fold_number(item[0]))
    surplus = len(fold_items) - expected_folds
    if surplus <= 0:
        return results, False

    retained = fold_items[surplus:]
    first_retained_images = [Path(item["image"]).name for item in retained[0][1]]
    for _, stale_fold in fold_items[:surplus]:
        stale_images = [Path(item["image"]).name for item in stale_fold]
        if stale_images != first_retained_images:
            return results, False

    repaired = {
        f"fold_{index}": fold_results
        for index, (_, fold_results) in enumerate(retained, start=1)
    }
    if "fold_times" in results:
        repaired["fold_times"] = results["fold_times"]
    return repaired, True
