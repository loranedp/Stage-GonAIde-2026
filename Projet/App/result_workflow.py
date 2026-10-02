"""Traitements partagés de prédiction, calcul et export des résultats."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

import calcul_volume_gonade
import config
import data_utils
import metadata_utils
import model_utils
from filename_utils import parse_filename
from utils import calcul_volume_fct


RESULT_CSV_COLUMNS = [
    "Id_poisson",
    "Volume_gonades",
    "Volume_cavite",
    "Volume_moyen_oeufs_mm3",
    "Volume_median_oeufs_mm3",
    "Fecondite_estimee",
    "Fecondite_IQR_borne_basse",
    "Fecondite_IQR_borne_haute",
]
LEGACY_RESULT_COLUMNS = {
    "Volume_gonades": "Volume_total_gonades_cm3",
    "Volume_cavite": "Volume_total_cavité_cm3",
    "Volume_moyen_oeufs_mm3": "Volume_moyen_oeuf_cm3",
    "Fecondite_estimee": "Fecondite_estimee",
}


@dataclass(frozen=True)
class UploadedImage:
    """Copie persistante d'un fichier fourni au widget Streamlit."""

    name: str
    data: bytes


@dataclass(frozen=True)
class WorkflowIssue:
    filename: str
    message: str
    fish_id: str | None = None


@dataclass
class PredictionBatch:
    predictions: dict[str, dict] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    corrected_keys: set[str] = field(default_factory=set)
    issues: list[WorkflowIssue] = field(default_factory=list)

    @property
    def failed_fish_ids(self) -> set[str]:
        return {issue.fish_id for issue in self.issues if issue.fish_id is not None}


def predict_uploads(
    uploads_by_type: Mapping[str, Sequence[UploadedImage]],
) -> PredictionBatch:
    """Prédit les fichiers fournis, en chargeant chaque famille de modèles au plus une fois."""

    batch = PredictionBatch()
    models_by_type: dict[str, list | None] = {"cavite": None, "oeufs": None}
    model_errors: dict[str, Exception] = {}

    for image_type in ("cavite", "oeufs"):
        for uploaded in uploads_by_type.get(image_type, ()):
            filename = uploaded.name
            try:
                parsed = parse_filename(filename)
            except ValueError as exc:
                batch.issues.append(WorkflowIssue(filename, str(exc)))
                continue

            prediction_key = f"{image_type}:{filename}"
            try:
                is_corrected = False
                corrected_data = data_utils.load_corrected_image_and_masks(
                    filename, image_type=image_type
                )
                if corrected_data is not None:
                    image = corrected_data["image"]
                    masks = corrected_data["masks"]
                    confidences = {}
                    instances = (
                        model_utils.split_egg_mask(masks["Oeuf"])
                        if image_type == "oeufs"
                        else None
                    )
                    is_corrected = True
                else:
                    image = Image.open(BytesIO(uploaded.data)).convert("RGB")
                    if image_type in model_errors:
                        raise model_errors[image_type]
                    if models_by_type[image_type] is None:
                        try:
                            models_by_type[image_type] = (
                                model_utils.load_egg_model()
                                if image_type == "oeufs"
                                else model_utils.load_models()
                            )
                        except (OSError, ValueError, KeyError, RuntimeError) as exc:
                            model_errors[image_type] = exc
                            raise
                    result = (
                        model_utils.predict_eggs(models_by_type[image_type], image)
                        if image_type == "oeufs"
                        else model_utils.predict(
                            models_by_type[image_type], image, parsed.id_image
                        )
                    )
                    masks = result["masks"]
                    confidences = result["confidences"]
                    instances = result.get("instances")
                status = (
                    data_utils.get_disk_status(filename, image_type=image_type)
                    or config.DEFAULT_STATUS
                )
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                data_utils.DataStoreError,
            ) as exc:
                batch.issues.append(
                    WorkflowIssue(filename, f"{filename} : {exc}", parsed.id_poisson)
                )
                continue

            prediction = {
                "filename": filename,
                "image_type": image_type,
                "image": image,
                "masks": masks,
                "confidences": confidences,
                "id_image": parsed.id_image,
                "heure": parsed.heure,
                "id_poisson": parsed.id_poisson,
            }
            if instances is not None:
                prediction["instances"] = instances
            if corrected_data is None and image_type == "oeufs":
                prediction["egg_areas_px"] = result["egg_areas_px"]
            batch.predictions[prediction_key] = prediction
            batch.statuses[prediction_key] = status
            if is_corrected:
                batch.corrected_keys.add(prediction_key)

    return batch


def calculate_fish_result(
    fish_id: str,
    predictions: Mapping[str, dict],
    statuses: Mapping[str, str],
) -> dict:
    """Calcule les résultats affichables/exportables d'un poisson."""

    fish_items = [
        (prediction_key, pred)
        for prediction_key, pred in predictions.items()
        if pred["id_poisson"] == fish_id
        and statuses.get(prediction_key, config.DEFAULT_STATUS) != "mauvaise"
    ]
    cavity_items = [item for item in fish_items if item[1]["image_type"] == "cavite"]
    egg_items = [item for item in fish_items if item[1]["image_type"] == "oeufs"]
    errors: list[str] = []

    images_ordered = []
    cavity_error = False
    fish_echelle_ocr = None
    for _, pred in cavity_items:
        filename = pred["filename"]
        meta = metadata_utils.lookup(pred["id_image"])
        if meta is None:
            errors.append(
                f"{filename} : aucune métadonnée trouvée pour id_image="
                f"'{pred['id_image']}' (position/long_gonade requis pour le volume)."
            )
            cavity_error = True
            continue
        echelle = meta["echelle"]
        if not np.isfinite(echelle):
            if fish_echelle_ocr is None:
                fish_echelle_ocr = calcul_volume_gonade.determine_fish_echelle(
                    [p["image"] for _, p in cavity_items]
                )
            echelle = fish_echelle_ocr
        surface = calcul_volume_gonade.surface_cm2(
            pred["masks"]["Gonade"], echelle, pred["image"].height
        )
        surface_cavite = calcul_volume_gonade.surface_cm2(
            pred["masks"]["Cavite"], echelle, pred["image"].height
        )
        images_ordered.append(
            {
                "id_image": pred["id_image"],
                "surface_cm2": surface,
                "surface_cavite_cm2": surface_cavite,
                "position": meta["position"],
                "long_gonade": meta["long_gonade"],
            }
        )

    rows = []
    anomalies = set()
    volume_total = None
    volume_total_cavite = None
    if not cavity_error and images_ordered:
        images_ordered.sort(key=lambda item: item["position"])
        anomalies = calcul_volume_gonade.detect_surface_gonade_anomalies(images_ordered)
        result = calcul_volume_gonade.compute_fish_volume(images_ordered)
        volume_total = result["volume_total_cm3"]
        result_cavite = calcul_volume_gonade.compute_fish_volume(
            images_ordered, surface_key="surface_cavite_cm2"
        )
        volume_total_cavite = result_cavite["volume_total_cm3"]
        rows = [
            {
                "Id_poisson": str(fish_id),
                "Id_image": item["id_image"],
                "Position": str(item["position"]),
                "Surface_gonades_cm2": round(item["surface_cm2"], 4),
                "Surface_cavité_cm2": round(item["surface_cavite_cm2"], 4),
                "Rayon_gonades_cm": round(
                    float(np.sqrt(item["surface_cm2"] / np.pi)), 4
                ),
            }
            for item in result["surfaces"]
        ]

    egg_inputs = []
    egg_echelle_ocr = None
    for _, pred in egg_items:
        meta = metadata_utils.lookup(pred["id_image"])
        echelle = meta["echelle"] if meta is not None else np.nan
        if not np.isfinite(echelle):
            if egg_echelle_ocr is None:
                egg_echelle_ocr = calcul_volume_gonade.determine_fish_echelle(
                    [p["image"] for _, p in egg_items]
                )
            echelle = egg_echelle_ocr
        instances = pred.get("instances")
        if instances is None:
            instances = model_utils.split_egg_mask(pred["masks"]["Oeuf"])
            pred["instances"] = instances
        egg_inputs.append(
            {
                "id_image": pred["id_image"],
                "instances": instances,
                "areas_px": pred.get("egg_areas_px"),
                "echelle": echelle,
                "image_height": pred["image"].height,
            }
        )

    egg_result = calcul_volume_fct.calculate_egg_volume_summary(egg_inputs)
    egg_rows = egg_result["images"]
    egg_median = egg_result["volume_median_oeufs_mm3"]
    fecundity = calculate_fecundity(volume_total, egg_median)
    fecundity_low = calculate_fecundity(volume_total, egg_result["volume_q3_oeufs_mm3"])
    fecundity_high = calculate_fecundity(volume_total, egg_result["volume_q1_oeufs_mm3"])

    return {
        "selected_fish": fish_id,
        "rows": rows,
        "anomalies": anomalies,
        "volume_total": volume_total,
        "volume_total_cavite": volume_total_cavite,
        "egg_rows": egg_rows,
        "egg_median": egg_median,
        "egg_q1": egg_result["volume_q1_oeufs_mm3"],
        "egg_q3": egg_result["volume_q3_oeufs_mm3"],
        "fecundity": fecundity,
        "fecundity_low": fecundity_low,
        "fecundity_high": fecundity_high,
        "has_cavity_images": bool(cavity_items),
        "has_egg_images": bool(egg_items),
        "has_eligible_images": bool(fish_items),
        "cavity_error": cavity_error,
        "cavity_ocr_scale": fish_echelle_ocr,
        "errors": errors,
    }


def _rounded_number(value, digits: int = 4):
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return round(number, digits) if np.isfinite(number) else None


def calculate_fecundity(volume_total_cm3, egg_volume_mm3):
    """Calcule la fécondité après conversion du volume de gonade en mm³."""

    if volume_total_cm3 is None or egg_volume_mm3 is None:
        return None
    gonad_volume = float(volume_total_cm3)
    egg_volume = float(egg_volume_mm3)
    if not np.isfinite(gonad_volume) or not np.isfinite(egg_volume) or egg_volume <= 0:
        return None
    return (gonad_volume * 1000) / egg_volume


def result_summary_row(result: Mapping) -> dict:
    """Convertit un résultat détaillé en une ligne synthétique pour le CSV."""

    def rounded_count(key):
        value = result.get(key)
        if value is None or pd.isna(value) or not np.isfinite(float(value)):
            return None
        return int(round(float(value)))

    return {
        "Id_poisson": str(result["selected_fish"]),
        "Volume_gonades": _rounded_number(result.get("volume_total")),
        "Volume_cavite": _rounded_number(result.get("volume_total_cavite")),
        "Volume_moyen_oeufs_mm3": _rounded_number(result.get("legacy_egg_mean")),
        "Volume_median_oeufs_mm3": _rounded_number(result.get("egg_median")),
        "Fecondite_estimee": rounded_count("fecundity"),
        "Fecondite_IQR_borne_basse": rounded_count("fecundity_low"),
        "Fecondite_IQR_borne_haute": rounded_count("fecundity_high"),
    }


def has_exportable_result(result: Mapping) -> bool:
    row = result_summary_row(result)
    return any(row[column] is not None for column in RESULT_CSV_COLUMNS[1:])


def _summary_dataframe(rows) -> pd.DataFrame:
    dataframe = pd.DataFrame(rows, columns=RESULT_CSV_COLUMNS)
    for column in ("Fecondite_estimee", "Fecondite_IQR_borne_basse", "Fecondite_IQR_borne_haute"):
        dataframe[column] = pd.array(dataframe[column], dtype="Int64")
    return dataframe


def migrate_results_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalise un CSV synthétique ou consolide l'ancien format par image."""

    if "Id_poisson" not in dataframe.columns:
        raise ValueError("Le CSV de résultats ne contient pas la colonne Id_poisson.")
    egg_volume_is_cm3 = False
    old_summary_columns = {
        "Id_poisson", "Volume_gonades", "Volume_cavite",
        "Volume_moyen_oeufs_mm3", "Fecondite_estimee",
    }
    if old_summary_columns.issubset(dataframe.columns):
        source_columns = {column: column for column in old_summary_columns if column != "Id_poisson"}
    elif {
        "Id_poisson",
        "Volume_gonades",
        "Volume_cavite",
        "Volume_moyen_oeufs",
        "Fecondite_estimee",
    }.issubset(dataframe.columns):
        source_columns = {
            "Volume_gonades": "Volume_gonades",
            "Volume_cavite": "Volume_cavite",
            "Volume_moyen_oeufs_mm3": "Volume_moyen_oeufs",
            "Fecondite_estimee": "Fecondite_estimee",
        }
        egg_volume_is_cm3 = True
    elif set(LEGACY_RESULT_COLUMNS.values()).issubset(dataframe.columns):
        source_columns = LEGACY_RESULT_COLUMNS
        egg_volume_is_cm3 = True
    else:
        raise ValueError("Le schéma du CSV de résultats est incompatible.")

    rows = []
    for fish_id, group in dataframe.groupby("Id_poisson", sort=False):
        result = {"selected_fish": str(fish_id)}
        result_keys = {
            "Volume_gonades": "volume_total",
            "Volume_cavite": "volume_total_cavite",
            "Volume_moyen_oeufs_mm3": "egg_mean",
            "Volume_median_oeufs_mm3": "egg_median",
            "Fecondite_estimee": "fecundity",
            "Fecondite_IQR_borne_basse": "fecundity_low",
            "Fecondite_IQR_borne_haute": "fecundity_high",
        }
        for target_column, result_key in result_keys.items():
            source_column = source_columns.get(target_column, target_column)
            if source_column not in group.columns:
                continue
            values = group[source_column].dropna()
            value = values.iloc[0] if not values.empty else None
            if (
                target_column == "Volume_moyen_oeufs_mm3"
                and value is not None
                and egg_volume_is_cm3
            ):
                value = float(value) * 1000
            result["legacy_egg_mean" if result_key == "egg_mean" else result_key] = value
        if has_exportable_result(result):
            rows.append(result_summary_row(result))

    return _summary_dataframe(rows)


def save_fish_results(results: Sequence[dict], csv_path: Path) -> int:
    """Crée ou remplace la ligne synthétique de chaque poisson calculé."""

    new_rows = [result_summary_row(result) for result in results if has_exportable_result(result)]
    if not new_rows:
        return 0

    new_rows_df = _summary_dataframe(new_rows)
    new_rows_df = new_rows_df.drop_duplicates(subset="Id_poisson", keep="last")
    fish_ids = set(new_rows_df["Id_poisson"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, encoding="utf-8", dtype={"Id_poisson": str})
        existing_df = migrate_results_dataframe(existing_df)
        existing_df = existing_df[~existing_df["Id_poisson"].isin(fish_ids)]
        combined_df = pd.DataFrame(
            [*existing_df.to_dict("records"), *new_rows_df.to_dict("records")],
            columns=RESULT_CSV_COLUMNS,
        )
    else:
        combined_df = new_rows_df
    combined_df = _summary_dataframe(combined_df.to_dict("records"))
    combined_df.to_csv(
        csv_path,
        columns=RESULT_CSV_COLUMNS,
        index=False,
        encoding="utf-8",
        float_format="%.4f",
    )
    return len(fish_ids)
