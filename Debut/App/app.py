"""Interface Streamlit : segmentation de gonades de poisson et calcul de volume."""

import logging

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import calcul_volume_gonade
import config
import data_utils
import metadata_utils
import model_utils
import train
from canvas_adapter import st_stable_canvas
from filename_utils import parse_filename
from mask_editor_utils import apply_pending_stroke, extract_stroke_mask

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Segmentation gonades", layout="wide")

BASE_FONT_SIZE_PX = 18
st.markdown(
    f"<style>html {{ font-size: {BASE_FONT_SIZE_PX}px; }}</style>",
    unsafe_allow_html=True,
)

CLASS_COLORS = {
    "Cavite": (10, 10, 200),
    "Gonade": (0, 150, 240),
    "Intestin": (170, 130, 250),
}
CLASS_ALPHAS = {
    "Cavite": 0.25,
    "Gonade": 0.55,
    "Intestin": 0.35,
}
EDIT_CLASS_ALPHAS = {
    "Cavite": 0.15,
    "Gonade": 0.45,
    "Intestin": 0.25,
}
IMAGE_DISPLAY_WIDTH = 500
EDIT_IMAGE_DISPLAY_WIDTH = 900
EDIT_ORIGIN_PREVIEW_WIDTH = 700
# Couleurs de tracé du pinceau de re-labellisation, absentes des images échographiques
# et des couleurs de classe, pour pouvoir isoler le tracé sur le canvas : vert pour
# "Ajouter", rouge pour "Effacer", afin de distinguer visuellement le mode actif.
STROKE_COLOR_ADD = "#39FF14"
STROKE_COLOR_ERASE = "#FF3131"


def init_state() -> None:
    st.session_state.setdefault("predictions", {})
    st.session_state.setdefault("statuses", {})
    st.session_state.setdefault("editing", {})
    st.session_state.setdefault("canvas_version", {})
    st.session_state.setdefault("corrected", set())
    st.session_state.setdefault("selected_fish", None)
    st.session_state.setdefault("uploader_version", 0)
    st.session_state.setdefault("volume_result", None)


CAPTION_FONT_SIZE_PX = 20
STATUS_FONT_SIZE_PX = 22
CAPTION_COLOR_CORRECTED = "green"


def render_captioned_image(image: Image.Image, caption: str, width: int, color: str | None = None) -> None:
    style = f"font-size:{CAPTION_FONT_SIZE_PX}px;"
    if color:
        style += f"color:{color};font-weight:600;"
    st.markdown(f'<p style="{style}">{caption}</p>', unsafe_allow_html=True)
    st.image(image, width=width)


def render_legend() -> None:
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
        f'background-color:rgb{CLASS_COLORS[class_name]};margin-right:6px;"></span>'
        f'{class_name}</span>'
        for class_name in config.CLASS_NAMES
    )
    st.markdown(f'<div style="margin-bottom:8px;">{swatches}</div>', unsafe_allow_html=True)


def crop_for_display(image: Image.Image) -> Image.Image:
    x, y, w, h = config.CROP_PARAMS
    return image.crop((x, y, x + w, y + h))


def overlay_masks(image: Image.Image, masks: dict, alphas: dict | None = None) -> Image.Image:
    alphas = alphas or CLASS_ALPHAS
    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    # Gonade dessinée en dernier (par-dessus) pour rester bien visible en cas de chevauchement.
    draw_order = ["Cavite", "Intestin", "Gonade"]
    for class_name in draw_order:
        color = CLASS_COLORS.get(class_name)
        if color is None:
            continue
        mask = masks.get(class_name)
        if mask is None or not mask.any():
            continue
        alpha = alphas.get(class_name, 0.4)
        color_arr = np.array(color, dtype=np.float32)
        for c in range(3):
            overlay[..., c] = np.where(
                mask, base[..., c] * (1 - alpha) + color_arr[c] * alpha, overlay[..., c]
            )
    return Image.fromarray(overlay.astype(np.uint8))


def compute_iou_per_class(pred_masks: dict, gt_masks: dict) -> dict:
    """IoU par classe entre masques prédits et masques de vérité terrain. Union nulle
    (les deux masques vides pour cette classe) -> IoU = 1.0 (accord parfait)."""
    ious = {}
    for class_name in config.CLASS_NAMES:
        pred_mask = pred_masks.get(class_name)
        gt_mask = gt_masks.get(class_name)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        ious[class_name] = float(intersection / union) if union > 0 else 1.0
    return ious


def _canvas_dims() -> tuple:
    """Taille (largeur, hauteur) du canvas d'édition, alignée sur l'aspect ratio du crop
    (config.CROP_PARAMS) à la largeur d'affichage habituelle (IMAGE_DISPLAY_WIDTH)."""
    _, _, w, h = config.CROP_PARAMS
    canvas_width = EDIT_IMAGE_DISPLAY_WIDTH
    canvas_height = round(EDIT_IMAGE_DISPLAY_WIDTH * h / w)
    return canvas_width, canvas_height


def render_mask_editor(filename: str, pred: dict) -> None:
    edit_state = st.session_state.editing[filename]
    edited_masks = edit_state["masks"]

    st.markdown("**Re-labellisation des masques**")
    edit_col1, edit_col2, edit_col3 = st.columns([2, 2, 2])
    with edit_col1:
        active_class = st.radio(
            "Classe", config.CLASS_NAMES, key=f"edit_class_{filename}", horizontal=True
        )
    with edit_col2:
        mode = st.radio(
            "Mode", ["Ajouter", "Effacer"], key=f"edit_mode_{filename}", horizontal=True
        )
    with edit_col3:
        stroke_width = st.slider(
            "Largeur du pinceau", min_value=2, max_value=40, value=12, key=f"edit_width_{filename}"
        )

    # Le composant est recréé au changement de classe ou de mode. Le tracé de
    # l'outil précédent, déjà renvoyé lors du mouse-up, est donc d'abord intégré.
    current_tool = (active_class, mode)
    previous_tool = edit_state.get("tool")
    if previous_tool is None:
        edit_state["tool"] = current_tool
    elif previous_tool != current_tool:
        try:
            apply_pending_stroke(edit_state, config.CROP_PARAMS)
        except (ValueError, KeyError) as exc:
            logger.exception(
                "Impossible d'appliquer le tracé avant changement d'outil",
                extra={"image_name": filename},
            )
            st.error(f"{filename} : tracé invalide ({exc}).")
        edit_state["tool"] = current_tool
        st.session_state.canvas_version[filename] = (
            st.session_state.canvas_version.get(filename, 0) + 1
        )

    canvas_width, canvas_height = _canvas_dims()
    preview = overlay_masks(pred["image"], edited_masks, alphas=EDIT_CLASS_ALPHAS)
    background = crop_for_display(preview).resize((canvas_width, canvas_height))

    stroke_color = STROKE_COLOR_ADD if mode == "Ajouter" else STROKE_COLOR_ERASE
    version = st.session_state.canvas_version.get(filename, 0)

    canvas_col, origin_col = st.columns([3, 1])
    with canvas_col:
        canvas_result = st_stable_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=background,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key=f"canvas_{filename}_{version}",
        )
    with origin_col:
        st.image(crop_for_display(pred["image"]), width=EDIT_ORIGIN_PREVIEW_WIDTH)

    # À chaque mouse-up, update_streamlit déclenche un rerun et fournit tous les
    # traits du canvas. On conserve le tracé courant afin qu'un clic direct sur
    # « Valider » ou un changement d'outil ne puisse pas le perdre.
    if canvas_result.image_data is not None:
        try:
            stroke_small = extract_stroke_mask(canvas_result.image_data, mode)
            edit_state["pending"] = (
                {"stroke": stroke_small.copy(), "class_name": active_class, "mode": mode}
                if stroke_small.any()
                else None
            )
        except ValueError as exc:
            logger.exception("Données de canvas invalides", extra={"image_name": filename})
            st.error(f"{filename} : le tracé reçu est invalide ({exc}).")

    action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
    with action_col1:
        if st.button("Appliquer le tracé", key=f"apply_stroke_{filename}"):
            try:
                if apply_pending_stroke(edit_state, config.CROP_PARAMS):
                    st.session_state.canvas_version[filename] = version + 1
                    st.rerun()
                else:
                    st.warning("Aucun tracé à appliquer.")
            except (ValueError, KeyError) as exc:
                logger.exception("Échec d'application du tracé", extra={"image_name": filename})
                st.error(f"{filename} : impossible d'appliquer le tracé ({exc}).")
    with action_col2:
        if st.button("Annuler", key=f"cancel_edit_{filename}"):
            del st.session_state.editing[filename]
            st.session_state.canvas_version.pop(filename, None)
            st.rerun()
    with action_col3:
        if st.button("Valider les modifications", key=f"save_edit_{filename}", type="primary"):
            try:
                apply_pending_stroke(edit_state, config.CROP_PARAMS)
                data_utils.add_image_to_corrected_set(filename, pred["image"], edited_masks)
                previous_status = st.session_state.statuses.get(filename, config.DEFAULT_STATUS)
                if previous_status == "bonne":
                    data_utils.remove_image_from_training_set(filename)
                elif previous_status == "mauvaise":
                    data_utils.remove_image_from_labellisation_set(filename)
            except (OSError, ValueError, KeyError, data_utils.DataStoreError) as exc:
                logger.exception(
                    "Échec de sauvegarde de la correction", extra={"image_name": filename}
                )
                st.error(
                    f"{filename} : la correction n'a pas pu être enregistrée ({exc}). "
                    "L'éditeur reste ouvert et votre travail est conservé."
                )
            else:
                pred["masks"] = {name: mask.copy() for name, mask in edited_masks.items()}
                st.session_state.statuses[filename] = config.DEFAULT_STATUS
                st.session_state.corrected.add(filename)
                del st.session_state.editing[filename]
                st.session_state.canvas_version.pop(filename, None)
                st.toast(f"{filename} modifiée !", icon="✅")
                st.rerun()


init_state()
tab_prediction, tab_entrainement = st.tabs(["Prédiction", "Entraînement"])

with tab_prediction:
    st.header("Données")
    upload_col, clear_col = st.columns([5, 1], vertical_alignment="center")
    with upload_col:
        uploaded_files = st.file_uploader(
            "Uploader une ou plusieurs images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_version}",
        )
    with clear_col:
        if st.button("Effacer", key="clear_uploaded_files"):
            st.session_state.uploader_version += 1
            st.session_state.predictions = {}
            st.session_state.statuses = {}
            st.session_state.editing = {}
            st.session_state.canvas_version = {}
            st.session_state.corrected = set()
            st.session_state.selected_fish = None
            st.session_state.volume_result = None
            st.rerun()

    if st.button("Prédiction"):
        if not uploaded_files:
            st.warning("Veuillez uploader au moins une image avant de lancer la prédiction.")
        else:
            models = model_utils.load_models()
            with st.spinner("Prédiction en cours..."):
                processed = 0
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    try:
                        parsed = parse_filename(filename)
                    except ValueError as exc:
                        st.error(str(exc))
                        continue

                    try:
                        corrected_data = data_utils.load_corrected_image_and_masks(filename)
                    except (OSError, ValueError, KeyError, data_utils.DataStoreError) as exc:
                        logger.exception(
                            "Échec de chargement d'une correction", extra={"image_name": filename}
                        )
                        st.error(
                            f"{filename} : correction enregistrée illisible ({exc}). "
                            "La prédiction n'a pas été lancée afin de ne pas masquer le problème."
                        )
                        continue
                    if corrected_data is not None:
                        image = corrected_data["image"]
                        masks = corrected_data["masks"]
                        confidences = {}
                        st.session_state.corrected.add(filename)
                    else:
                        image = Image.open(uploaded_file).convert("RGB")
                        st.session_state.corrected.discard(filename)
                        try:
                            result = model_utils.predict(models, image, parsed.id_image)
                        except ValueError as exc:
                            st.error(f"{filename} : {exc}")
                            continue
                        masks = result["masks"]
                        confidences = result["confidences"]

                    st.session_state.predictions[filename] = {
                        "image": image,
                        "masks": masks,
                        "confidences": confidences,
                        "id_image": parsed.id_image,
                        "heure": parsed.heure,
                        "id_poisson": parsed.id_poisson,
                    }
                    st.session_state.statuses.setdefault(
                        filename, data_utils.get_disk_status(filename) or config.DEFAULT_STATUS
                    )
                    processed += 1
            if processed:
                st.success(f"{processed} image(s) traitée(s).")

    st.header("Résultats")
    predictions = st.session_state.predictions
    selected_fish = None

    if not predictions:
        st.info("Lancez une prédiction pour afficher les résultats.")
    else:
        fish_ids = sorted({p["id_poisson"] for p in predictions.values()})
        if st.session_state.selected_fish not in fish_ids:
            st.session_state.selected_fish = fish_ids[0]
            st.session_state.volume_result = None

        def _select_fish(fish_id):
            st.session_state.selected_fish = fish_id
            st.session_state.volume_result = None

        st.write("Sélectionner un poisson :")
        for row_start in range(0, len(fish_ids), 8):
            row_ids = fish_ids[row_start : row_start + 8]
            for col, fish_id in zip(st.columns(len(row_ids)), row_ids):
                with col:
                    st.checkbox(
                        str(fish_id),
                        value=(fish_id == st.session_state.selected_fish),
                        key=f"fish_checkbox_{fish_id}",
                        on_change=_select_fish,
                        args=(fish_id,),
                    )

        selected_fish = st.session_state.selected_fish

        st.subheader("Informations poisson")
        if not predictions:
            st.info("Lancez une prédiction pour afficher les informations du poisson.")
        elif not selected_fish:
            st.warning("Veuillez sélectionner un poisson dans la liste ci-dessus.")
        else:
            fish_images = [p for p in predictions.values() if p["id_poisson"] == selected_fish]
            info = next(
                (m for m in (metadata_utils.lookup(p["id_image"]) for p in fish_images) if m is not None),
                None,
            )
            if info is None:
                st.warning(f"Aucune information trouvée pour le poisson {selected_fish}.")
            else:
                age_value = info["age_poisson"] if pd.notna(info["age_poisson"]) else "N/A"
                st.dataframe(
                    pd.DataFrame(
                        [{
                            "Id poisson": str(info["cap_id"]),
                            "Longueur gonade (en cm)": info["long_gonade"],
                            "Longueur poisson (en cm)": info["long_poisson"]/10,
                            "Poids poisson (en g)": info["poids_poisson"],
                            "Âge poisson": age_value,
                        }]
                    ),
                    use_container_width=True,
                )

        if selected_fish:
            fish_items = sorted(
                ((fn, p) for fn, p in predictions.items() if p["id_poisson"] == selected_fish),
                key=lambda item: item[1]["heure"],
            )
            render_legend()
            for filename, pred in fish_items:
                col_left, col_right = st.columns([4, 1], vertical_alignment="center")
                with col_left:
                    meta = metadata_utils.lookup(pred["id_image"])
                    op_value = (
                        meta["type_image"]
                        if meta is not None and pd.notna(meta["type_image"])
                        else "N/A"
                    )
                    st.subheader(f"Image : {filename} — Op : {op_value}")
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        render_captioned_image(
                            crop_for_display(pred["image"]),
                            "Image d'origine",
                            IMAGE_DISPLAY_WIDTH,
                        )
                    with img_col2:
                        overlay_img = overlay_masks(pred["image"], pred["masks"])
                        is_corrected = filename in st.session_state.corrected
                        caption = (
                            "Masques corrigés manuellement"
                            if is_corrected
                            else "Masques prédits"
                        )
                        render_captioned_image(
                            crop_for_display(overlay_img),
                            caption,
                            IMAGE_DISPLAY_WIDTH,
                            color=CAPTION_COLOR_CORRECTED if is_corrected else None,
                        )
                    #conf_cols = st.columns(len(config.CLASS_NAMES))
                    #for conf_col, class_name in zip(conf_cols, config.CLASS_NAMES):
                    #    confidence = pred["confidences"].get(class_name)
                    #    conf_col.metric(
                    #        f"Confiance ({class_name})",
                    #        f"{confidence:.1%}" if confidence is not None else "N/A",
                    #    )
                    gt_masks = data_utils.load_coco_ground_truth_masks(filename)
                    if gt_masks is not None:
                        ious = compute_iou_per_class(pred["masks"], gt_masks)
                        st.caption(
                            "Image présente dans data/COCO — IoU par classe (vs vérité terrain) :"
                        )
                        iou_cols = st.columns(len(config.CLASS_NAMES))
                        for iou_col, class_name in zip(iou_cols, config.CLASS_NAMES):
                            iou_col.metric(f"IoU ({class_name})", f"{ious[class_name]:.1%}")
                with col_right:
                    current_status = st.session_state.statuses.get(filename, config.DEFAULT_STATUS)
                    st.markdown(
                        f'<p style="font-size:{STATUS_FONT_SIZE_PX}px;">Statut : <strong>{current_status}</strong></p>',
                        unsafe_allow_html=True,
                    )
                    for status in config.STATUS_OPTIONS:
                        if st.button(status.capitalize(), key=f"status_{filename}_{status}"):
                            previous_status = st.session_state.statuses.get(
                                filename, config.DEFAULT_STATUS
                            )
                            try:
                                if status == "bonne":
                                    data_utils.add_image_to_training_set(
                                        filename, pred["image"], pred["masks"]
                                    )
                                elif previous_status == "bonne":
                                    data_utils.remove_image_from_training_set(filename)

                                if status == "mauvaise":
                                    data_utils.add_image_to_labellisation_set(
                                        filename, pred["image"], pred["masks"]
                                    )
                                elif previous_status == "mauvaise":
                                    data_utils.remove_image_from_labellisation_set(filename)
                            except (OSError, ValueError, KeyError, data_utils.DataStoreError) as exc:
                                logger.exception(
                                    "Échec du changement de statut",
                                    extra={"image_name": filename, "status": status},
                                )
                                st.error(f"{filename} : statut non enregistré ({exc}).")
                            else:
                                st.session_state.statuses[filename] = status
                                st.rerun()

                    if filename not in st.session_state.editing:
                        relabel_col, delete_col = st.columns(2)
                        with relabel_col:
                            if st.button("Re-labellisation", key=f"edit_toggle_{filename}"):
                                st.session_state.editing[filename] = {
                                    "masks": {c: m.copy() for c, m in pred["masks"].items()},
                                    "pending": None,
                                    "tool": None,
                                }
                                st.rerun()
                        with delete_col:
                            if filename in st.session_state.corrected:
                                if st.button(
                                    "Supprimer labellisation", key=f"delete_correction_{filename}"
                                ):
                                    try:
                                        # Ne retirer la seule correction valide qu'une fois
                                        # son remplacement par une prédiction confirmé.
                                        models = model_utils.load_models()
                                        result = model_utils.predict(
                                            models, pred["image"], pred["id_image"]
                                        )
                                        data_utils.remove_image_from_corrected_set(filename)
                                    except (
                                        OSError,
                                        ValueError,
                                        KeyError,
                                        data_utils.DataStoreError,
                                    ) as exc:
                                        logger.exception(
                                            "Échec de suppression de la correction",
                                            extra={"image_name": filename},
                                        )
                                        st.error(
                                            f"{filename} : labellisation non supprimée ({exc}). "
                                            "La correction existante a été conservée."
                                        )
                                    else:
                                        pred["masks"] = result["masks"]
                                        pred["confidences"] = result["confidences"]
                                        st.session_state.corrected.discard(filename)
                                        st.session_state.statuses[filename] = (
                                            data_utils.get_disk_status(filename)
                                            or config.DEFAULT_STATUS
                                        )
                                        st.toast(
                                            f"Labellisation de {filename} supprimée !", icon="🗑️"
                                        )
                                        st.rerun()

                if filename in st.session_state.editing:
                    render_mask_editor(filename, pred)
                st.divider()


    st.header("Calcul du volume")
    if st.button("Calcul du volume"):
        if not predictions:
            st.warning("Aucune prédiction disponible. Lancez d'abord une prédiction.")
        elif not selected_fish:
            st.warning("Veuillez sélectionner un poisson dans la liste ci-dessus.")
        else:
            fish_items = [
                (fn, p)
                for fn, p in predictions.items()
                if p["id_poisson"] == selected_fish
                and st.session_state.statuses.get(fn, config.DEFAULT_STATUS) != "mauvaise"
            ]
            if not fish_items:
                st.warning(
                    "Toutes les images de ce poisson sont marquées 'mauvaise', "
                    "aucune surface ne peut être calculée."
                )
            else:
                images_ordered = []
                error = False
                fish_echelle_ocr = None
                for filename, pred in fish_items:
                    meta = metadata_utils.lookup(pred["id_image"])
                    if meta is None:
                        st.error(
                            f"{filename} : aucune métadonnée trouvée pour id_image="
                            f"'{pred['id_image']}' (position/long_gonade requis pour le volume)."
                        )
                        error = True
                        continue
                    echelle = meta["echelle"]
                    if np.isnan(echelle):
                        if fish_echelle_ocr is None:
                            fish_echelle_ocr = calcul_volume_gonade.determine_fish_echelle(
                                [p["image"] for _, p in fish_items]
                            )
                            st.info(
                                f"Echelle : {fish_echelle_ocr} cm déterminée par OCR"
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

                if not error and images_ordered:
                    images_ordered.sort(key=lambda item: item["position"])
                    anomalies = calcul_volume_gonade.detect_surface_gonade_anomalies(
                        images_ordered
                    )
                    result = calcul_volume_gonade.compute_fish_volume(images_ordered)
                    volume_total = result["volume_total_cm3"]
                    result_cavite = calcul_volume_gonade.compute_fish_volume(
                        images_ordered, surface_key="surface_cavite_cm2"
                    )
                    volume_total_cavite = result_cavite["volume_total_cm3"]
                    rows = [
                        {
                            "Id_poisson": str(selected_fish),
                            "Id_image": item["id_image"],
                            "Position": str(item["position"]),
                            "Surface_gonades_cm2": round(item["surface_cm2"], 4),
                            "Surface_cavité_cm2": round(item["surface_cavite_cm2"], 4),
                            "Rayon_gonades_cm": round(float(np.sqrt(item["surface_cm2"] / np.pi)), 4),
                            "Volume_total_gonades_cm3": (
                                round(volume_total, 4) if volume_total is not None else None
                            ),
                            "Volume_total_cavité_cm3": (
                                round(volume_total_cavite, 4)
                                if volume_total_cavite is not None
                                else None
                            ),
                        }
                        for item in result["surfaces"]
                    ]
                    st.session_state.volume_result = {
                        "selected_fish": selected_fish,
                        "rows": rows,
                        "anomalies": anomalies,
                        "volume_total": volume_total,
                    }

    volume_result = st.session_state.volume_result
    if volume_result is not None:
        rows = volume_result["rows"]
        anomalies = volume_result["anomalies"]
        volume_total = volume_result["volume_total"]
        df_result = pd.DataFrame(rows)

        def _highlight_anomaly(row):
            style = (
                "color: red; font-weight: 600;" if row["Id_image"] in anomalies else ""
            )
            return [style] * len(row)

        st.dataframe(
            df_result.style.apply(_highlight_anomaly, axis=1),
            use_container_width=True,
        )
        if anomalies:
            st.caption(
                "Surface de gonade potentiellement incohérente (variation "
                "anormale entre échos consécutifs) pour l'image / les images : "
                + ", ".join(sorted(anomalies))
            )
        if volume_total is not None:
            st.metric("Volume total de la gonade", f"{volume_total:.4f} cm³")
        else:
            st.warning("Au moins 2 images sont nécessaires pour calculer un volume.")

        if st.button("Save"):
            config.RESULTATS_DIR.mkdir(parents=True, exist_ok=True)
            new_rows_df = pd.DataFrame(rows)
            if config.RESULTATS_CSV_PATH.exists():
                existing_df = pd.read_csv(
                    config.RESULTATS_CSV_PATH, encoding="utf-8", dtype={"Id_poisson": str}
                )
                existing_df = existing_df[
                    existing_df["Id_poisson"] != str(volume_result["selected_fish"])
                ]
                combined_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
            else:
                combined_df = new_rows_df
            combined_df.to_csv(config.RESULTATS_CSV_PATH, index=False, encoding="utf-8")
            st.success(f"Résultats enregistrés dans {config.RESULTATS_CSV_PATH}")

with tab_entrainement:
    st.header("Ré-entraînement du modèle")
    added_count = len(data_utils.list_added_filenames())
    st.write(
        f"{added_count} image(s) validée(s) 'bonne' actuellement ajoutées de façon "
        "permanente aux données d'entraînement (voir `data/train_added/`)."
    )

    epochs = st.number_input("Nombre d'epochs (par fold)", min_value=1, value=5, step=1)
    lr = st.number_input("Taux d'apprentissage", min_value=1e-6, value=1e-4, format="%.6f")

    if st.button("Ré-entraîner le modèle"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _progress_callback(fold_idx: int, num_folds: int, epoch: int, total_epochs: int, loss: float) -> None:
            overall = ((fold_idx - 1) * total_epochs + epoch) / (num_folds * total_epochs)
            progress_bar.progress(overall)
            status_text.text(f"Fold {fold_idx}/{num_folds} — epoch {epoch}/{total_epochs} — loss : {loss:.4f}")

        try:
            with st.spinner("Ré-entraînement en cours..."):
                result = train.train_model(
                    epochs=int(epochs), lr=float(lr), progress_callback=_progress_callback
                )
            model_utils.load_models.clear()
            losses_str = ", ".join(f"{loss:.4f}" for loss in result["final_losses"])
            st.success(
                f"Modèle ré-entraîné sur {result['num_samples']} images "
                f"(loss finale par fold : {losses_str})."
            )

            st.subheader("Validation par fold (généralisation sur poissons non vus par ce fold)")
            fold_rows = [
                {
                    "Fold": m["fold"],
                    "Images train": m["num_train"],
                    "Images val (held-out)": m["num_val"],
                    "Loss entraînement finale": round(m["train_loss"], 4),
                    "Score Dice validation": (
                        round(m["val_dice"], 4) if m["val_dice"] is not None else "N/A"
                    ),
                }
                for m in result["fold_metrics"]
            ]
            st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)
            if any(m["val_dice"] is None for m in result["fold_metrics"]):
                st.warning(
                    "Certains folds n'ont aucune image de validation disponible (trop peu "
                    "de poissons distincts par rapport au nombre de folds) : leur score de "
                    "généralisation est marqué 'N/A'. Ces folds restent entraînés "
                    "normalement, seule leur évaluation honnête n'est pas mesurable pour "
                    "l'instant."
                )
        except ValueError as exc:
            st.error(str(exc))
