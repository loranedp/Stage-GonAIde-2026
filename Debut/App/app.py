"""Interface Streamlit : segmentation de gonades de poisson et calcul de volume."""

import logging

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
import data_utils
import metadata_utils
from metrics_utils import compute_iou_per_class
import model_utils
import result_workflow
from utils import calcul_volume_fct
from canvas_adapter import st_stable_canvas
from mask_editor_utils import apply_pending_stroke, extract_stroke_mask
from rendering_utils import draw_instance_contours

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
    "Oeuf": (245, 170, 35),
}
CLASS_ALPHAS = {
    "Cavite": 0.25,
    "Gonade": 0.55,
    "Intestin": 0.35,
    "Oeuf": 0.50,
}
EDIT_CLASS_ALPHAS = {
    "Cavite": 0.15,
    "Gonade": 0.45,
    "Intestin": 0.25,
    "Oeuf": 0.40,
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
    st.session_state.setdefault("uploaded_batches", {"cavite": [], "oeufs": []})
    st.session_state.setdefault("volume_result", None)


def snapshot_uploaded_files(uploaded_files) -> list[result_workflow.UploadedImage]:
    """Conserve les uploads même lorsque leur widget n'est plus affiché."""
    return [
        result_workflow.UploadedImage(uploaded_file.name, uploaded_file.getvalue())
        for uploaded_file in uploaded_files
    ]


def apply_prediction_batch(batch: result_workflow.PredictionBatch) -> None:
    """Intègre un résultat de prédiction dans la session Streamlit."""
    st.session_state.predictions.update(batch.predictions)
    for prediction_key, status in batch.statuses.items():
        st.session_state.statuses.setdefault(prediction_key, status)
        if prediction_key in batch.corrected_keys:
            st.session_state.corrected.add(prediction_key)
        else:
            st.session_state.corrected.discard(prediction_key)
    st.session_state.volume_result = None


def render_prediction_issues(batch: result_workflow.PredictionBatch) -> None:
    for issue in batch.issues:
        st.error(issue.message)


CAPTION_FONT_SIZE_PX = 22
STATUS_FONT_SIZE_PX = 22
CAPTION_COLOR_CORRECTED = "green"


def render_captioned_image(image: Image.Image, caption: str, width: int, color: str | None = None) -> None:
    style = f"font-size:{CAPTION_FONT_SIZE_PX}px;"
    if color:
        style += f"color:{color};font-weight:600;"
    st.markdown(f'<p style="{style}">{caption}</p>', unsafe_allow_html=True)
    st.image(image, width=width)


def render_legend() -> None:
    render_class_legend(config.CLASS_NAMES)


def render_class_legend(class_names: list[str]) -> None:
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
        f'background-color:rgb{CLASS_COLORS[class_name]};margin-right:6px;"></span>'
        f'{class_name}</span>'
        for class_name in class_names
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
    draw_order = ["Cavite", "Intestin", "Gonade", "Oeuf"]
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


def _canvas_dims() -> tuple:
    """Taille (largeur, hauteur) du canvas d'édition, alignée sur l'aspect ratio du crop
    (config.CROP_PARAMS) à la largeur d'affichage habituelle (IMAGE_DISPLAY_WIDTH)."""
    _, _, w, h = config.CROP_PARAMS
    canvas_width = EDIT_IMAGE_DISPLAY_WIDTH
    canvas_height = round(EDIT_IMAGE_DISPLAY_WIDTH * h / w)
    return canvas_width, canvas_height


def render_mask_editor(prediction_key: str, pred: dict) -> None:
    filename = pred["filename"]
    image_type = pred["image_type"]
    class_names = config.EGG_CLASS_NAMES if image_type == "oeufs" else config.CLASS_NAMES
    edit_state = st.session_state.editing[prediction_key]
    edited_masks = edit_state["masks"]

    st.markdown("**Re-labellisation des masques**")
    edit_col1, edit_col2, edit_col3 = st.columns([2, 2, 2])
    with edit_col1:
        active_class = st.radio(
            "Classe", class_names, key=f"edit_class_{prediction_key}", horizontal=True
        )
    with edit_col2:
        mode = st.radio(
            "Mode", ["Ajouter", "Effacer"], key=f"edit_mode_{prediction_key}", horizontal=True
        )
    with edit_col3:
        stroke_width = st.slider(
            "Largeur du pinceau", min_value=2, max_value=40, value=12, key=f"edit_width_{prediction_key}"
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
        st.session_state.canvas_version[prediction_key] = (
            st.session_state.canvas_version.get(prediction_key, 0) + 1
        )

    canvas_width, canvas_height = _canvas_dims()
    preview = overlay_masks(pred["image"], edited_masks, alphas=EDIT_CLASS_ALPHAS)
    background = crop_for_display(preview).resize((canvas_width, canvas_height))

    stroke_color = STROKE_COLOR_ADD if mode == "Ajouter" else STROKE_COLOR_ERASE
    version = st.session_state.canvas_version.get(prediction_key, 0)

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
            key=f"canvas_{prediction_key}_{version}",
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
        if st.button("Appliquer le tracé", key=f"apply_stroke_{prediction_key}"):
            try:
                if apply_pending_stroke(edit_state, config.CROP_PARAMS):
                    st.session_state.canvas_version[prediction_key] = version + 1
                    st.rerun()
                else:
                    st.warning("Aucun tracé à appliquer.")
            except (ValueError, KeyError) as exc:
                logger.exception("Échec d'application du tracé", extra={"image_name": filename})
                st.error(f"{filename} : impossible d'appliquer le tracé ({exc}).")
    with action_col2:
        if st.button("Annuler", key=f"cancel_edit_{prediction_key}"):
            del st.session_state.editing[prediction_key]
            st.session_state.canvas_version.pop(prediction_key, None)
            st.rerun()
    with action_col3:
        if st.button(
            "Valider les modifications", key=f"save_edit_{prediction_key}", type="primary"
        ):
            try:
                apply_pending_stroke(edit_state, config.CROP_PARAMS)
                data_utils.add_image_to_corrected_set(
                    filename, pred["image"], edited_masks, image_type=image_type
                )
                previous_status = st.session_state.statuses.get(
                    prediction_key, config.DEFAULT_STATUS
                )
                if previous_status == "bonne":
                    data_utils.remove_image_from_training_set(filename, image_type=image_type)
                elif previous_status == "mauvaise":
                    data_utils.remove_image_from_labellisation_set(filename, image_type=image_type)
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
                if image_type == "oeufs":
                    pred["instances"] = model_utils.split_egg_mask(pred["masks"]["Oeuf"])
                st.session_state.statuses[prediction_key] = config.DEFAULT_STATUS
                st.session_state.corrected.add(prediction_key)
                del st.session_state.editing[prediction_key]
                st.session_state.canvas_version.pop(prediction_key, None)
                st.session_state.volume_result = None
                st.toast(f"{filename} modifiée !", icon="✅")
                st.rerun()


init_state()
tab_prediction = st.container()

with tab_prediction:
    st.header("Données")
    upload_type = st.radio(
        "Type des images à ajouter",
        ["cavite", "oeufs"],
        format_func=lambda value: "Cavité" if value == "cavite" else "Œufs",
        horizontal=True,
        key="upload_image_type",
    )
    upload_col, clear_col = st.columns([5, 1], vertical_alignment="center")
    with upload_col:
        uploaded_files = st.file_uploader(
            "Uploader une ou plusieurs images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"file_uploader_{upload_type}_{st.session_state.uploader_version}",
        )
    with clear_col:
        if st.button("Effacer", key="clear_uploaded_files"):
            st.session_state.uploader_version += 1
            st.session_state.uploaded_batches = {"cavite": [], "oeufs": []}
            st.session_state.predictions = {}
            st.session_state.statuses = {}
            st.session_state.editing = {}
            st.session_state.canvas_version = {}
            st.session_state.corrected = set()
            st.session_state.selected_fish = None
            st.session_state.volume_result = None
            st.rerun()

    if uploaded_files:
        st.session_state.uploaded_batches[upload_type] = snapshot_uploaded_files(
            uploaded_files
        )
    cavity_upload_count = len(st.session_state.uploaded_batches["cavite"])
    egg_upload_count = len(st.session_state.uploaded_batches["oeufs"])
    if cavity_upload_count or egg_upload_count:
        st.caption(
            f"Fichiers conservés — Cavité : {cavity_upload_count} · "
            f"Œufs : {egg_upload_count}"
        )

    if st.button("Sauvegarde des résultats", key="automatic_results_save"):
        uploads_by_type = st.session_state.uploaded_batches
        if not any(uploads_by_type.values()):
            st.warning("Veuillez uploader au moins une image avant la sauvegarde.")
        else:
            with st.spinner("Prédiction, calcul et sauvegarde en cours..."):
                batch = result_workflow.predict_uploads(uploads_by_type)
                apply_prediction_batch(batch)

                successful_fish_ids = {
                    pred["id_poisson"] for pred in batch.predictions.values()
                }
                candidate_fish_ids = sorted(
                    successful_fish_ids - batch.failed_fish_ids
                )
                calculated_results = []
                for fish_id in candidate_fish_ids:
                    try:
                        result = result_workflow.calculate_fish_result(
                            fish_id,
                            batch.predictions,
                            st.session_state.statuses,
                        )
                    except (OSError, ValueError, KeyError, RuntimeError) as exc:
                        logger.exception(
                            "Échec du calcul automatique",
                            extra={"fish_id": fish_id},
                        )
                        continue
                    if result_workflow.has_exportable_result(result):
                        calculated_results.append(result)

                try:
                    saved_count = result_workflow.save_fish_results(
                        calculated_results, config.RESULTATS_CSV_PATH
                    )
                except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                    logger.exception("Échec de sauvegarde automatique des résultats")
                    saved_count = 0
                    save_error = str(exc)
                else:
                    save_error = None

            all_known_fish_ids = successful_fish_ids | batch.failed_fish_ids
            skipped_count = len(all_known_fish_ids) - saved_count
            if save_error is not None:
                st.error(f"Les résultats n'ont pas pu être sauvegardés ({save_error}).")
            elif saved_count:
                st.success(
                    f"Résultats sauvegardés pour {saved_count} poisson(s) dans "
                    f"{config.RESULTATS_CSV_PATH}."
                )
            else:
                st.error("Aucun poisson n'a pu être sauvegardé.")
            if save_error is None and (skipped_count or batch.issues):
                st.warning(
                    f"Sauvegarde partielle : {skipped_count} poisson(s) et "
                    f"{len([issue for issue in batch.issues if issue.fish_id is None])} "
                    "fichier(s) non rattaché(s) ont été ignorés."
                )

    st.header("Résultats par poisson")
    if st.button("Prédiction", key="manual_prediction"):
        current_uploads = st.session_state.uploaded_batches[upload_type]
        if not current_uploads:
            st.warning("Veuillez uploader au moins une image avant de lancer la prédiction.")
        else:
            with st.spinner("Prédiction en cours..."):
                batch = result_workflow.predict_uploads({upload_type: current_uploads})
                apply_prediction_batch(batch)
            render_prediction_issues(batch)
            if batch.predictions:
                st.success(f"{len(batch.predictions)} image(s) traitée(s).")

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
                ((key, p) for key, p in predictions.items() if p["id_poisson"] == selected_fish),
                key=lambda item: (item[1]["image_type"] == "oeufs", item[1]["heure"]),
            )
            visible_classes = list(config.CLASS_NAMES)
            if any(pred["image_type"] == "oeufs" for _, pred in fish_items):
                visible_classes.extend(config.EGG_CLASS_NAMES)
            render_class_legend(visible_classes)
            for prediction_key, pred in fish_items:
                filename = pred["filename"]
                image_type = pred["image_type"]
                instances = None
                if image_type == "oeufs":
                    instances = pred.get("instances")
                    if instances is None:
                        instances = model_utils.split_egg_mask(pred["masks"]["Oeuf"])
                        pred["instances"] = instances
                col_left, col_right = st.columns([4, 1], vertical_alignment="center")
                with col_left:
                    meta = metadata_utils.lookup(pred["id_image"])
                    op_value = (
                        meta["type_image"]
                        if meta is not None and pd.notna(meta["type_image"])
                        else "N/A"
                    )
                    descriptor = "Œufs" if image_type == "oeufs" else f"Op : {op_value}"
                    st.subheader(f"Image : {filename} — {descriptor}")
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        render_captioned_image(
                            crop_for_display(pred["image"]),
                            "Image d'origine",
                            IMAGE_DISPLAY_WIDTH,
                        )
                    with img_col2:
                        overlay_img = overlay_masks(pred["image"], pred["masks"])
                        if instances is not None:
                            overlay_img = draw_instance_contours(overlay_img, instances)
                        is_corrected = prediction_key in st.session_state.corrected
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
                    if instances is not None:
                        distinct_egg_count = calcul_volume_fct.count_egg_instances(instances)
                        st.caption(
                            f"Nombre d'œufs distincts retrouvés : {distinct_egg_count}"
                        )
                    #conf_cols = st.columns(len(config.CLASS_NAMES))
                    #for conf_col, class_name in zip(conf_cols, config.CLASS_NAMES):
                    #    confidence = pred["confidences"].get(class_name)
                    #    conf_col.metric(
                    #        f"Confiance ({class_name})",
                    #        f"{confidence:.1%}" if confidence is not None else "N/A",
                    #    )
                    class_names = (
                        config.EGG_CLASS_NAMES
                        if image_type == "oeufs"
                        else config.CLASS_NAMES
                    )
                    gt_masks = data_utils.load_coco_ground_truth_masks(
                        filename, image_type=image_type
                    )
                    if gt_masks is not None:
                        ious = compute_iou_per_class(
                            pred["masks"], gt_masks, class_names
                        )
                        st.caption(
                            "Image présente dans data/labels/COCO/cavite — IoU par classe (vs vérité terrain) :"
                        )
                        iou_cols = st.columns(len(class_names))
                        for iou_col, class_name in zip(iou_cols, class_names):
                            iou_col.metric(f"IoU ({class_name})", f"{ious[class_name]:.1%}")
                with col_right:
                    current_status = st.session_state.statuses.get(
                        prediction_key, config.DEFAULT_STATUS
                    )
                    st.markdown(
                        f'<p style="font-size:{STATUS_FONT_SIZE_PX}px;">Statut : <strong>{current_status}</strong></p>',
                        unsafe_allow_html=True,
                    )
                    for status in config.STATUS_OPTIONS:
                        if st.button(
                            status.capitalize(), key=f"status_{prediction_key}_{status}"
                        ):
                            previous_status = st.session_state.statuses.get(
                                prediction_key, config.DEFAULT_STATUS
                            )
                            try:
                                if status == "bonne":
                                    data_utils.add_image_to_training_set(
                                        filename,
                                        pred["image"],
                                        pred["masks"],
                                        image_type=image_type,
                                    )
                                elif previous_status == "bonne":
                                    data_utils.remove_image_from_training_set(
                                        filename, image_type=image_type
                                    )

                                if status == "mauvaise":
                                    data_utils.add_image_to_labellisation_set(
                                        filename,
                                        pred["image"],
                                        pred["masks"],
                                        image_type=image_type,
                                    )
                                elif previous_status == "mauvaise":
                                    data_utils.remove_image_from_labellisation_set(
                                        filename, image_type=image_type
                                    )
                            except (OSError, ValueError, KeyError, data_utils.DataStoreError) as exc:
                                logger.exception(
                                    "Échec du changement de statut",
                                    extra={"image_name": filename, "status": status},
                                )
                                st.error(f"{filename} : statut non enregistré ({exc}).")
                            else:
                                st.session_state.statuses[prediction_key] = status
                                st.session_state.volume_result = None
                                st.rerun()

                    if prediction_key not in st.session_state.editing:
                        relabel_col, delete_col = st.columns(2)
                        with relabel_col:
                            if st.button(
                                "Re-labellisation", key=f"edit_toggle_{prediction_key}"
                            ):
                                st.session_state.editing[prediction_key] = {
                                    "masks": {c: m.copy() for c, m in pred["masks"].items()},
                                    "pending": None,
                                    "tool": None,
                                }
                                st.rerun()
                        with delete_col:
                            if prediction_key in st.session_state.corrected:
                                if st.button(
                                    "Supprimer labellisation",
                                    key=f"delete_correction_{prediction_key}",
                                ):
                                    try:
                                        # Ne retirer la seule correction valide qu'une fois
                                        # son remplacement par une prédiction confirmé.
                                        models = (
                                            model_utils.load_egg_models()
                                            if image_type == "oeufs"
                                            else model_utils.load_models()
                                        )
                                        result = (
                                            model_utils.predict_eggs(models, pred["image"])
                                            if image_type == "oeufs"
                                            else model_utils.predict(
                                                models, pred["image"], pred["id_image"]
                                            )
                                        )
                                        data_utils.remove_image_from_corrected_set(
                                            filename, image_type=image_type
                                        )
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
                                        if image_type == "oeufs":
                                            pred["instances"] = result["instances"]
                                        st.session_state.corrected.discard(prediction_key)
                                        st.session_state.statuses[prediction_key] = (
                                            data_utils.get_disk_status(
                                                filename, image_type=image_type
                                            )
                                            or config.DEFAULT_STATUS
                                        )
                                        st.session_state.volume_result = None
                                        st.toast(
                                            f"Labellisation de {filename} supprimée !", icon="🗑️"
                                        )
                                        st.rerun()

                if prediction_key in st.session_state.editing:
                    render_mask_editor(prediction_key, pred)
                st.divider()


    st.header("Calculs")
    if st.button("Calculer les volumes et la fécondité"):
        if not predictions:
            st.warning("Aucune prédiction disponible. Lancez d'abord une prédiction.")
        elif not selected_fish:
            st.warning("Veuillez sélectionner un poisson dans la liste ci-dessus.")
        else:
            result = result_workflow.calculate_fish_result(
                selected_fish, predictions, st.session_state.statuses
            )
            if not result["has_eligible_images"]:
                st.warning(
                    "Toutes les images de ce poisson sont marquées 'mauvaise', "
                    "aucune surface ne peut être calculée."
                )
            else:
                for error in result["errors"]:
                    st.error(error)
                if result["cavity_ocr_scale"] is not None:
                    st.info(
                        f"Echelle : {result['cavity_ocr_scale']} cm déterminée par OCR"
                    )
                st.session_state.volume_result = result

    volume_result = st.session_state.volume_result
    if volume_result is not None:
        rows = volume_result["rows"]
        anomalies = volume_result["anomalies"]
        volume_total = volume_result["volume_total"]
        volume_total_cavite = volume_result["volume_total_cavite"]
        egg_rows = volume_result["egg_rows"]
        egg_mean = volume_result["egg_mean"]
        fecundity = volume_result["fecundity"]

        st.subheader("Volume des gonades et de la cavité")
        if rows:
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
                gonad_volume_col, cavity_volume_col = st.columns(2)
                gonad_volume_col.metric(
                    "Volume total de la gonade", f"{volume_total:.4f} cm³"
                )
                cavity_volume_col.metric(
                    "Volume total de la cavité",
                    (
                        f"{volume_total_cavite:.4f} cm³"
                        if volume_total_cavite is not None
                        else "N/A"
                    ),
                )
            else:
                st.warning("Au moins 2 images cavité sont nécessaires pour calculer un volume.")
        elif volume_result["cavity_error"]:
            st.error("Le volume de la gonade n'a pas pu être calculé à cause des métadonnées.")
        else:
            st.info("Aucune image cavité bonne/ok n'est disponible pour ce poisson.")

        st.subheader("Volume moyen des œufs")
        if egg_rows:
            egg_display_rows = [
                {
                    "Id_image": row["id_image"],
                    "Nombre d'œufs utilisés pour la moyenne": row[
                        "nombre_oeufs_utilises_pour_moyenne"
                    ],
                    "Surface_moyenne_oeufs_mm2": round(
                        row["surface_moyenne_oeufs_mm2"], 4
                    ),
                    "Volume_moyen_oeufs_mm3": round(row["volume_moyen_oeufs_mm3"], 4),
                }
                for row in egg_rows
            ]
            st.dataframe(pd.DataFrame(egg_display_rows), use_container_width=True)
            if egg_mean is not None and egg_mean > 0:
                st.metric("Volume moyen d'un œuf", f"{egg_mean:.4f} mm³")
            else:
                st.warning("Aucune instance d'œuf exploitable n'a été détectée.")
        else:
            st.info("Aucune image d'œufs bonne/ok n'est disponible pour ce poisson.")

        st.subheader("Fécondité")
        if fecundity is not None:
            st.metric("Fécondité estimée", f"{int(round(fecundity))} œufs")
        else:
            st.info(
                "Le volume de la gonade et un volume moyen d'œuf strictement positif "
                "sont nécessaires pour estimer la fécondité."
            )

        if result_workflow.has_exportable_result(volume_result) and st.button("Save"):
            try:
                result_workflow.save_fish_results(
                    [volume_result], config.RESULTATS_CSV_PATH
                )
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                logger.exception("Échec de sauvegarde manuelle des résultats")
                st.error(f"Les résultats n'ont pas pu être enregistrés ({exc}).")
            else:
                st.success(f"Résultats enregistrés dans {config.RESULTATS_CSV_PATH}")
        elif not result_workflow.has_exportable_result(volume_result):
            st.caption(
                "Aucun résultat calculé n'est disponible pour la sauvegarde CSV."
            )
