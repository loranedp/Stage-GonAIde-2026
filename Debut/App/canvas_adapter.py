"""Adaptateur robuste pour ``streamlit-drawable-canvas``.

Le frontend 0.9.3 préfixe sans condition ``backgroundImageURL`` avec l'origine
du serveur. Cela casse aussi bien certains chemins déployés que les data URI.
Le contournement consiste à charger le PNG comme ``backgroundImage`` dans le
JSON Fabric du canvas principal : aucun chargement HTTP temporaire n'intervient.
"""

import base64
import copy
import io

import numpy as np
import streamlit_drawable_canvas as drawable_canvas
from PIL import Image


def image_to_data_uri(image: Image.Image, width: int, height: int) -> str:
    """Retourne une data URI PNG RGB aux dimensions exactes du canvas."""
    if width <= 0 or height <= 0:
        raise ValueError("Les dimensions du canvas doivent être strictement positives.")
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def drawing_with_background(
    background_image: Image.Image,
    width: int,
    height: int,
    initial_drawing: dict | None = None,
) -> dict:
    """Construit un état Fabric dont le fond est une image embarquée."""
    drawing = (
        copy.deepcopy(initial_drawing)
        if initial_drawing is not None
        else {"version": "4.4.0", "objects": []}
    )
    drawing.setdefault("objects", [])
    drawing["background"] = ""
    drawing["backgroundImage"] = {
        "type": "image",
        "version": "4.4.0",
        "originX": "left",
        "originY": "top",
        "left": 0,
        "top": 0,
        "width": width,
        "height": height,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "flipX": False,
        "flipY": False,
        "opacity": 1,
        "visible": True,
        "objectCaching": False,
        "src": image_to_data_uri(background_image, width, height),
        "crossOrigin": None,
        "filters": [],
    }
    return drawing


def st_stable_canvas(
    *,
    background_image: Image.Image,
    fill_color: str,
    stroke_width: int,
    stroke_color: str,
    update_streamlit: bool,
    height: int,
    width: int,
    drawing_mode: str,
    key: str,
    initial_drawing: dict | None = None,
    display_toolbar: bool = True,
):
    """Même contrat utile que ``st_canvas``, avec un fond embarqué et stable.

    L'accès aux deux symboles privés est volontairement concentré ici. La
    dépendance est figée à 0.9.3 ; une incompatibilité future sera ainsi
    détectée à un seul endroit et produira une exception explicite.
    """
    component_func = getattr(drawable_canvas, "_component_func", None)
    decode_result = getattr(drawable_canvas, "_data_url_to_image", None)
    result_type = getattr(drawable_canvas, "CanvasResult", None)
    if component_func is None or decode_result is None or result_type is None:
        raise RuntimeError(
            "Version incompatible de streamlit-drawable-canvas : "
            "l'adaptateur requiert l'API interne de la version 0.9.3."
        )

    drawing = drawing_with_background(
        background_image,
        width,
        height,
        initial_drawing,
    )
    component_value = component_func(
        fillColor=fill_color,
        strokeWidth=stroke_width,
        strokeColor=stroke_color,
        backgroundColor="",
        # Ne jamais utiliser le canal backgroundImageURL : le frontend lui
        # préfixe l'origine Streamlit, même lorsqu'il est déjà absolu.
        backgroundImageURL="",
        realtimeUpdateStreamlit=update_streamlit and drawing_mode != "polygon",
        canvasHeight=height,
        canvasWidth=width,
        drawingMode=drawing_mode,
        initialDrawing=drawing,
        displayToolbar=display_toolbar,
        displayRadius=3,
        key=key,
        default=None,
    )
    if component_value is None:
        return result_type()
    return result_type(
        np.asarray(decode_result(component_value["data"])),
        component_value["raw"],
    )
