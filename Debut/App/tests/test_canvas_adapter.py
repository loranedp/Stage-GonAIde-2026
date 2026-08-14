import base64
import io
import unittest
from unittest import mock

from PIL import Image, ImageStat

import canvas_adapter


class CanvasAdapterTests(unittest.TestCase):
    def test_data_uri_contains_resized_non_white_image(self):
        source = Image.new("RGB", (8, 6), (12, 34, 56))
        uri = canvas_adapter.image_to_data_uri(source, 20, 10)

        prefix, payload = uri.split(",", 1)
        decoded = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(decoded.size, (20, 10))
        self.assertEqual(
            tuple(round(value) for value in ImageStat.Stat(decoded).mean), (12, 34, 56)
        )

    def test_canvas_receives_embedded_fabric_background(self):
        captured = {}

        def fake_component(**kwargs):
            captured.update(kwargs)
            return None

        with mock.patch.object(canvas_adapter.drawable_canvas, "_component_func", fake_component):
            result = canvas_adapter.st_stable_canvas(
                background_image=Image.new("RGB", (4, 4), "black"),
                fill_color="transparent",
                stroke_width=5,
                stroke_color="#39FF14",
                update_streamlit=True,
                height=40,
                width=50,
                drawing_mode="freedraw",
                key="test-canvas",
            )

        self.assertIsNone(result.image_data)
        self.assertEqual(captured["backgroundImageURL"], "")
        fabric_background = captured["initialDrawing"]["backgroundImage"]
        self.assertEqual(fabric_background["type"], "image")
        self.assertEqual((fabric_background["width"], fabric_background["height"]), (50, 40))
        self.assertTrue(fabric_background["src"].startswith("data:image/png;base64,"))
        self.assertEqual(captured["initialDrawing"]["objects"], [])

    def test_drawing_builder_does_not_mutate_existing_state(self):
        initial = {"version": "4.4.0", "objects": [{"type": "path"}]}
        drawing = canvas_adapter.drawing_with_background(
            Image.new("RGB", (4, 4), "black"), 8, 6, initial
        )

        self.assertNotIn("backgroundImage", initial)
        self.assertEqual(drawing["objects"], initial["objects"])
        self.assertEqual(
            (drawing["backgroundImage"]["width"], drawing["backgroundImage"]["height"]),
            (8, 6),
        )


if __name__ == "__main__":
    unittest.main()
