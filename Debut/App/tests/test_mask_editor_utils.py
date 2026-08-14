import unittest

import numpy as np

from mask_editor_utils import apply_pending_stroke, apply_stroke_to_masks, extract_stroke_mask


class MaskEditorUtilsTests(unittest.TestCase):
    def test_extracts_only_visible_stroke_color(self):
        rgba = np.zeros((3, 3, 4), dtype=np.uint8)
        rgba[0, 0] = (57, 255, 20, 255)
        rgba[0, 1] = (57, 255, 20, 0)
        rgba[1, 0] = (255, 49, 49, 255)

        added = extract_stroke_mask(rgba, "Ajouter")
        erased = extract_stroke_mask(rgba, "Effacer")

        self.assertEqual(int(added.sum()), 1)
        self.assertTrue(added[0, 0])
        self.assertEqual(int(erased.sum()), 1)
        self.assertTrue(erased[1, 0])

    def test_add_and_erase_are_limited_to_crop(self):
        masks = {"Gonade": np.zeros((8, 10), dtype=bool)}
        stroke = np.zeros((4, 6), dtype=bool)
        stroke[1:3, 2:4] = True

        apply_stroke_to_masks(masks, "Gonade", stroke, "Ajouter", (2, 2, 6, 4))
        self.assertGreater(int(masks["Gonade"].sum()), 0)
        self.assertFalse(masks["Gonade"][:2].any())
        self.assertFalse(masks["Gonade"][:, :2].any())

        apply_stroke_to_masks(masks, "Gonade", stroke, "Effacer", (2, 2, 6, 4))
        self.assertFalse(masks["Gonade"].any())

    def test_rejects_crop_outside_mask(self):
        masks = {"Gonade": np.zeros((4, 4), dtype=bool)}
        with self.assertRaises(ValueError):
            apply_stroke_to_masks(
                masks, "Gonade", np.ones((2, 2), dtype=bool), "Ajouter", (3, 3, 2, 2)
            )

    def test_pending_stroke_is_applied_once_before_validation(self):
        edit_state = {
            "masks": {"Gonade": np.zeros((8, 10), dtype=bool)},
            "pending": {
                "class_name": "Gonade",
                "mode": "Ajouter",
                "stroke": np.ones((4, 6), dtype=bool),
            },
        }

        self.assertTrue(apply_pending_stroke(edit_state, (2, 2, 6, 4)))
        first_result = edit_state["masks"]["Gonade"].copy()
        self.assertIsNone(edit_state["pending"])
        self.assertFalse(apply_pending_stroke(edit_state, (2, 2, 6, 4)))
        self.assertTrue(np.array_equal(edit_state["masks"]["Gonade"], first_result))


if __name__ == "__main__":
    unittest.main()
