import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import config
import data_utils


class DataUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.image_dir = root / "Corrigees" / "images"
        self.ann_path = root / "Corrigees" / "annotations.json"
        self.config_patch = mock.patch.multiple(
            config,
            CORRECTED_IMAGES_DIR=self.image_dir,
            CORRECTED_ANN_PATH=self.ann_path,
            TRAIN_ADDED_IMAGES_DIR=root / "train" / "images",
            TRAIN_ADDED_ANN_PATH=root / "train" / "annotations.json",
            LABELLISATION_IMAGES_DIR=root / "label" / "images",
            LABELLISATION_ANN_PATH=root / "label" / "annotations.json",
        )
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def masks(active_class="Gonade"):
        result = {name: np.zeros((8, 10), dtype=bool) for name in config.CLASS_NAMES}
        result[active_class][2:6, 3:7] = True
        return result

    def test_round_trip_replace_and_remove(self):
        filename = "0000001_10.00.00_123456_2026.jpg"
        image = Image.new("RGB", (10, 8), (20, 30, 40))

        data_utils.add_image_to_corrected_set(filename, image, self.masks())
        loaded = data_utils.load_corrected_image_and_masks(filename)
        self.assertEqual(loaded["image"].size, image.size)
        self.assertTrue(np.array_equal(loaded["masks"]["Gonade"], self.masks()["Gonade"]))

        replacement = self.masks("Cavite")
        data_utils.add_image_to_corrected_set(filename, image, replacement)
        loaded = data_utils.load_corrected_image_and_masks(filename)
        self.assertFalse(loaded["masks"]["Gonade"].any())
        self.assertTrue(np.array_equal(loaded["masks"]["Cavite"], replacement["Cavite"]))

        data_utils.remove_image_from_corrected_set(filename)
        self.assertIsNone(data_utils.load_corrected_image_and_masks(filename))
        self.assertFalse((self.image_dir / filename).exists())

    def test_failed_json_write_restores_previous_image_and_annotations(self):
        filename = "0000001_10.00.00_123456_2026.jpg"
        data_utils.add_image_to_corrected_set(
            filename, Image.new("RGB", (10, 8), (10, 20, 30)), self.masks()
        )
        image_path = self.image_dir / filename
        previous_image = image_path.read_bytes()
        previous_json = self.ann_path.read_bytes()

        with mock.patch.object(
            data_utils, "_save_added_annotations", side_effect=OSError("disk full")
        ):
            with self.assertRaises(data_utils.DataStoreError):
                data_utils.add_image_to_corrected_set(
                    filename, Image.new("RGB", (10, 8), (200, 210, 220)), self.masks("Cavite")
                )

        self.assertEqual(image_path.read_bytes(), previous_image)
        self.assertEqual(self.ann_path.read_bytes(), previous_json)

    def test_invalid_store_and_unsafe_filename_are_rejected(self):
        self.ann_path.parent.mkdir(parents=True, exist_ok=True)
        self.ann_path.write_text(json.dumps({"images": [{}], "annotations": []}))
        with self.assertRaises(data_utils.DataStoreError):
            data_utils.load_added_annotations(self.ann_path)
        with self.assertRaises(ValueError):
            data_utils.add_image_to_corrected_set(
                "../escape.jpg", Image.new("RGB", (10, 8)), self.masks()
            )


if __name__ == "__main__":
    unittest.main()
