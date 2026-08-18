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
            EGG_CORRECTED_IMAGES_DIR=root / "egg_corrected" / "images",
            EGG_CORRECTED_ANN_PATH=root / "egg_corrected" / "annotations.json",
            EGG_TRAIN_ADDED_IMAGES_DIR=root / "egg_train" / "images",
            EGG_TRAIN_ADDED_ANN_PATH=root / "egg_train" / "annotations.json",
            EGG_LABELLISATION_IMAGES_DIR=root / "egg_label" / "images",
            EGG_LABELLISATION_ANN_PATH=root / "egg_label" / "annotations.json",
            COCO_ANN_PATH=root / "COCO" / "annotations.json",
            EGG_COCO_ANN_PATH=root / "COCO" / "annotations.oeufs.json",
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

    def test_egg_stores_are_round_tripped_and_isolated(self):
        filename = "0000002_10.00.00_123456_2026.jpg"
        image = Image.new("RGB", (10, 8), (40, 50, 60))
        egg_mask = {"Oeuf": np.zeros((8, 10), dtype=bool)}
        egg_mask["Oeuf"][2:5, 3:8] = True

        data_utils.add_image_to_corrected_set(
            filename, image, egg_mask, image_type="oeufs"
        )
        loaded = data_utils.load_corrected_image_and_masks(filename, image_type="oeufs")
        self.assertTrue(np.array_equal(loaded["masks"]["Oeuf"], egg_mask["Oeuf"]))
        self.assertIsNone(data_utils.load_corrected_image_and_masks(filename))

        data_utils.add_image_to_training_set(filename, image, egg_mask, image_type="oeufs")
        self.assertEqual(data_utils.get_disk_status(filename, image_type="oeufs"), "bonne")
        self.assertIsNone(data_utils.get_disk_status(filename, image_type="cavite"))

        data_utils.remove_image_from_training_set(filename, image_type="oeufs")
        data_utils.remove_image_from_corrected_set(filename, image_type="oeufs")
        self.assertIsNone(data_utils.get_disk_status(filename, image_type="oeufs"))

    @staticmethod
    def _write_coco_annotation(path, filename, category_id):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": filename, "width": 10, "height": 8}
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": category_id,
                            "segmentation": [[2, 2, 7, 2, 7, 6, 2, 6]],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_loads_coco_ground_truth_for_cavity_and_eggs(self):
        cavity_filename = "0000003_10.00.00_123456_2026.jpg"
        egg_filename = "0000004_10.00.00_123456_2026.jpg"
        self._write_coco_annotation(config.COCO_ANN_PATH, cavity_filename, category_id=2)
        self._write_coco_annotation(
            config.EGG_COCO_ANN_PATH, egg_filename, category_id=1
        )

        cavity_masks = data_utils.load_coco_ground_truth_masks(cavity_filename)
        egg_masks = data_utils.load_coco_ground_truth_masks(
            egg_filename, image_type="oeufs"
        )

        self.assertEqual(set(cavity_masks), set(config.CLASS_NAMES))
        self.assertTrue(cavity_masks["Gonade"].any())
        self.assertFalse(cavity_masks["Cavite"].any())
        self.assertEqual(set(egg_masks), {"Oeuf"})
        self.assertTrue(egg_masks["Oeuf"].any())

    def test_coco_ground_truth_absence_and_unknown_type(self):
        self.assertIsNone(data_utils.load_coco_ground_truth_masks("absente.jpg"))
        self.assertIsNone(
            data_utils.load_coco_ground_truth_masks("absente.jpg", image_type="oeufs")
        )
        with self.assertRaises(ValueError):
            data_utils.load_coco_ground_truth_masks("image.jpg", image_type="inconnu")


if __name__ == "__main__":
    unittest.main()
