"""Vérifications des volumes individuels d'œufs et de leur export."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import config
import model_utils
import result_workflow
from utils.calcul_volume_fct import calculate_egg_volume_summary


class EggFecundityTests(unittest.TestCase):
    def test_empty_image_does_not_change_pooled_median(self):
        images = [
            {"id_image": "empty", "areas_px": [], "echelle": 1, "image_height": 10},
            {"id_image": "one", "areas_px": [1], "echelle": 1, "image_height": 10},
            {"id_image": "three", "areas_px": [4, 9, 16], "echelle": 1, "image_height": 10},
        ]
        result = calculate_egg_volume_summary(images)
        individual_volumes = [4 / 3 * np.pi * (area / np.pi) ** 1.5 for area in (1, 4, 9, 16)]
        self.assertEqual(result["images"][0]["nombre_oeufs_distincts"], 0)
        self.assertIsNone(result["images"][0]["volume_median_oeufs_mm3"])
        self.assertAlmostEqual(result["images"][2]["volume_median_oeufs_mm3"], individual_volumes[2])
        self.assertAlmostEqual(result["volume_median_oeufs_mm3"], np.median(individual_volumes))
        self.assertAlmostEqual(result["volume_q1_oeufs_mm3"], np.percentile(individual_volumes, 25))
        self.assertAlmostEqual(result["volume_q3_oeufs_mm3"], np.percentile(individual_volumes, 75))
        self.assertLess(
            result_workflow.calculate_fecundity(1, result["volume_q3_oeufs_mm3"]),
            result_workflow.calculate_fecundity(1, result["volume_q1_oeufs_mm3"]),
        )

    def test_one_egg_yields_point_interval(self):
        instances = np.zeros((10, 10), dtype=np.int32)
        instances[1:4, 1:4] = 1
        summary = calculate_egg_volume_summary([
            {"id_image": "one", "instances": instances, "echelle": 1, "image_height": 10},
        ])
        self.assertEqual(summary["volume_q1_oeufs_mm3"], summary["volume_q3_oeufs_mm3"])
        self.assertIsNotNone(result_workflow.calculate_fecundity(1, summary["volume_median_oeufs_mm3"]))

    def test_fish_fecundity_uses_eggs_from_nonempty_image(self):
        image = Image.new("RGB", (10, 10))
        cavity_masks = {"Gonade": np.ones((10, 10), dtype=bool),
                        "Cavite": np.ones((10, 10), dtype=bool)}
        predictions = {
            "cavity:1": {"id_poisson": "fish", "id_image": "c1", "filename": "c1.jpg",
                         "image_type": "cavite", "image": image, "masks": cavity_masks},
            "cavity:2": {"id_poisson": "fish", "id_image": "c2", "filename": "c2.jpg",
                         "image_type": "cavite", "image": image, "masks": cavity_masks},
            "egg:empty": {"id_poisson": "fish", "id_image": "empty", "filename": "empty.jpg",
                          "image_type": "oeufs", "image": image,
                          "masks": {"Oeuf": np.zeros((10, 10), dtype=bool)},
                          "instances": np.zeros((10, 10), dtype=np.int32), "egg_areas_px": []},
            "egg:one": {"id_poisson": "fish", "id_image": "one", "filename": "one.jpg",
                        "image_type": "oeufs", "image": image,
                        "masks": {"Oeuf": np.ones((10, 10), dtype=bool)},
                        "instances": np.ones((10, 10), dtype=np.int32), "egg_areas_px": [4]},
        }
        metadata = {
            "c1": {"echelle": 1, "position": 1, "long_gonade": 3},
            "c2": {"echelle": 1, "position": 2, "long_gonade": 3},
            "empty": {"echelle": 1}, "one": {"echelle": 1},
        }
        with patch.object(result_workflow.metadata_utils, "lookup", side_effect=metadata.get):
            result = result_workflow.calculate_fish_result("fish", predictions, {})
        self.assertEqual([row["nombre_oeufs_distincts"] for row in result["egg_rows"]], [0, 1])
        self.assertIsNotNone(result["fecundity"])
        self.assertEqual(result["fecundity_low"], result["fecundity_high"])

    def test_overlapping_yolo_masks_keep_both_areas(self):
        class MaskData:
            def __init__(self, masks):
                self.data = self
                self.masks = masks

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.masks

        class Prediction:
            def __init__(self, masks):
                self.masks = MaskData(masks)
                self.boxes = None

        class Model:
            def predict(self, **kwargs):
                return [Prediction(np.ones((2, 4, 4), dtype=float))]

        with patch.object(config, "CROP_PARAMS", [0, 0, 4, 4]):
            result = model_utils.predict_eggs(Model(), Image.new("RGB", (4, 4)))
        self.assertEqual(result["egg_areas_px"], [16, 16])
        self.assertEqual(int(result["instances"].max()), 2)

    def test_legacy_csv_mean_is_preserved_when_saving_median(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            pd.DataFrame([{
                "Id_poisson": "old", "Volume_gonades": 2.0,
                "Volume_cavite": 3.0, "Volume_moyen_oeufs_mm3": 4.0,
                "Fecondite_estimee": 500,
            }]).to_csv(path, index=False)
            result = {
                "selected_fish": "new", "volume_total": 2.0,
                "volume_total_cavite": 3.0, "egg_median": 4.0,
                "fecundity": 500, "fecundity_low": 400, "fecundity_high": 600,
            }
            result_workflow.save_fish_results([result], path)
            saved = pd.read_csv(path, dtype={"Id_poisson": str}).set_index("Id_poisson")
            self.assertEqual(saved.loc["old", "Volume_moyen_oeufs_mm3"], 4.0)
            self.assertTrue(pd.isna(saved.loc["old", "Volume_median_oeufs_mm3"]))
            self.assertEqual(saved.loc["new", "Volume_median_oeufs_mm3"], 4.0)
            self.assertTrue(pd.isna(saved.loc["new", "Volume_moyen_oeufs_mm3"]))
            self.assertEqual(saved.loc["new", "Fecondite_IQR_borne_basse"], 400)
            self.assertEqual(saved.loc["new", "Fecondite_IQR_borne_haute"], 600)


if __name__ == "__main__":
    unittest.main()
