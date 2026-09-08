import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

import result_workflow


class ResultWorkflowTests(unittest.TestCase):
    @staticmethod
    def upload(name):
        stream = BytesIO()
        Image.new("RGB", (10, 8), (20, 30, 40)).save(stream, format="PNG")
        return result_workflow.UploadedImage(name, stream.getvalue())

    @staticmethod
    def cavity_masks(active=True):
        masks = {
            "Cavite": np.zeros((8, 10), dtype=bool),
            "Gonade": np.zeros((8, 10), dtype=bool),
            "Intestin": np.zeros((8, 10), dtype=bool),
        }
        if active:
            masks["Cavite"][1:7, 1:9] = True
            masks["Gonade"][2:6, 2:8] = True
        return masks

    def test_predicts_both_upload_types_and_loads_each_model_family_once(self):
        uploads = {
            "cavite": [
                self.upload("1000001_10.00.00_123_2026.png"),
                self.upload("1000002_10.01.00_123_2026.png"),
            ],
            "oeufs": [self.upload("1000003_10.02.00_123_2026.png")],
        }
        egg_instances = np.zeros((8, 10), dtype=np.int32)
        egg_result = {
            "masks": {"Oeuf": egg_instances > 0},
            "instances": egg_instances,
            "confidences": {"Oeuf": None},
        }
        cavity_result = {
            "masks": self.cavity_masks(),
            "confidences": {"Cavite": 0.9, "Gonade": 0.8, "Intestin": None},
        }

        with (
            mock.patch.object(
                result_workflow.data_utils,
                "load_corrected_image_and_masks",
                return_value=None,
            ),
            mock.patch.object(
                result_workflow.data_utils, "get_disk_status", return_value=None
            ),
            mock.patch.object(
                result_workflow.model_utils, "load_models", return_value=[object()]
            ) as load_cavity,
            mock.patch.object(
                result_workflow.model_utils,
                "load_egg_models",
                return_value=[object()],
            ) as load_eggs,
            mock.patch.object(
                result_workflow.model_utils, "predict", return_value=cavity_result
            ),
            mock.patch.object(
                result_workflow.model_utils, "predict_eggs", return_value=egg_result
            ),
        ):
            batch = result_workflow.predict_uploads(uploads)

        self.assertEqual(len(batch.predictions), 3)
        self.assertFalse(batch.issues)
        load_cavity.assert_called_once_with()
        load_eggs.assert_called_once_with()

    def test_prediction_failure_marks_only_the_affected_fish(self):
        uploads = {
            "cavite": [
                self.upload("1000001_10.00.00_123_2026.png"),
                self.upload("1000002_10.01.00_456_2026.png"),
            ]
        }
        cavity_result = {
            "masks": self.cavity_masks(),
            "confidences": {},
        }

        with (
            mock.patch.object(
                result_workflow.data_utils,
                "load_corrected_image_and_masks",
                return_value=None,
            ),
            mock.patch.object(
                result_workflow.data_utils, "get_disk_status", return_value=None
            ),
            mock.patch.object(
                result_workflow.model_utils, "load_models", return_value=[object()]
            ),
            mock.patch.object(
                result_workflow.model_utils,
                "predict",
                side_effect=[ValueError("prediction impossible"), cavity_result],
            ),
        ):
            batch = result_workflow.predict_uploads(uploads)

        self.assertEqual(batch.failed_fish_ids, {"123"})
        self.assertEqual({pred["id_poisson"] for pred in batch.predictions.values()}, {"456"})

    def test_saved_correction_is_used_without_loading_models(self):
        upload = self.upload("1000001_10.00.00_123_2026.png")
        corrected = {
            "image": Image.new("RGB", (10, 8)),
            "masks": self.cavity_masks(),
        }
        with (
            mock.patch.object(
                result_workflow.data_utils,
                "load_corrected_image_and_masks",
                return_value=corrected,
            ),
            mock.patch.object(
                result_workflow.data_utils, "get_disk_status", return_value="bonne"
            ),
            mock.patch.object(result_workflow.model_utils, "load_models") as load_models,
            mock.patch.object(result_workflow.model_utils, "predict") as predict,
        ):
            batch = result_workflow.predict_uploads({"cavite": [upload]})

        prediction_key = f"cavite:{upload.name}"
        self.assertIn(prediction_key, batch.corrected_keys)
        self.assertEqual(batch.statuses[prediction_key], "bonne")
        load_models.assert_not_called()
        predict.assert_not_called()

    def test_calculation_excludes_bad_images(self):
        image = Image.new("RGB", (10, 8))
        predictions = {
            "cavite:first": {
                "filename": "first.png",
                "image_type": "cavite",
                "image": image,
                "masks": self.cavity_masks(),
                "id_image": "1",
                "id_poisson": "123",
            },
            "cavite:bad": {
                "filename": "bad.png",
                "image_type": "cavite",
                "image": image,
                "masks": self.cavity_masks(False),
                "id_image": "2",
                "id_poisson": "123",
            },
        }
        metadata = {
            "echelle": 2.0,
            "position": 1.0,
            "long_gonade": 5.0,
        }
        with mock.patch.object(
            result_workflow.metadata_utils, "lookup", return_value=metadata
        ):
            result = result_workflow.calculate_fish_result(
                "123", predictions, {"cavite:bad": "mauvaise"}
            )

        self.assertEqual(len(result["rows"]), 1)
        self.assertIsNone(result["volume_total"])
        self.assertTrue(result["has_eligible_images"])

    def test_save_replaces_valid_fish_and_preserves_others(self):
        columns = result_workflow.RESULT_CSV_COLUMNS
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "results" / "volumes.csv"
            csv_path.parent.mkdir()
            pd.DataFrame(
                [
                    ["123", 1.0, 2.0, None, None],
                    ["999", 3.0, 4.0, 20.0, 150],
                ],
                columns=columns,
            ).to_csv(csv_path, index=False)
            valid = {
                "selected_fish": "123",
                "volume_total": 5.123456,
                "volume_total_cavite": 6.987654,
                "egg_mean": 15.0599,
                "fecundity": 340.6,
            }
            skipped = {
                "selected_fish": "456",
                "volume_total": None,
                "volume_total_cavite": None,
                "egg_mean": None,
                "fecundity": None,
            }

            saved_count = result_workflow.save_fish_results(
                [valid, skipped], csv_path
            )
            saved = pd.read_csv(csv_path, dtype={"Id_poisson": str})

        self.assertEqual(saved_count, 1)
        self.assertEqual(set(saved["Id_poisson"]), {"123", "999"})
        self.assertEqual(saved.columns.tolist(), columns)
        updated = saved.loc[saved["Id_poisson"] == "123"].iloc[0]
        self.assertEqual(updated["Volume_gonades"], 5.1235)
        self.assertEqual(updated["Volume_cavite"], 6.9877)
        self.assertEqual(updated["Volume_moyen_oeufs_mm3"], 15.0599)
        self.assertEqual(updated["Fecondite_estimee"], 341)

    def test_partial_summary_is_exportable_but_empty_summary_is_not(self):
        partial = {
            "selected_fish": "123",
            "volume_total": None,
            "volume_total_cavite": None,
            "egg_mean": 12.345,
            "fecundity": None,
        }
        empty = {
            "selected_fish": "456",
            "volume_total": None,
            "volume_total_cavite": None,
            "egg_mean": None,
            "fecundity": None,
        }

        self.assertTrue(result_workflow.has_exportable_result(partial))
        self.assertFalse(result_workflow.has_exportable_result(empty))
        self.assertEqual(
            result_workflow.result_summary_row(partial)["Volume_moyen_oeufs_mm3"],
            12.345,
        )

    def test_migrates_legacy_rows_and_drops_fish_without_results(self):
        legacy = pd.DataFrame(
            [
                ["123", "1", 1.23456, 2.34567, 0.0150599, 678.57],
                ["123", "2", 1.23456, 2.34567, 0.0150599, 678.57],
                ["456", "3", None, None, None, None],
            ],
            columns=[
                "Id_poisson",
                "Id_image",
                "Volume_total_gonades_cm3",
                "Volume_total_cavité_cm3",
                "Volume_moyen_oeuf_cm3",
                "Fecondite_estimee",
            ],
        )

        migrated = result_workflow.migrate_results_dataframe(legacy)

        self.assertEqual(migrated.columns.tolist(), result_workflow.RESULT_CSV_COLUMNS)
        self.assertEqual(migrated["Id_poisson"].tolist(), ["123"])
        self.assertEqual(migrated.iloc[0]["Volume_gonades"], 1.2346)
        self.assertEqual(migrated.iloc[0]["Volume_cavite"], 2.3457)
        self.assertEqual(migrated.iloc[0]["Volume_moyen_oeufs_mm3"], 15.0599)
        self.assertEqual(migrated.iloc[0]["Fecondite_estimee"], 679)

    def test_migrates_previous_summary_volume_from_cm3_to_mm3(self):
        previous = pd.DataFrame(
            [["123", 1.0, 2.0, 0.015, 67]],
            columns=[
                "Id_poisson",
                "Volume_gonades",
                "Volume_cavite",
                "Volume_moyen_oeufs",
                "Fecondite_estimee",
            ],
        )

        migrated = result_workflow.migrate_results_dataframe(previous)

        self.assertEqual(migrated.iloc[0]["Volume_moyen_oeufs_mm3"], 15.0)

    def test_fecundity_converts_gonad_volume_from_cm3_to_mm3(self):
        self.assertEqual(result_workflow.calculate_fecundity(2.0, 4.0), 500.0)
        self.assertIsNone(result_workflow.calculate_fecundity(2.0, 0.0))
        self.assertIsNone(result_workflow.calculate_fecundity(2.0, np.nan))


if __name__ == "__main__":
    unittest.main()
