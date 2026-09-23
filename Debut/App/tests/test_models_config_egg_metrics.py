import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.models_config import (
    _median_instance_area_px,
    metrics_by_class,
    save_validation_masks,
)
from utils.post_traitement import split_eggs
from utils.recover_unet_egg_exports import recover_instance_labels
from utils.stats_fct import overlay_colored_mask


class EggSurfaceMetricTests(unittest.TestCase):
    def test_median_uses_all_instances(self):
        labels = torch.zeros((1, 1, 200), dtype=torch.int32)
        for label_id, area in enumerate((20, 40, 60, 80), start=1):
            start = sum((20, 40, 60, 80)[: label_id - 1])
            labels[0, 0, start : start + area] = label_id

        self.assertEqual(_median_instance_area_px(labels).item(), 50.0)

    def test_egg_surface_difference_is_in_mm2(self):
        true = torch.zeros((1, 1, 100, 400), dtype=torch.bool)
        for start, width in ((0, 20), (60, 30), (120, 40), (200, 50)):
            true[0, 0, 20:40, start : start + width] = True

        predicted_labels = torch.zeros((1, 100, 400), dtype=torch.int32)
        for label_id, (start, width) in enumerate(((0, 25), (60, 35), (120, 45), (200, 55)), start=1):
            predicted_labels[0, 20:40, start : start + width] = label_id
        preds = (predicted_labels > 0).unsqueeze(1)
        scale_cm = torch.tensor([[2.0]])

        *_, diff_surface = metrics_by_class(
            true, preds, scale_cm, "oeufs", egg_instance_labels=predicted_labels
        )
        true_labels = split_eggs(
            true, min_distance=8, min_area=100, ouverture_size=3, ouverture_iterations=1
        )
        expected = (
            _median_instance_area_px(true_labels)
            - _median_instance_area_px(predicted_labels)
        ).abs() * (20.0 / 100) ** 2

        self.assertAlmostEqual(diff_surface.item(), expected.item(), places=5)

    def test_export_keeps_adjacent_egg_instances_separate(self):
        labels = torch.zeros((512, 512), dtype=torch.int32)
        labels[166:186, 101:121] = 1
        labels[166:186, 121:141] = 2

        with tempfile.TemporaryDirectory() as directory:
            save_validation_masks(
                directory,
                [("sample.jpg", torch.empty(0), labels)],
                dataset_name="oeufs",
            )
            lines = (Path(directory) / "pred_sample.txt").read_text().splitlines()

        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.startswith("0 ") for line in lines))

    def test_instance_labels_are_recovered_exactly_from_overlay(self):
        image = np.arange(30 * 36 * 3, dtype=np.uint8).reshape(30, 36, 3)
        labels = np.zeros((30, 36), dtype=np.int32)
        for instance_id in range(1, 10):
            row, column = divmod(instance_id - 1, 3)
            labels[row * 10 + 1 : row * 10 + 8,
                   column * 12 + 1 : column * 12 + 10] = instance_id
        overlay = overlay_colored_mask(
            image, labels, alpha=0.45, dataset_name="oeufs"
        )

        recovered = recover_instance_labels(image, overlay, labels > 0)

        np.testing.assert_array_equal(recovered, labels)


if __name__ == "__main__":
    unittest.main()
