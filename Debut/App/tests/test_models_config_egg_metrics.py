import unittest

import torch

from utils.models_config import _median_instance_area_px, metrics_by_class
from utils.post_traitement import split_eggs


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


if __name__ == "__main__":
    unittest.main()
