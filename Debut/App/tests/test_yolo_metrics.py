import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.yolo_metrics import (
    load_yolo_polygon_masks,
    metrics_from_counts,
)


class YoloMetricsTests(unittest.TestCase):
    def write_labels(self, content):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "sample.txt"
        path.write_text(content)
        self.addCleanup(directory.cleanup)
        return path

    def test_instance_masks_preserve_nested_polygons(self):
        labels = self.write_labels(
            "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
            "1 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n"
            "2 0.45 0.45 0.55 0.45 0.55 0.55 0.45 0.55\n"
        )

        masks = load_yolo_polygon_masks(labels, (100, 100), 3)

        self.assertGreater(masks[0].sum(), 0)
        self.assertGreater(masks[1].sum(), 0)
        self.assertGreater(masks[2].sum(), 0)
        self.assertGreater(int((masks[0] & masks[1]).sum()), 0)
        self.assertGreater(int((masks[1] & masks[2]).sum()), 0)

    def test_instance_masks_preserve_nested_class_overlap(self):
        labels = self.write_labels(
            "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
            "1 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n"
        )

        masks = load_yolo_polygon_masks(labels, (100, 100), 3)

        self.assertGreater(int((masks[0] & masks[1]).sum()), 0)

    def test_metrics_from_pooled_counts(self):
        metrics = metrics_from_counts(
            intersection=np.array([6, 0]),
            union=np.array([10, 0]),
            predicted_area=np.array([8, 0]),
            true_area=np.array([8, 0]),
        )

        self.assertAlmostEqual(metrics["iou"][0], 0.6)
        self.assertAlmostEqual(metrics["dice"][0], 0.75)
        self.assertAlmostEqual(metrics["precision"][0], 0.75)
        self.assertAlmostEqual(metrics["recall"][0], 0.75)
        self.assertEqual(metrics["iou"][1], 1.0)


if __name__ == "__main__":
    unittest.main()
