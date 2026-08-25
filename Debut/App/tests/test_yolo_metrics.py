import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.yolo_metrics import (
    load_yolo_polygon_masks,
    metrics_from_counts,
    semantic_result_to_masks,
)


class YoloMetricsTests(unittest.TestCase):
    def write_labels(self, content):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "sample.txt"
        path.write_text(content)
        self.addCleanup(directory.cleanup)
        return path

    def test_semantic_masks_make_nested_polygons_exclusive(self):
        labels = self.write_labels(
            "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
            "1 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n"
            "2 0.45 0.45 0.55 0.45 0.55 0.55 0.45 0.55\n"
        )

        masks = load_yolo_polygon_masks(labels, (100, 100), 3, semantic=True)

        self.assertGreater(masks[0].sum(), 0)
        self.assertGreater(masks[1].sum(), 0)
        self.assertGreater(masks[2].sum(), 0)
        self.assertEqual(int((masks.sum(axis=0) > 1).sum()), 0)
        self.assertEqual(int((masks[0] & masks[1]).sum()), 0)
        self.assertEqual(int((masks[1] & masks[2]).sum()), 0)

    def test_ignore_polygon_erases_the_underlying_semantic_class(self):
        labels = self.write_labels(
            "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
            "3 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n"
        )

        masks = load_yolo_polygon_masks(labels, (100, 100), 3, semantic=True)

        self.assertEqual(int(masks[0, 40:60, 40:60].sum()), 0)

    def test_instance_masks_preserve_nested_class_overlap(self):
        labels = self.write_labels(
            "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
            "1 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n"
        )

        masks = load_yolo_polygon_masks(labels, (100, 100), 3, semantic=False)

        self.assertGreater(int((masks[0] & masks[1]).sum()), 0)

    def test_semantic_prediction_channels_follow_class_names(self):
        semantic_map = np.array([[0, 1, 3], [2, 3, 0]], dtype=np.uint8)

        masks = semantic_result_to_masks(
            semantic_map,
            {0: "Cavite", 1: "Gonade", 2: "Intestin", 3: "background"},
            ["Cavite", "Gonade", "Intestin"],
        )

        self.assertEqual(masks[:, 0, :].tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 0]])

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
