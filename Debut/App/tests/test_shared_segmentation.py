import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.models_config import metrics_by_class as legacy_metrics_by_class
from utils.segmentation_metrics import metrics_by_class, metrics_from_counts
from utils.segmentation_visualization import (
    overlay_semantic_instances,
    save_overlay_comparison,
)
from utils.stats_fct import overlay_colored_mask as legacy_overlay_colored_mask
from utils.segmentation_visualization import overlay_colored_mask


class SharedSegmentationTests(unittest.TestCase):
    def test_legacy_metric_import_reexports_shared_function(self):
        self.assertIs(legacy_metrics_by_class, metrics_by_class)

    def test_legacy_visualization_import_reexports_shared_function(self):
        self.assertIs(legacy_overlay_colored_mask, overlay_colored_mask)

    def test_empty_counts_are_perfect(self):
        values = metrics_from_counts(
            intersection=[0], union=[0], predicted_area=[0], true_area=[0]
        )
        self.assertEqual(values["iou"][0], 1.0)
        self.assertEqual(values["dice"][0], 1.0)
        self.assertEqual(values["precision"][0], 1.0)
        self.assertEqual(values["recall"][0], 1.0)

    def test_semantic_and_instance_overlay_keeps_both_layers(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        semantic = np.zeros((1, 8, 8), dtype=np.uint8)
        instances = np.zeros((8, 8), dtype=np.int32)
        semantic[0, 1:4, 1:4] = 1
        instances[4:7, 4:7] = 1

        overlay = overlay_semantic_instances(
            image, semantic, instances, alpha=1.0
        )

        self.assertTrue(np.any(overlay[1:4, 1:4]))
        self.assertTrue(np.any(overlay[4:7, 4:7]))

    def test_comparison_export_has_twice_the_image_width(self):
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        mask = np.zeros((1, 8, 9), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"
            save_overlay_comparison(output, image, mask, mask)
            with Image.open(output) as exported:
                self.assertEqual(exported.size, (18, 8))


if __name__ == "__main__":
    unittest.main()
