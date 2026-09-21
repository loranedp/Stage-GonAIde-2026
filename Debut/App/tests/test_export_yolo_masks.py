import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


YOLO_ROOT = Path(__file__).resolve().parents[2] / "YOLO26"
sys.path.insert(0, str(YOLO_ROOT))

from export_yolo_masks import instance_masks_to_yolo_lines
from utils.yolo_metrics import load_yolo_polygon_instances


class ExportYoloMasksTests(unittest.TestCase):
    def test_export_preserves_instances_at_image_edges(self):
        image_shape = (38, 51)
        top_left_mask = np.zeros(image_shape, dtype=np.uint8)
        top_left_mask[0:6, 0:7] = 1
        bottom_right_mask = np.zeros(image_shape, dtype=np.uint8)
        bottom_right_mask[-7:, -8:] = 1

        lines = instance_masks_to_yolo_lines(
            [(0, top_left_mask), (0, bottom_right_mask)], image_shape
        )

        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "prediction.txt"
            label_path.write_text("\n".join(lines) + "\n")
            restored = load_yolo_polygon_instances(label_path, image_shape)

        self.assertGreater(np.count_nonzero(restored[0:6, 0:7]), 0)
        self.assertGreater(np.count_nonzero(restored[-7:, -8:]), 0)

    def test_export_rejects_a_mask_in_another_coordinate_frame(self):
        mask = np.zeros((20, 30), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "mêmes dimensions"):
            instance_masks_to_yolo_lines([(0, mask)], (38, 51))


if __name__ == "__main__":
    unittest.main()
