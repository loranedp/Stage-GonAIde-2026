import tempfile
import unittest
from pathlib import Path

import numpy as np

from Resultats.visualisation_modeles import mask_from_yolo_file


class VisualisationModelesTests(unittest.TestCase):
    def test_oeufsclasses_keeps_only_egg_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction_path = Path(directory) / "prediction.txt"
            prediction_path.write_text(
                "0 0.05 0.05 0.45 0.05 0.45 0.45 0.05 0.45\n"
                "1 0.55 0.05 0.70 0.05 0.70 0.20 0.55 0.20\n"
                "1 0.75 0.75 0.95 0.75 0.95 0.95 0.75 0.95\n"
            )

            mask = mask_from_yolo_file(
                prediction_path,
                num_classes=1,
                width=100,
                height=100,
                preserve_instances=True,
                class_id_map={1: 0},
            )

        self.assertEqual(mask[20, 20], 0)
        self.assertEqual(set(np.unique(mask)), {0, 1, 2})
        self.assertGreater(mask[10, 60], 0)
        self.assertGreater(mask[85, 85], 0)


if __name__ == "__main__":
    unittest.main()
