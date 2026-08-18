import unittest

import numpy as np

from metrics_utils import compute_iou_per_class


class MetricsUtilsTests(unittest.TestCase):
    def test_egg_iou_for_identical_partial_and_empty_masks(self):
        first = np.zeros((2, 2), dtype=bool)
        first[0, :] = True
        identical = first.copy()
        partial = np.zeros((2, 2), dtype=bool)
        partial[:, 1] = True
        empty = np.zeros((2, 2), dtype=bool)

        self.assertEqual(
            compute_iou_per_class(
                {"Oeuf": first}, {"Oeuf": identical}, ["Oeuf"]
            )["Oeuf"],
            1.0,
        )
        self.assertAlmostEqual(
            compute_iou_per_class(
                {"Oeuf": first}, {"Oeuf": partial}, ["Oeuf"]
            )["Oeuf"],
            1 / 3,
        )
        self.assertEqual(
            compute_iou_per_class(
                {"Oeuf": empty}, {"Oeuf": empty}, ["Oeuf"]
            )["Oeuf"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
