import math
import unittest

import torch

from segmentation_metrics import metrics_by_class


class SegmentationMetricTests(unittest.TestCase):
    def test_overlap_and_surface_match_reference_formulas(self):
        true = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.bool)
        pred = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.bool)
        result = metrics_by_class(pred, true, scale_cm=2, image_height_px=2)
        self.assertAlmostEqual(result['iou'].item(), 1 / 3, places=6)
        self.assertAlmostEqual(result['dice'].item(), .5, places=6)
        self.assertAlmostEqual(result['precision'].item(), .5, places=6)
        self.assertAlmostEqual(result['recall'].item(), .5, places=6)
        self.assertEqual(result['diff_surface'].item(), 0)

    def test_empty_mask_conventions(self):
        empty = torch.zeros((1, 2, 2), dtype=torch.bool)
        result = metrics_by_class(empty, empty, scale_cm=2, image_height_px=2)
        self.assertEqual(result['iou'].item(), 1)
        self.assertAlmostEqual(result['dice'].item(), 1, places=6)
        self.assertEqual(result['precision'].item(), 0)
        self.assertEqual(result['recall'].item(), 0)
        self.assertTrue(math.isnan(result['diff_surface'].item()))

    def test_missing_scale_only_disables_surface(self):
        true = torch.ones((1, 2, 2), dtype=torch.bool)
        pred = true.clone()
        result = metrics_by_class(pred, true, scale_cm=float('nan'), image_height_px=2)
        self.assertAlmostEqual(result['dice'].item(), 1, places=6)
        self.assertTrue(math.isnan(result['diff_surface'].item()))

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            metrics_by_class(torch.zeros(2, 2), torch.zeros(2, 2))


if __name__ == '__main__':
    unittest.main()
