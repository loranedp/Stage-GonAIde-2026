import unittest

import cv2
import numpy as np
import torch

from utils.post_traitement import split_eggs


class EggPostProcessingTests(unittest.TestCase):
    def test_empty_mask_returns_empty_label_map(self):
        labels = split_eggs(torch.zeros((1, 1, 64, 64), dtype=torch.bool))
        self.assertEqual(tuple(labels.shape), (1, 64, 64))
        self.assertEqual(int(labels.max()), 0)

    def test_watershed_separates_touching_eggs(self):
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (38, 48), 20, 1, -1)
        cv2.circle(mask, (58, 48), 20, 1, -1)

        labels = split_eggs(
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
            min_distance=8,
            min_area=50,
        )[0].numpy()

        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertTrue(np.all(labels[mask == 0] == 0))

    def test_small_noise_is_removed_and_labels_are_contiguous(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(mask, (20, 20), 8, 1, -1)
        cv2.circle(mask, (60, 60), 8, 1, -1)
        mask[40, 40] = 1

        labels = split_eggs(
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0), min_area=20
        )[0].numpy()
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertEqual(int(labels[40, 40]), 0)


if __name__ == "__main__":
    unittest.main()
