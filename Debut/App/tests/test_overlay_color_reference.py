import unittest

import numpy as np
import torch

from utils.stats_fct import (
    CLASS_OVERLAY_COLORS,
    INSTANCE_OVERLAY_COLORS,
    overlay_colored_mask,
    overlay_mask,
)


class OverlayColorReferenceTests(unittest.TestCase):
    def test_colored_semantic_overlay_uses_the_shared_reference(self):
        image = np.zeros((1, 3, 3), dtype=np.uint8)
        mask = np.zeros((3, 1, 3), dtype=np.float32)
        mask[0, 0, 0] = 1
        mask[1, 0, 1] = 1
        mask[2, 0, 2] = 1

        overlay = overlay_colored_mask(image, mask, alpha=1)

        np.testing.assert_array_equal(overlay[0, :3], CLASS_OVERLAY_COLORS)

    def test_legacy_overlay_uses_the_shared_reference(self):
        image = np.zeros((3, 5, 5), dtype=np.float32)
        mask = torch.zeros((3, 5, 5), dtype=torch.float32)
        mask[0, 0, 0] = 1
        mask[1, 0, 1] = 1
        mask[2, 0, 2] = 1

        overlay = overlay_mask(image, mask, alpha=1)

        np.testing.assert_array_equal(overlay[0, :3], CLASS_OVERLAY_COLORS)

    def test_egg_instances_keep_their_dedicated_palette(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        instances = np.array([[1]], dtype=np.int32)

        overlay = overlay_colored_mask(
            image, instances, alpha=1, dataset_name="oeufs"
        )

        np.testing.assert_array_equal(overlay[0, 0], INSTANCE_OVERLAY_COLORS[0])


if __name__ == "__main__":
    unittest.main()
