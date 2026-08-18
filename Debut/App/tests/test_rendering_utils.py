import unittest

import numpy as np
from PIL import Image

from rendering_utils import draw_instance_contours


class InstanceContourRenderingTests(unittest.TestCase):
    def test_draws_black_separation_while_preserving_instance_fill(self):
        orange = (245, 170, 35)
        image_array = np.full((24, 24, 3), 90, dtype=np.uint8)
        instances = np.zeros((24, 24), dtype=np.int32)
        instances[3:21, 3:12] = 1
        instances[3:21, 12:21] = 2
        image_array[instances > 0] = orange

        rendered = np.asarray(
            draw_instance_contours(Image.fromarray(image_array), instances)
        )

        self.assertTrue(np.array_equal(rendered[10, 11], (0, 0, 0)))
        self.assertTrue(np.array_equal(rendered[10, 12], (0, 0, 0)))
        self.assertTrue(np.array_equal(rendered[10, 6], orange))
        self.assertTrue(np.array_equal(rendered[0, 0], (90, 90, 90)))

    def test_empty_instance_map_leaves_image_unchanged(self):
        image_array = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
        instances = np.zeros((12, 10), dtype=np.int32)

        rendered = np.asarray(
            draw_instance_contours(Image.fromarray(image_array), instances)
        )

        self.assertTrue(np.array_equal(rendered, image_array))

    def test_rejects_an_instance_map_with_different_dimensions(self):
        image = Image.new("RGB", (10, 12))

        with self.assertRaisesRegex(ValueError, "mêmes dimensions"):
            draw_instance_contours(image, np.zeros((11, 10), dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
