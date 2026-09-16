import unittest

import numpy as np
import torch
from PIL import Image

from utils.stats_fct import mask_to_yolo_polygons
from utils.unet_spatial import (
    CONTENT_OFFSET,
    CROPPED_IMAGE_SIZE,
    MODEL_IMAGE_SIZE,
    pad_array,
    pad_image,
    unpad_array,
)


class UNetSpatialTests(unittest.TestCase):
    def test_padding_preserves_content_at_its_expected_position(self):
        image = Image.new("RGB", CROPPED_IMAGE_SIZE, (12, 34, 56))
        padded = np.asarray(pad_image(image))
        x, y = CONTENT_OFFSET

        self.assertEqual(padded.shape[:2], MODEL_IMAGE_SIZE[::-1])
        self.assertTrue(np.array_equal(padded[y : y + 380, x : x + 510], np.asarray(image)))
        self.assertTrue(np.all(padded[:y] == 0))

    def test_array_round_trip_keeps_mask_coordinates(self):
        mask = np.zeros((2, 380, 510), dtype=np.uint8)
        mask[1, 17, 23] = 1

        padded = pad_array(mask)

        self.assertEqual(padded.shape, (2, 512, 512))
        self.assertEqual(padded[1, 83, 24], 1)
        self.assertTrue(np.array_equal(unpad_array(padded), mask))

    def test_non_cropped_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "510x380"):
            pad_image(Image.new("RGB", (640, 480)))

    def test_unpadding_accepts_a_torch_prediction(self):
        prediction = torch.zeros((1, 512, 512), dtype=torch.bool)
        prediction[0, 66, 1] = True

        result = unpad_array(prediction)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, prediction.dtype)
        self.assertTrue(result[0, 0, 0])

    def test_unpadded_torch_prediction_is_compatible_with_yolo_export(self):
        prediction = torch.zeros((512, 512), dtype=torch.bool)
        prediction[66:70, 1:5] = True

        polygons = mask_to_yolo_polygons(unpad_array(prediction), class_id=0)

        self.assertEqual(len(polygons), 1)
        self.assertTrue(polygons[0].startswith("0 "))


if __name__ == "__main__":
    unittest.main()
