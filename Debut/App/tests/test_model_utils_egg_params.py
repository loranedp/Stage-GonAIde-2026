from types import SimpleNamespace
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import model_utils
import config


class EggSeparationParameterTests(unittest.TestCase):
    def test_split_helper_exposes_the_configured_defaults(self):
        split_parameters = inspect.signature(model_utils.split_egg_mask).parameters

        self.assertEqual(split_parameters["min_distance"].default, config.EGG_MIN_DISTANCE)
        self.assertEqual(split_parameters["min_area"].default, config.EGG_MIN_AREA)

    def test_split_egg_mask_forwards_custom_parameters(self):
        labels = torch.zeros((1, 8, 10), dtype=torch.int32)
        with patch("model_utils.split_eggs", return_value=labels) as mocked_split:
            result = model_utils.split_egg_mask(
                np.zeros((8, 10), dtype=bool), min_distance=3, min_area=40
            )

        self.assertEqual(result.shape, (8, 10))
        _, kwargs = mocked_split.call_args
        self.assertEqual(kwargs, {"min_distance": 3, "min_area": 40})

    def test_predict_eggs_preserves_yolo_instances_in_source_coordinates(self):
        detection_mask = torch.zeros((1, 3, 4), dtype=torch.float32)
        detection_mask[0, 1, 2] = 1
        detection = SimpleNamespace(
            masks=SimpleNamespace(data=detection_mask),
            boxes=SimpleNamespace(conf=torch.tensor([0.9])),
        )
        model = SimpleNamespace(predict=lambda **_kwargs: [detection])
        image = Image.new("RGB", (8, 6))

        with patch.object(config, "CROP_PARAMS", [1, 1, 4, 3]):
            result = model_utils.predict_eggs(model, image)

        self.assertEqual(result["instances"].shape, (6, 8))
        self.assertEqual(result["instances"][2, 3], 1)
        self.assertAlmostEqual(result["confidences"]["Oeuf"], 0.9)
        self.assertEqual(int(result["masks"]["Oeuf"].sum()), 1)

    def test_predict_eggs_returns_empty_masks_when_yolo_finds_no_instance(self):
        model = SimpleNamespace(
            predict=lambda **_kwargs: [SimpleNamespace(masks=None, boxes=None)]
        )
        with patch.object(config, "CROP_PARAMS", [1, 1, 4, 3]):
            result = model_utils.predict_eggs(model, Image.new("RGB", (8, 6)))

        self.assertFalse(result["masks"]["Oeuf"].any())
        self.assertFalse(result["instances"].any())
        self.assertIsNone(result["confidences"]["Oeuf"])


if __name__ == "__main__":
    unittest.main()
