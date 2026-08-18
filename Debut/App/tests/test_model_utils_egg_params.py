import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import model_utils
import config


class EggSeparationParameterTests(unittest.TestCase):
    def test_public_helpers_expose_the_new_defaults(self):
        predict_parameters = inspect.signature(model_utils.predict_eggs).parameters
        split_parameters = inspect.signature(model_utils.split_egg_mask).parameters

        self.assertEqual(predict_parameters["min_distance"].default, config.EGG_MIN_DISTANCE)
        self.assertEqual(predict_parameters["min_area"].default, config.EGG_MIN_AREA)
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

    def test_predict_eggs_forwards_custom_parameters(self):
        model = torch.nn.Conv2d(3, 1, kernel_size=1)

        def empty_instances(tensor, **_kwargs):
            return torch.zeros(
                (tensor.shape[0], tensor.shape[2], tensor.shape[3]),
                dtype=torch.int32,
                device=tensor.device,
            )

        with patch("model_utils.split_eggs", side_effect=empty_instances) as mocked_split:
            model_utils.predict_eggs(
                [model],
                Image.new("RGB", (640, 480)),
                min_distance=4,
                min_area=75,
            )

        _, kwargs = mocked_split.call_args
        self.assertEqual(kwargs, {"min_distance": 4, "min_area": 75})


if __name__ == "__main__":
    unittest.main()
