import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image

from utils.dataset_unet import EggHVDataset, RoboflowUNetDataset
from utils.models_config import collate_fn, collate_fn_egg_hv, metrics_by_class


class EggAnnotationSurfaceTests(unittest.TestCase):
    def test_datasets_preserve_touching_overlapping_and_small_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'egg.png'
            Image.new('RGB', (510, 380)).save(path)
            # Two touching squares, an overlapping square and a small egg.
            polygons = [
                [10, 10, 20, 10, 20, 20, 10, 20],
                [20, 10, 30, 10, 30, 20, 20, 20],
                [15, 10, 25, 10, 25, 20, 15, 20],
                [40, 10, 43, 10, 43, 13, 40, 13],
            ]
            sample = {'image_path': str(path), 'annotations': [
                {'category_id': 2, 'segmentation': [polygon]} for polygon in polygons
            ]}
            datasets = [
                (RoboflowUNetDataset(
                    [sample], directory, pd.DataFrame({'categorie': ['A']}),
                    (512, 512), num_classes=1, class_ids=[2],
                ), collate_fn),
                (EggHVDataset([sample], (512, 512), egg_category_id=2), collate_fn_egg_hv),
            ]
            for dataset, collate in datasets:
                with self.subTest(dataset=type(dataset).__name__):
                    item = dataset[0]
                    self.assertEqual(item['egg_annotation_areas_px'], [100, 100, 100, 9])
                    self.assertEqual(item['original_size'], (510, 380))
                    self.assertEqual(item['mask'].shape[-2:], (512, 512))
                    batch = collate([item, item])
                    self.assertEqual(batch['egg_annotation_areas_px'], [[100, 100, 100, 9]] * 2)
                    dataset.is_train = True
                    with patch('utils.dataset_unet.data_augmentation', side_effect=lambda image, masks, **kwargs: (image, masks)):
                        self.assertIsNone(dataset[0]['egg_annotation_areas_px'])

    def test_surface_conversion_is_invariant_to_padding_and_ignores_margins(self):
        labels = torch.zeros((1, 380, 510), dtype=torch.int32)
        labels[0, 10:20, 10:30] = 1  # 200 pixels
        labels[0, 30:40, 10:50] = 2  # 400 pixels; median = 300
        padded = torch.zeros((1, 512, 512), dtype=torch.int32)
        padded[:, 66:446, 1:511] = labels
        padded[0, :10, :10] = 3  # False egg entirely in the padding
        scale = torch.tensor([[3.8]])  # 0.1 mm / pixel at height 380
        for instances in (labels, padded):
            masks = (instances > 0).unsqueeze(1)
            with patch('utils.models_config.split_eggs', side_effect=AssertionError('Annotations must not be split')):
                *_, difference = metrics_by_class(
                    masks, masks, scale, 'oeufs', egg_instance_labels=instances,
                    egg_annotation_areas_px=[[100, 200]], image_height_px=[380],
                )
            self.assertAlmostEqual(difference.item(), 1.5, places=5)

    def test_empty_annotations_or_predictions_produce_nan(self):
        labels = torch.zeros((2, 380, 510), dtype=torch.int32)
        labels[0, 10:20, 10:20] = 1
        masks = (labels > 0).unsqueeze(1)
        *_, difference = metrics_by_class(
            masks, masks, torch.tensor([3.8, 3.8]), 'oeufs',
            egg_instance_labels=labels, egg_annotation_areas_px=[[], [100]],
            image_height_px=[380, 380],
        )
        self.assertEqual(difference.shape, (2, 1))
        self.assertTrue(torch.isnan(difference).all())


if __name__ == '__main__':
    unittest.main()
