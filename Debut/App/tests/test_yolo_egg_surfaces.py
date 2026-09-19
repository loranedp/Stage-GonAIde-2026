import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from utils.models_config import metrics_by_class
from utils.yolo_metrics import load_yolo_polygon_areas, result_to_instance_areas


class YoloEggSurfaceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'eggs.txt'
        # Two touching squares, one overlapping square, one small square.
        self.path.write_text(
            '0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n'
            '0 0.2 0.1 0.3 0.1 0.3 0.2 0.2 0.2\n'
            '0 0.15 0.1 0.25 0.1 0.25 0.2 0.15 0.2\n'
            '0 0.4 0.1 0.42 0.1 0.42 0.12 0.4 0.12\n'
            '1 0 0 0.9 0 0.9 0.9 0 0.9\n'
        )
        self.result = SimpleNamespace(
            orig_shape=(100, 100), names={0: 'Oeuf', 1: 'Gonade'},
            boxes=SimpleNamespace(cls=torch.tensor([0, 0, 1])),
            masks=SimpleNamespace(xy=[
                np.array([[10, 10], [30, 10], [30, 30], [10, 30]]),
                np.array([[20, 10], [40, 10], [40, 30], [20, 30]]),
                np.array([[0, 0], [90, 0], [90, 90], [0, 90]]),
            ]),
        )

    def test_one_area_per_label_line_and_detection(self):
        self.assertEqual(load_yolo_polygon_areas(self.path, (100, 100)), [121, 121, 121, 9])
        self.assertEqual(result_to_instance_areas(self.result, ['Oeuf', 'Gonade']), [441, 441])
        self.result.masks = None
        self.assertEqual(result_to_instance_areas(self.result, ['Oeuf', 'Gonade']), [])

    def notebook_metrics(self, dataset):
        notebook = Path(__file__).resolve().parents[2] / 'YOLO26' / 'YOLO.ipynb'
        cells = json.loads(notebook.read_text())['cells']
        source = next(''.join(c['source']) for c in cells if 'def compute_dataset_metrics(' in ''.join(c.get('source', [])))
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'compute_dataset_metrics')
        environment = dict(
            torch=torch, metrics_by_class=metrics_by_class,
            load_yolo_polygon_areas=load_yolo_polygon_areas,
            result_to_instance_areas=result_to_instance_areas,
            DATASET_NAME=dataset,
            dataset_config={'egg_class_index': 0}, names=['Oeuf', 'Gonade'],
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), 'exec'), environment)
        return environment['compute_dataset_metrics']

    def test_instance_metric_uses_individual_areas_and_original_height(self):
        for dataset, classes in [('oeufs', 1), ('oeufsclasses', 2)]:
            masks = torch.zeros((1, classes, 100, 100), dtype=torch.bool)
            with self.subTest(dataset=dataset), patch('utils.models_config.split_eggs', side_effect=AssertionError('Must not split instances')):
                *_, difference = self.notebook_metrics(dataset)(
                    masks, masks, torch.tensor([[1.0]]), self.path, self.result,
                )
                self.assertAlmostEqual(difference[0, 0].item(), 3.2, places=5)
                if classes == 2:
                    self.assertTrue(torch.isnan(difference[0, 1]))

    def test_empty_surface_lists_produce_nan(self):
        masks = torch.zeros((2, 1, 100, 100), dtype=torch.bool)
        *_, difference = metrics_by_class(
            masks, masks, torch.tensor([1., 1.]), 'oeufs',
            egg_annotation_areas_px=[[], [121]], egg_prediction_areas_px=[[441], []],
        )
        self.assertTrue(torch.isnan(difference).all())


if __name__ == '__main__':
    unittest.main()
