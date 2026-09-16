import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from utils.calcul_volume_fct import (
    calculate_areas,
    calculate_eggs_areas,
    calculate_eggs_volumes,
    calculate_volumes,
)


class PolygonVolumeCoordinatesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / 'image.jpg'
        Image.new('RGB', (640, 480)).save(self.image)
        self.source = self.root / 'source.txt'
        self.crop = self.root / 'crop.txt'
        self.write_polygons(self.source, cropped=False)
        self.write_polygons(self.crop, cropped=True)

    @staticmethod
    def write_polygons(path, cropped):
        lines = []
        for side in (10, 20, 30, 40):
            points = np.array([[100, 100], [100 + side, 100],
                               [100 + side, 100 + side], [100, 100 + side]], dtype=float)
            points = ((points - (85, 33)) / (510, 380) if cropped
                      else points / (640, 480))
            lines.append('0 ' + ' '.join(map(str, points.ravel())))
        path.write_text('\n'.join(lines))

    def test_same_physical_areas_in_both_coordinate_frames(self):
        source = calculate_eggs_areas(self.source, self.image, 3.8)
        crop = calculate_eggs_areas(self.crop, self.image, 3.8, mask_size=(510, 380))
        np.testing.assert_allclose(source[1], crop[1])
        expected = np.array([1600, 900, 400, 100]) * (38 / 480) ** 2
        np.testing.assert_allclose(crop[1], expected)
        self.assertAlmostEqual(crop[0], np.mean(expected[1:3]))
        self.assertAlmostEqual(source[0], crop[0])
        np.testing.assert_allclose(
            source[1], calculate_eggs_areas(self.source, self.image, 3.8, mask_size=(640, 480))[1]
        )
        # Test both cavity and gonad class surfaces in cm².
        for class_id in (0, 1):
            self.write_polygons(self.source, cropped=False)
            self.write_polygons(self.crop, cropped=True)
            for path in (self.source, self.crop):
                path.write_text('\n'.join(
                    f'{class_id} {line.split(" ", 1)[1]}'
                    for line in path.read_text().splitlines()
                ))
            np.testing.assert_allclose(
                calculate_areas(self.source, self.image, 3.8),
                calculate_areas(self.crop, self.image, 3.8, mask_size=(510, 380)),
            )

    def test_empty_predictions(self):
        self.crop.write_text('')
        self.assertEqual(calculate_eggs_areas(self.crop, self.image, 3.8, mask_size=(510, 380)), (0, []))
        self.assertEqual(calculate_areas(self.crop, self.image, 3.8, mask_size=(510, 380)), (0, 0))

    def test_volume_wrappers_propagate_coordinate_frame(self):
        for task in ('oeufs', 'cavite'):
            image_dir = self.root / 'data' / 'images' / task
            image_dir.mkdir(parents=True)
            Image.open(self.image).save(image_dir / 'echo.jpg')
        report_dir = self.root / 'Resultats'
        report_dir.mkdir()
        self.source.rename(self.root / 'source_echo.txt')
        self.crop.rename(self.root / 'crop_echo.txt')
        rows = pd.DataFrame([
            dict(new_name_file='echo', image_id='1', echelle=3.8,
                 type_image='œufs', position=0, long_gonade=3),
        ])
        with contextlib.chdir(report_dir), contextlib.redirect_stdout(io.StringIO()):
            true, _ = calculate_eggs_volumes(rows, 'fish', 'source_')
            pred, _ = calculate_eggs_volumes(rows, 'fish', 'crop_', mask_size=(510, 380))
            np.testing.assert_allclose(true['volume_moyen_oeufs_mm3'], pred['volume_moyen_oeufs_mm3'])
            area = np.mean([900, 400]) * (38 / 480) ** 2
            expected_volume = 4 / 3 * np.pi * (area / np.pi) ** 1.5
            self.assertAlmostEqual(pred['volume_moyen_oeufs_mm3'][0], expected_volume)
            rows = pd.concat([rows, rows], ignore_index=True)
            rows['type_image'] = 'cavite'
            rows['position'] = [1, 2]
            true = calculate_volumes(rows, 'fish', 'source_')
            pred = calculate_volumes(rows, 'fish', 'crop_', mask_size=(510, 380))
            np.testing.assert_allclose(true['volume_cavite'], pred['volume_cavite'])
            self.assertGreater(pred['volume_cavite'][0], 0)


if __name__ == '__main__':
    unittest.main()
