import unittest

import numpy as np

from utils.calcul_volume_fct import (
    calculate_egg_volume_from_instances,
    calculate_eggs_area_from_instances,
    calculate_mean_egg_volume,
    count_egg_instances,
)


class EggVolumeTests(unittest.TestCase):
    @staticmethod
    def label_map(areas):
        labels = np.zeros((10, sum(areas)), dtype=np.int32)
        offset = 0
        for label_id, area in enumerate(areas, start=1):
            labels[0, offset : offset + area] = label_id
            offset += area
        return labels

    def test_uses_twenty_percent_largest_instances(self):
        labels = self.label_map([20, 40, 60, 80, 100])
        surface = calculate_eggs_area_from_instances(labels, echelle=2.0, image_height=10)
        self.assertAlmostEqual(surface, 4.0)

    def test_counts_distinct_and_selected_instances(self):
        for egg_count, expected_selected_count in ((0, 0), (1, 1), (5, 1), (10, 2)):
            with self.subTest(egg_count=egg_count):
                labels = self.label_map([1] * egg_count)
                values = calculate_egg_volume_from_instances(labels, 2.0, 10)
                self.assertEqual(count_egg_instances(labels), egg_count)
                self.assertEqual(values["nombre_oeufs_distincts"], egg_count)
                self.assertEqual(
                    values["nombre_oeufs_utilises_pour_moyenne"],
                    expected_selected_count,
                )

    def test_spherical_volume_and_average_between_images(self):
        first = self.label_map([100])
        second = self.label_map([25])
        first_values = calculate_egg_volume_from_instances(first, 2.0, 10)
        result = calculate_mean_egg_volume(
            [
                {"id_image": "1", "instances": first, "echelle": 2.0, "image_height": 10},
                {"id_image": "2", "instances": second, "echelle": 2.0, "image_height": 10},
            ]
        )

        radius = np.sqrt(4.0 / np.pi)
        expected_first = 4 / 3 * np.pi * radius**3
        self.assertAlmostEqual(first_values["volume_moyen_oeufs_cm3"], expected_first)
        expected_second = expected_first / 8
        self.assertAlmostEqual(
            result["volume_moyen_oeufs_cm3"], (expected_first + expected_second) / 2
        )
        self.assertEqual(result["images"][0]["nombre_oeufs_distincts"], 1)
        self.assertEqual(
            result["images"][0]["nombre_oeufs_utilises_pour_moyenne"], 1
        )

    def test_empty_inputs_are_safe(self):
        empty = np.zeros((8, 8), dtype=np.int32)
        self.assertEqual(calculate_eggs_area_from_instances(empty, 3.1, 480), 0.0)
        self.assertIsNone(calculate_mean_egg_volume([])["volume_moyen_oeufs_cm3"])


if __name__ == "__main__":
    unittest.main()
