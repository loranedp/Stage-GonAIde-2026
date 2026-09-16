import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from utils.stats_fct import distribution


class DistributionTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @patch("utils.stats_fct.plt.show")
    def test_historical_call_still_works(self, mock_show):
        distribution([1, 2, 2, 3], bins=3)

        self.assertEqual(len(plt.gca().patches), 3)
        mock_show.assert_called_once_with()

    @patch("utils.stats_fct.plt.show")
    def test_nombre_barres_sets_exact_number_of_bars(self, _mock_show):
        distribution(np.arange(20), nombre_barres=7)

        self.assertEqual(len(plt.gca().patches), 7)

    @patch("utils.stats_fct.plt.show")
    def test_nombre_graduations_x_sets_the_locator(self, _mock_show):
        distribution(np.arange(20), nombre_graduations_x=4)

        locator = plt.gca().xaxis.get_major_locator()
        self.assertIsInstance(locator, MaxNLocator)
        self.assertEqual(locator._nbins, 4)

    @patch("utils.stats_fct.plt.show")
    def test_axe_x_fixe_sets_limits_and_ticks(self, _mock_show):
        ax = distribution(np.arange(20), axe_x_fixe=(0, 10, 2))

        np.testing.assert_allclose(ax.get_xlim(), (0, 10))
        np.testing.assert_allclose(ax.get_xticks(), (0, 2, 4, 6, 8, 10))

    @patch("utils.stats_fct.plt.show")
    def test_axe_x_fixe_does_not_add_an_unaligned_maximum_tick(self, _mock_show):
        ax = distribution(np.arange(20), axe_x_fixe=(0, 10, 3))

        np.testing.assert_allclose(ax.get_xlim(), (0, 10))
        np.testing.assert_allclose(ax.get_xticks(), (0, 3, 6, 9))

    @patch("utils.stats_fct.plt.show")
    def test_axe_x_fixe_supports_decimal_steps(self, _mock_show):
        ax = distribution(np.arange(2), axe_x_fixe=(0, 0.3, 0.1))

        np.testing.assert_allclose(ax.get_xticks(), (0, 0.1, 0.2, 0.3))

    @patch("utils.stats_fct.plt.show")
    def test_axe_x_fixe_ignores_nombre_graduations_x(self, _mock_show):
        ax = distribution(
            np.arange(20),
            nombre_graduations_x=-1,
            axe_x_fixe=(0, 10, 5),
        )

        np.testing.assert_allclose(ax.get_xticks(), (0, 5, 10))

    def test_new_parameters_must_be_positive_integers(self):
        invalid_values = (0, -1, 2.5, True)
        for parameter in ("nombre_graduations_x", "nombre_barres"):
            for value in invalid_values:
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaisesRegex(ValueError, parameter):
                        distribution([1, 2, 3], **{parameter: value})

    def test_bins_and_nombre_barres_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "soit 'bins', soit 'nombre_barres'"):
            distribution([1, 2, 3], bins=2, nombre_barres=3)

    def test_axe_x_fixe_must_have_three_values(self):
        for value in ((0, 10), (0, 10, 2, 4), 10):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "axe_x_fixe"):
                    distribution([1, 2, 3], axe_x_fixe=value)

    def test_axe_x_fixe_values_must_be_valid(self):
        invalid_values = (
            (10, 0, 1),
            (1, 1, 1),
            (0, 10, 0),
            (0, 10, -1),
            (False, 10, 1),
            (0, "10", 1),
            (0, np.inf, 1),
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "axe_x_fixe"):
                    distribution([1, 2, 3], axe_x_fixe=value)


if __name__ == "__main__":
    unittest.main()
