import unittest

import torch

from utils.models_config import metrics_by_class


class SurfaceCalibrationTests(unittest.TestCase):
    @staticmethod
    def _crop_masks():
        true = torch.zeros((1, 1, 380, 510), dtype=torch.bool)
        pred = torch.zeros_like(true)
        true[:, :, 10:30, 10:20] = True       # 200 pixels
        pred[:, :, 10:20, 10:20] = True       # 100 pixels
        return true, pred

    def test_yolo_crop_uses_physical_crop_height(self):
        true, pred = self._crop_masks()
        *_, difference = metrics_by_class(
            true, pred, torch.tensor([[3.8]]), "2classes",
            image_height_px=380,
        )
        # 3.8 cm / 380 px = 0.01 cm/px; difference = 100 px.
        self.assertAlmostEqual(difference.item(), 0.01, places=7)

    def test_unet_padding_has_same_surface_as_yolo_crop(self):
        true, pred = self._crop_masks()
        *_, yolo_difference = metrics_by_class(
            true, pred, torch.tensor([[3.8]]), "2classes",
            image_height_px=380,
        )

        padded_true = torch.zeros((1, 1, 512, 512), dtype=torch.bool)
        padded_pred = torch.zeros_like(padded_true)
        padded_true[:, :, 66:446, 1:511] = true
        padded_pred[:, :, 66:446, 1:511] = pred
        padded_pred[:, :, :10, :10] = True  # Prédiction hors du crop physique.

        *_, unet_difference = metrics_by_class(
            padded_true, padded_pred, torch.tensor([[3.8]]), "2classes",
            image_height_px=380,
        )
        self.assertAlmostEqual(unet_difference.item(), yolo_difference.item(), places=7)

    def test_scale_and_height_are_applied_per_image(self):
        true, pred = self._crop_masks()
        true = true.expand(2, -1, -1, -1).clone()
        pred = pred.expand(2, -1, -1, -1).clone()
        *_, difference = metrics_by_class(
            true, pred, torch.tensor([3.8, 7.6]), "2classes",
            image_height_px=[380, 380],
        )
        torch.testing.assert_close(difference[:, 0], torch.tensor([0.01, 0.04]))

    def test_invalid_physical_height_is_rejected(self):
        true, pred = self._crop_masks()
        with self.assertRaisesRegex(ValueError, "hauteur"):
            metrics_by_class(
                true, pred, torch.tensor([[3.8]]), "2classes",
                image_height_px=0,
            )


if __name__ == "__main__":
    unittest.main()
