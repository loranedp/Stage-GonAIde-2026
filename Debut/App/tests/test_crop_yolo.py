from pathlib import Path

import cv2
import numpy as np

from utils.crop_YOLO import crop_YOLO, crop_image


def test_crop_image_returns_final_size_image_unchanged():
    image = np.zeros((380, 510, 3), dtype=np.uint8)

    result = crop_image(image)

    assert result is image


def test_crop_yolo_crops_image_and_updates_annotation(tmp_path: Path):
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "image.txt"
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[133, 185] = (255, 255, 255)
    assert cv2.imwrite(str(image_path), image)
    mask_path.write_text(
        "2 0.2890625 0.27708333333333335 0.4453125 0.48541666666666666\n",
        encoding="utf-8",
    )

    assert crop_YOLO(image_path, mask_path)

    cropped_image = cv2.imread(str(image_path))
    assert cropped_image.shape[:2] == (380, 510)
    class_id, *coordinates = mask_path.read_text(encoding="utf-8").split()
    assert class_id == "2"
    assert np.allclose(
        list(map(float, coordinates)),
        [100 / 510, 100 / 380, 200 / 510, 200 / 380],
    )


def test_crop_yolo_does_not_rewrite_already_cropped_files(tmp_path: Path):
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "image.txt"
    image = np.zeros((380, 510, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    original_mask = "0 0.1 0.1 0.2 0.2\n"
    mask_path.write_text(original_mask, encoding="utf-8")
    original_image = image_path.read_bytes()

    assert not crop_YOLO(image_path, mask_path)

    assert image_path.read_bytes() == original_image
    assert mask_path.read_text(encoding="utf-8") == original_mask
