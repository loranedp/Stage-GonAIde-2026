import ast
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


def load_crop_images_yolo():
    source_path = Path(__file__).resolve().parents[2] / "utils" / "pretraitement.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "crop_images_YOLO"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"Path": Path, "cv2": cv2, "json": json}
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["crop_images_YOLO"]


class CropImagesYoloTests(unittest.TestCase):
    def test_pairs_are_matched_by_stem_and_second_run_is_unchanged(self):
        crop_images_yolo = load_crop_images_yolo()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first_image = directory / "first.jpg"
            second_image = directory / "second.jpg"
            first_mask = directory / "first.txt"
            second_mask = directory / "second.txt"
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(first_image), image))
            self.assertTrue(cv2.imwrite(str(second_image), image))
            first_mask.write_text("0 0.2890625 0.27708333333333335\n", encoding="utf-8")
            second_mask.write_text("1 0.4453125 0.48541666666666666\n", encoding="utf-8")

            crop_images_yolo(
                [str(second_image), str(first_image)],
                [str(first_mask), str(second_mask)],
            )

            self.assertEqual(first_mask.read_text(encoding="utf-8").split()[0], "0")
            self.assertEqual(second_mask.read_text(encoding="utf-8").split()[0], "1")
            image_bytes = first_image.read_bytes()
            mask_content = first_mask.read_text(encoding="utf-8")

            crop_images_yolo([str(first_image)], [str(first_mask)])

            self.assertEqual(first_image.read_bytes(), image_bytes)
            self.assertEqual(first_mask.read_text(encoding="utf-8"), mask_content)

    def test_missing_mask_stops_before_image_is_modified(self):
        crop_images_yolo = load_crop_images_yolo()
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            original_image = image_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "Images sans masque"):
                crop_images_yolo([str(image_path)], [])

            self.assertEqual(image_path.read_bytes(), original_image)

    def test_coco_annotations_are_translated_once_with_the_image(self):
        crop_images_yolo = load_crop_images_yolo()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image_path = directory / "image.jpg"
            mask_path = directory / "image.txt"
            coco_path = directory / "annotations.json"
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            mask_path.write_text("0 0.5 0.5\n", encoding="utf-8")
            coco_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "image.jpg", "width": 640, "height": 480}
                        ],
                        "annotations": [
                            {
                                "id": 2,
                                "image_id": 1,
                                "bbox": [100, 50, 20, 30],
                                "segmentation": [[100, 50, 120, 50, 120, 80, 100, 80]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            crop_images_yolo([str(image_path)], [str(mask_path)], coco_path)

            cropped = json.loads(coco_path.read_text(encoding="utf-8"))
            self.assertEqual(cropped["images"][0]["width"], 510)
            self.assertEqual(cropped["images"][0]["height"], 380)
            annotation = cropped["annotations"][0]
            self.assertEqual(annotation["bbox"], [15, 17, 20, 30])
            self.assertEqual(annotation["segmentation"], [[15, 17, 35, 17, 35, 47, 15, 47]])
            first_content = coco_path.read_text(encoding="utf-8")

            crop_images_yolo([str(image_path)], [str(mask_path)], coco_path)

            self.assertEqual(coco_path.read_text(encoding="utf-8"), first_content)
