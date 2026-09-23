import unittest

from utils.cv_results import normalize_and_validate_cv_results, repair_repeated_leading_folds


def yolo_fold(*names):
    return [{"image": name, "metrics": {}} for name in names]


class CrossValidationResultsTests(unittest.TestCase):
    def test_normalizes_five_integer_folds(self):
        results = {index: yolo_fold(f"image_{index}.jpg") for index in range(1, 6)}
        results["fold_times"] = {index: 1.0 for index in range(1, 6)}

        normalized = normalize_and_validate_cv_results(
            results, expected_images=[f"image_{index}.jpg" for index in range(1, 6)]
        )

        self.assertEqual(list(normalized)[:5], [f"fold_{index}" for index in range(1, 6)])
        self.assertIn("fold_times", normalized)

    def test_rejects_seven_accumulated_folds(self):
        results = {index: yolo_fold(f"image_{index}.jpg") for index in range(1, 8)}

        with self.assertRaisesRegex(ValueError, "Folds invalides"):
            normalize_and_validate_cv_results(results)

    def test_rejects_image_shared_by_two_folds(self):
        results = {index: yolo_fold(f"image_{index}.jpg") for index in range(1, 6)}
        results[5] = yolo_fold("image_1.jpg")

        with self.assertRaisesRegex(ValueError, "plusieurs folds"):
            normalize_and_validate_cv_results(results)

    def test_rejects_missing_expected_image(self):
        results = {index: yolo_fold(f"image_{index}.jpg") for index in range(1, 6)}

        with self.assertRaisesRegex(ValueError, r"1 image\(s\) manquante"):
            normalize_and_validate_cv_results(results, expected_images=[
                "image_1.jpg", "image_2.jpg", "image_3.jpg",
                "image_4.jpg", "image_5.jpg", "image_6.jpg",
            ])

    def test_repairs_known_repeated_leading_fold_artifact(self):
        results = {
            1: yolo_fold("first.jpg"),
            2: yolo_fold("first.jpg"),
            3: yolo_fold("first.jpg"),
            4: yolo_fold("second.jpg"),
            5: yolo_fold("third.jpg"),
            6: yolo_fold("fourth.jpg"),
            7: yolo_fold("fifth.jpg"),
        }

        repaired, changed = repair_repeated_leading_folds(results)
        normalized = normalize_and_validate_cv_results(repaired)

        self.assertTrue(changed)
        self.assertEqual(normalized["fold_1"][0]["image"], "first.jpg")

    def test_does_not_guess_when_extra_folds_are_different(self):
        results = {index: yolo_fold(f"image_{index}.jpg") for index in range(1, 8)}

        repaired, changed = repair_repeated_leading_folds(results)

        self.assertFalse(changed)
        self.assertIs(repaired, results)


if __name__ == "__main__":
    unittest.main()
