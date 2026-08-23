from __future__ import annotations

import unittest

from ires_design.classification import (
    binary_classification_metrics,
    expected_calibration_error,
    stratified_bootstrap_intervals,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        for name in ("auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc"):
            self.assertAlmostEqual(metrics[name], 1.0)
        self.assertAlmostEqual(metrics["ece10"], 0.15)

    def test_ece_includes_probability_one(self) -> None:
        self.assertEqual(expected_calibration_error([1], [1.0]), 0.0)

    def test_bootstrap_is_reproducible(self) -> None:
        first = stratified_bootstrap_intervals([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 20, 7)
        second = stratified_bootstrap_intervals([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 20, 7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
