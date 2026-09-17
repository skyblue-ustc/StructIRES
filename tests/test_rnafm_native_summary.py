from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_rnafm_native_folds.py"
SPEC = importlib.util.spec_from_file_location("summarize_rnafm_native_folds", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RNAFMNativeSummaryTests(unittest.TestCase):
    def test_perfect_fixed_threshold_metrics(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=int)
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=float)
        metrics = MODULE.compute_metrics(labels, probabilities)
        for name in ("auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc"):
            with self.subTest(metric=name):
                self.assertAlmostEqual(metrics[name], 1.0)
        self.assertAlmostEqual(metrics["ece10"], 0.15)

    def test_stratified_bootstrap_is_reproducible(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
        probabilities = np.asarray([0.1, 0.3, 0.4, 0.6, 0.7, 0.9], dtype=float)
        first = MODULE.stratified_bootstrap(labels, probabilities, replicates=20, seed=42)
        second = MODULE.stratified_bootstrap(labels, probabilities, replicates=20, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {
                f"{metric}_{bound}"
                for metric in MODULE.METRIC_NAMES
                for bound in ("ci_low", "ci_high")
            },
        )


if __name__ == "__main__":
    unittest.main()
