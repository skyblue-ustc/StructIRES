from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_locked_native_models.py"
    )
    spec = importlib.util.spec_from_file_location("locked_native_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LockedNativeComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_exact_sign_flip_detects_consistent_fold_gain(self) -> None:
        delta = np.ones(10, dtype=np.float64)
        self.assertAlmostEqual(self.module.exact_sign_flip_pvalue(delta), 2 / 1024)

    def test_paired_bootstrap_preserves_model_pairing(self) -> None:
        label = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        structure = np.asarray([.05, .10, .15, .20, .80, .85, .90, .95])
        baseline = np.asarray([.90, .70, .60, .40, .30, .20, .10, .05])
        report = self.module.paired_stratified_bootstrap(
            label, structure, baseline, replicates=100, seed=1337
        )
        self.assertGreater(report["auc"]["lower_95"], 0)
        self.assertGreater(report["aupr"]["lower_95"], 0)


if __name__ == "__main__":
    unittest.main()
