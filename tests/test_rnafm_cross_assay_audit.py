from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_rnafm_cross_assay_heterogeneity.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("audit_rnafm_cross_assay_heterogeneity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RNAFMCrossAssayAuditTests(unittest.TestCase):
    def test_probability_summary_preserves_label_views(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=int)
        probabilities = np.asarray([0.1, 0.3, 0.7, 0.9], dtype=float)
        summary = MODULE.probability_summary(labels, probabilities)
        self.assertAlmostEqual(summary["probability_mean"], 0.5)
        self.assertAlmostEqual(summary["probability_mean_negative"], 0.2)
        self.assertAlmostEqual(summary["probability_mean_positive"], 0.8)

    def test_fold_column_pattern_is_strict(self) -> None:
        self.assertIsNotNone(MODULE.FOLD_COLUMN.fullmatch("probability_fold9"))
        self.assertIsNone(MODULE.FOLD_COLUMN.fullmatch("probability_fold9_extra"))


if __name__ == "__main__":
    unittest.main()
