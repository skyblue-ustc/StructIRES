from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_published_ireslm_release.py"
SPEC = importlib.util.spec_from_file_location("audit_published_ireslm_release", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PublishedReleaseAuditTest(unittest.TestCase):
    def test_recomputes_released_mean_and_sample_std(self) -> None:
        columns = ["", *MODULE.METRIC_COLUMNS.values()]
        fold_rows = []
        for fold in range(3):
            fold_rows.append({"": str(fold), **{column: str(fold + 0.1) for column in MODULE.METRIC_COLUMNS.values()}})
        mean = {"": "mean", **{column: "1.1" for column in MODULE.METRIC_COLUMNS.values()}}
        std = {"": "std", **{column: "1.0" for column in MODULE.METRIC_COLUMNS.values()}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows([*fold_rows, mean, std])
            folds, means = MODULE.read_released_metrics(path, 3)
        self.assertEqual(len(folds), 3)
        self.assertAlmostEqual(means["auc"], 1.1)

    def test_detects_test_aupr_checkpoint_selection(self) -> None:
        source = """
test_metrics, test_loss, test_result = predict_step(test_dataloader, model)
if test_metrics['AUPR'].values[0] > best_aupr:
    pass
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.py"
            path.write_text(source, encoding="utf-8")
            result = MODULE.selection_audit(path)
        self.assertTrue(result["test_evaluated_each_training_epoch"])
        self.assertTrue(result["checkpoint_selected_on_test_aupr"])
        self.assertFalse(result["independent_blind_test"])


if __name__ == "__main__":
    unittest.main()
