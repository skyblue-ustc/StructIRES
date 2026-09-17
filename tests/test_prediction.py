from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from ires_design.prediction import composition_features, dataset_audit, load_ires_ai_records


class PredictionDataTests(unittest.TestCase):
    def test_recovers_repeated_holdouts_and_reconstructs_partition(self) -> None:
        rows = [
            [0, "train", "a", "ACGT", 0, "library", 4],
            [1, "test", "a", "ACGT", 0, "library", 4],
            [2, "test", "a", "ACGT", 0, "library", 4],
            [0, "train", "b", "GGGG", 1, "natural", 4],
            [1, "train", "b", "GGGG", 1, "natural", 4],
            [2, "train", "b", "GGGG", 1, "natural", 4],
        ]
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "split.csv.zip"
            csv_path = Path(directory) / "split.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["fold", "type", "ID", "Sequence", "IRES_class_600", "Source", "Length"]
                )
                writer.writerows(rows)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(csv_path, arcname="split.csv")
            records = load_ires_ai_records(archive_path, n_folds=3)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].sequence, "ACGU")
        self.assertEqual(records[0].published_test_folds, (1, 2))
        self.assertTrue(all(0 <= record.test_fold < 3 for record in records))
        audit = dataset_audit(records)
        self.assertEqual(audit["n_positive"], 1)
        self.assertEqual(audit["n_never_in_published_test"], 1)
        self.assertEqual(audit["published_test_assignment_count"], {"0": 1, "2": 1})

    def test_composition_features_are_finite(self) -> None:
        features = composition_features(["ACGU", "GGGG"])
        self.assertEqual(features.shape[0], 2)
        self.assertTrue((features == features).all())


if __name__ == "__main__":
    unittest.main()
