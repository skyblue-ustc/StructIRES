from __future__ import annotations

import unittest

from ires_design.metrics import longest_homopolymer, sequence_metrics, summarize_sequences
from ires_design.schemas import CandidateRecord


class SequenceMetricTests(unittest.TestCase):
    def test_candidate_normalizes_dna_to_rna(self) -> None:
        record = CandidateRecord(
            candidate_id="x",
            sequence="ATCG",
            method="test",
            task="de_novo",
            seed=42,
        )
        self.assertEqual(record.sequence, "AUCG")

    def test_homopolymer(self) -> None:
        self.assertEqual(longest_homopolymer("ACCCGU"), 3)
        self.assertEqual(longest_homopolymer(""), 0)

    def test_expected_length_gate(self) -> None:
        valid = sequence_metrics("ACGU", expected_length=4)
        invalid = sequence_metrics("ACGN", expected_length=4)
        self.assertTrue(valid["valid"])
        self.assertFalse(invalid["valid"])

    def test_summary_reports_survival_and_uniqueness(self) -> None:
        summary = summarize_sequences(["ACGU", "ACGU", "AAAA", "ACG"], expected_length=4)
        self.assertEqual(summary["n_raw"], 4)
        self.assertEqual(summary["n_valid"], 3)
        self.assertAlmostEqual(summary["survival_rate_pct"], 75.0)
        self.assertAlmostEqual(summary["unique_rate_pct"], 75.0)


if __name__ == "__main__":
    unittest.main()

