from __future__ import annotations

import unittest

from ires_design.assay_audit import _local_alignment_stats, summarize_cross_assay


class AssayAuditTests(unittest.TestCase):
    def test_local_alignment_detects_contained_fragment(self) -> None:
        identity, coverage, identities = _local_alignment_stats("AAAAACGUACGUUUUU", "ACGUACGU")
        self.assertAlmostEqual(identity, 1.0)
        self.assertAlmostEqual(coverage, 1.0)
        self.assertEqual(identities, 8)

    def test_summary_separates_direct_statuses(self) -> None:
        rows = [
            {
                "direct_consensus_status": "active",
                "exact_sequence_match": True,
                "legacy_contained_in_direct": True,
                "high_similarity_90_80": True,
                "legacy_label": 1,
            },
            {
                "direct_consensus_status": "inactive",
                "exact_sequence_match": False,
                "legacy_contained_in_direct": False,
                "high_similarity_90_80": False,
                "legacy_label": 0,
            },
        ]
        summary = summarize_cross_assay(rows)
        self.assertEqual(summary["n_direct_labelled"], 2)
        self.assertEqual(summary["overlap_by_direct_status"]["active"]["exact"], 1)


if __name__ == "__main__":
    unittest.main()
