from __future__ import annotations

import unittest

from ires_design.trappr import TrapprMeasurement, build_sequence_view, interpret_activity


def measurement(record_id: str, sequence: str, label: int | None, status: str) -> TrapprMeasurement:
    return TrapprMeasurement(
        record_id=record_id,
        source_table="S1",
        source_row=2,
        construct_name="example",
        sequence=sequence,
        sequence_sha256=__import__("hashlib").sha256(sequence.encode()).hexdigest(),
        length=len(sequence),
        alphabet_valid=True,
        assay_context="direct RNA",
        condition="baseline",
        group="",
        ires_type="",
        organism="",
        activity_label=label,
        activity_status=status,
        te_ires=None,
        te_cdi=None,
        fold_difference_from_wt=None,
        position=None,
        is_control=False,
        notes="",
        pmid="",
    )


class TrapprTests(unittest.TestCase):
    def test_qualified_active_calls_are_not_promoted(self) -> None:
        self.assertEqual(interpret_activity("active"), (1, "active"))
        self.assertEqual(interpret_activity("inactive"), (0, "inactive"))
        label, status = interpret_activity("active (does not validate)")
        self.assertIsNone(label)
        self.assertTrue(status.startswith("ambiguous:"))

    def test_sequence_view_flags_label_conflicts(self) -> None:
        rows = [
            measurement("a", "ACGU", 1, "active"),
            measurement("b", "ACGU", 0, "inactive"),
            measurement("c", "GGGG", None, "unlabeled"),
        ]
        view = {row["sequence"]: row for row in build_sequence_view(rows)}
        self.assertEqual(view["ACGU"]["consensus_status"], "conflict")
        self.assertIsNone(view["ACGU"]["consensus_label"])
        self.assertEqual(view["GGGG"]["consensus_status"], "unlabeled_or_ambiguous")


if __name__ == "__main__":
    unittest.main()
