from __future__ import annotations

import unittest

from ires_design.structure import (
    FoldResult,
    base_pair_distance,
    context_structure_consistency,
    dot_bracket_pairs,
    ensemble_anchor_pairs,
    ires_crosstalk_ratio,
    paired_fraction,
    pairing_profile_distance,
    structure_state_identity,
    weighted_anchor_retention,
)


def folded(sequence: str, paired: tuple[float, ...], pairs=()) -> FoldResult:
    return FoldResult(
        sequence=sequence,
        mfe_structure="." * len(sequence),
        mfe_kcal_mol=-1.0,
        ensemble_free_energy_kcal_mol=-1.2,
        centroid_structure="." * len(sequence),
        centroid_distance=0.0,
        ensemble_diversity=0.0,
        paired_probabilities=paired,
        base_pair_probabilities=tuple(pairs),
        viennarna_version="test",
    )


class StructureMetricTests(unittest.TestCase):
    def test_dot_bracket_and_base_pair_distance(self) -> None:
        self.assertEqual(dot_bracket_pairs("((..))"), frozenset({(0, 5), (1, 4)}))
        self.assertEqual(base_pair_distance("((..))", ".(..)."), 1)
        self.assertAlmostEqual(paired_fraction("((..))"), 4 / 6)

    def test_state_identity_and_profile_distance(self) -> None:
        self.assertAlmostEqual(structure_state_identity("((..))", ".(..)."), 4 / 6)
        self.assertAlmostEqual(pairing_profile_distance((0.0, 1.0), (0.2, 0.6)), 0.3)

    def test_context_metrics(self) -> None:
        reference = folded("ACGU", (0.1, 0.5, 0.2, 0.8))
        context = folded(
            "ACGUAA",
            (0.2, 0.4, 0.2, 0.7, 0.3, 0.3),
            ((0, 5, 0.3), (1, 4, 0.2), (2, 3, 0.5)),
        )
        self.assertAlmostEqual(ires_crosstalk_ratio(context, 4), 0.125)
        self.assertAlmostEqual(context_structure_consistency(reference, context), 0.925)

    def test_parent_ensemble_anchor_retention(self) -> None:
        parent = folded(
            "ACGU",
            (0.9, 0.9, 0.8, 0.8),
            ((0, 3, 0.90), (1, 2, 0.80), (0, 2, 0.10)),
        )
        anchors = ensemble_anchor_pairs(parent, min_probability=0.5)
        self.assertEqual(anchors, ((0, 3, 0.90), (1, 2, 0.80)))
        self.assertAlmostEqual(
            weighted_anchor_retention(anchors, ((0, 3, 0.90), (1, 2, 0.40))),
            1.30 / 1.70,
        )
        self.assertTrue(str(weighted_anchor_retention((), ())).lower() == "nan")

    def test_unbalanced_structure_fails(self) -> None:
        with self.assertRaises(ValueError):
            dot_bracket_pairs("((.)")


if __name__ == "__main__":
    unittest.main()
