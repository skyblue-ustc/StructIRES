from __future__ import annotations

import unittest

from ires_design.design import make_mutation_pool, max_allowed_edits


class SeededPoolTests(unittest.TestCase):
    def test_pool_is_deterministic_unique_and_bounded(self) -> None:
        args = dict(
            experiment_id="test",
            run_seed=42,
            candidates_per_parent=12,
            max_edit_distance=10,
            max_edit_fraction=0.03,
        )
        parents = [("a", "A" * 100), ("b", "C" * 200)]
        first = make_mutation_pool(parents, **args)
        second = make_mutation_pool(parents, **args)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(len({record.sequence for record in first if record.parent_id == "a"}), 12)
        for record in first:
            parent = dict(parents)[record.parent_id]
            edits = sum(a != b for a, b in zip(parent, record.sequence))
            self.assertGreaterEqual(edits, 1)
            self.assertLessEqual(edits, max_allowed_edits(parent, 10, 0.03))


if __name__ == "__main__":
    unittest.main()
