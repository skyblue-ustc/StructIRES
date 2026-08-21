from __future__ import annotations

import random
import unittest

from ires_design.adapters.random_mutation import generate_random_mutants, mutate_sequence


class RandomMutationTests(unittest.TestCase):
    def test_exact_edit_count(self) -> None:
        source = "ACGUACGU"
        mutant, positions = mutate_sequence(source, 3, random.Random(42))
        distance = sum(left != right for left, right in zip(source, mutant))
        self.assertEqual(distance, 3)
        self.assertEqual(len(positions), 3)

    def test_reproducible_generation(self) -> None:
        parents = [("seed", "ACGUACGU")]
        first = generate_random_mutants(parents, num_per_parent=4, n_mutations=2, seed=42)
        second = generate_random_mutants(parents, num_per_parent=4, n_mutations=2, seed=42)
        self.assertEqual([record.sequence for record in first], [record.sequence for record in second])


if __name__ == "__main__":
    unittest.main()

