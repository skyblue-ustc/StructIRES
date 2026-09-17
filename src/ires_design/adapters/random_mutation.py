"""Uniform mutation baseline with explicit parent and edit provenance."""

from __future__ import annotations

import random
from collections.abc import Iterable

from ..schemas import CandidateRecord, normalize_rna


RNA_ALPHABET = "ACGU"


def mutate_sequence(sequence: str, n_mutations: int, rng: random.Random) -> tuple[str, list[int]]:
    sequence = normalize_rna(sequence)
    if not sequence or set(sequence) - set(RNA_ALPHABET):
        raise ValueError("sequence must contain only A/C/G/U or A/C/G/T")
    if n_mutations < 1 or n_mutations > len(sequence):
        raise ValueError("n_mutations must be between 1 and sequence length")
    positions = sorted(rng.sample(range(len(sequence)), n_mutations))
    chars = list(sequence)
    for position in positions:
        choices = [base for base in RNA_ALPHABET if base != chars[position]]
        chars[position] = rng.choice(choices)
    return "".join(chars), positions


def generate_random_mutants(
    parents: Iterable[tuple[str, str]],
    *,
    num_per_parent: int,
    n_mutations: int,
    seed: int,
) -> list[CandidateRecord]:
    rng = random.Random(seed)
    records: list[CandidateRecord] = []
    for parent_id, sequence in parents:
        for replicate in range(num_per_parent):
            mutant, positions = mutate_sequence(sequence, n_mutations, rng)
            records.append(
                CandidateRecord(
                    candidate_id=f"random_mutation__s{seed}__{parent_id}__{replicate:04d}",
                    sequence=mutant,
                    method="random_mutation",
                    task="mutation",
                    seed=seed,
                    parent_id=parent_id,
                    metadata={"n_mutations": n_mutations, "mutated_positions_0based": positions},
                )
            )
    return records

