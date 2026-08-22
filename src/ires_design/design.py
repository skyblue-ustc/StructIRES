"""Deterministic, budget-matched seeded mutation pools for IRES design.

This module deliberately separates *proposal* from scoring/selection.  Every
method receives byte-identical candidate pools; a later objective may rank or
filter them but may never change parents, edit limits, or candidate budget.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from .adapters.random_mutation import RNA_ALPHABET
from .schemas import CandidateRecord, normalize_rna


def stable_seed(*parts: object) -> int:
    """Derive a platform-independent RNG seed from protocol identifiers."""
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def max_allowed_edits(sequence: str, max_edit_distance: int, max_edit_fraction: float) -> int:
    sequence = normalize_rna(sequence)
    if not sequence:
        raise ValueError("seed sequence must be non-empty")
    if max_edit_distance < 1 or not 0.0 < max_edit_fraction <= 1.0:
        raise ValueError("invalid edit-limit contract")
    return max(1, min(max_edit_distance, int(len(sequence) * max_edit_fraction)))


def make_mutation_pool(
    parents: Iterable[tuple[str, str]],
    *,
    experiment_id: str,
    run_seed: int,
    candidates_per_parent: int,
    max_edit_distance: int,
    max_edit_fraction: float,
) -> list[CandidateRecord]:
    """Generate a deterministic, unique substitution pool per parent.

    Edit counts cycle uniformly from one through the protocol maximum, avoiding
    a hidden preference for one-edit candidates.  Duplicates are rejected and
    resampled deterministically; this makes the pool a shared experimental
    asset rather than an optimizer-specific implementation detail.
    """
    if candidates_per_parent < 1:
        raise ValueError("candidates_per_parent must be positive")
    records: list[CandidateRecord] = []
    for parent_id, raw_sequence in parents:
        sequence = normalize_rna(raw_sequence)
        if not sequence or set(sequence) - set(RNA_ALPHABET):
            raise ValueError(f"parent {parent_id!r} is not canonical RNA")
        limit = max_allowed_edits(sequence, max_edit_distance, max_edit_fraction)
        rng = random.Random(stable_seed(experiment_id, run_seed, parent_id, sequence))
        observed = {sequence}
        for index in range(candidates_per_parent):
            n_edits = (index % limit) + 1
            for _attempt in range(10_000):
                positions = sorted(rng.sample(range(len(sequence)), n_edits))
                mutant = list(sequence)
                for position in positions:
                    mutant[position] = rng.choice(
                        [base for base in RNA_ALPHABET if base != mutant[position]]
                    )
                candidate = "".join(mutant)
                if candidate not in observed:
                    observed.add(candidate)
                    break
            else:  # pragma: no cover - impossible for the frozen sequence sizes
                raise RuntimeError(f"could not create unique mutant for parent {parent_id}")
            records.append(
                CandidateRecord(
                    candidate_id=f"pool__s{run_seed}__{parent_id}__{index:05d}",
                    sequence=candidate,
                    method="shared_mutation_pool",
                    task="mutation",
                    seed=run_seed,
                    parent_id=parent_id,
                    metadata={
                        "proposal": "uniform_substitution_pool",
                        "n_mutations": n_edits,
                        "mutated_positions_0based": positions,
                        "max_allowed_edits": limit,
                    },
                )
            )
    return records
