"""Low-cost sequence audits run before model- or structure-based scoring."""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean, pstdev
from typing import Any

from .schemas import normalize_rna


RNA_ALPHABET = frozenset("ACGU")


def longest_homopolymer(sequence: str) -> int:
    sequence = normalize_rna(sequence)
    if not sequence:
        return 0
    longest = current = 1
    for previous, current_base in zip(sequence, sequence[1:]):
        current = current + 1 if current_base == previous else 1
        longest = max(longest, current)
    return longest


def shannon_entropy(sequence: str) -> float:
    sequence = normalize_rna(sequence)
    if not sequence:
        return float("nan")
    counts = Counter(sequence)
    total = len(sequence)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def sequence_metrics(sequence: str, expected_length: int | None = None) -> dict[str, Any]:
    sequence = normalize_rna(sequence)
    alphabet_valid = bool(sequence) and set(sequence) <= RNA_ALPHABET
    gc = (
        100.0 * (sequence.count("G") + sequence.count("C")) / len(sequence)
        if sequence
        else float("nan")
    )
    return {
        "length": len(sequence),
        "length_valid": expected_length is None or len(sequence) == expected_length,
        "alphabet_valid": alphabet_valid,
        "valid": alphabet_valid and (expected_length is None or len(sequence) == expected_length),
        "gc_pct": gc,
        "max_homopolymer": longest_homopolymer(sequence),
        "base_entropy_bits": shannon_entropy(sequence),
    }


def summarize_sequences(sequences: list[str], expected_length: int | None = None) -> dict[str, Any]:
    rows = [sequence_metrics(sequence, expected_length) for sequence in sequences]
    if not rows:
        return {
            "n_raw": 0,
            "n_valid": 0,
            "survival_rate_pct": float("nan"),
            "unique_rate_pct": float("nan"),
        }

    normalized = [normalize_rna(sequence) for sequence in sequences]
    valid_rows = [row for row in rows if row["valid"]]
    lengths = [row["length"] for row in rows]
    gc_values = [row["gc_pct"] for row in valid_rows]
    homopolymers = [row["max_homopolymer"] for row in valid_rows]
    entropies = [row["base_entropy_bits"] for row in valid_rows]

    def mean_or_nan(values: list[float | int]) -> float:
        return fmean(values) if values else float("nan")

    return {
        "n_raw": len(rows),
        "n_valid": len(valid_rows),
        "survival_rate_pct": 100.0 * len(valid_rows) / len(rows),
        "n_unique": len(set(normalized)),
        "unique_rate_pct": 100.0 * len(set(normalized)) / len(normalized),
        "length_min": min(lengths),
        "length_mean": fmean(lengths),
        "length_max": max(lengths),
        "gc_mean_pct": mean_or_nan(gc_values),
        "gc_std_pct": pstdev(gc_values) if len(gc_values) > 1 else 0.0 if gc_values else float("nan"),
        "max_homopolymer_mean": mean_or_nan(homopolymers),
        "max_homopolymer_max": max(homopolymers) if homopolymers else None,
        "base_entropy_mean_bits": mean_or_nan(entropies),
        "expected_length": expected_length,
    }

