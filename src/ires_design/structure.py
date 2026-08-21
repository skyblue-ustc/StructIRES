"""ViennaRNA-backed ensemble metrics and dependency-free structure comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from .schemas import normalize_rna


@dataclass(frozen=True)
class FoldResult:
    sequence: str
    mfe_structure: str
    mfe_kcal_mol: float
    ensemble_free_energy_kcal_mol: float
    centroid_structure: str
    centroid_distance: float
    ensemble_diversity: float
    paired_probabilities: tuple[float, ...]
    base_pair_probabilities: tuple[tuple[int, int, float], ...]
    viennarna_version: str


def dot_bracket_pairs(structure: str) -> frozenset[tuple[int, int]]:
    """Parse an unpseudoknotted dot-bracket string into zero-based pairs."""
    stack: list[int] = []
    pairs: set[tuple[int, int]] = set()
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced dot-bracket structure")
            pairs.add((stack.pop(), index))
        elif symbol != ".":
            raise ValueError(f"unsupported dot-bracket symbol: {symbol!r}")
    if stack:
        raise ValueError("unbalanced dot-bracket structure")
    return frozenset(pairs)


def base_pair_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("base-pair distance requires equal-length structures")
    return len(dot_bracket_pairs(left).symmetric_difference(dot_bracket_pairs(right)))


def paired_fraction(structure: str) -> float:
    if not structure:
        return float("nan")
    pairs = dot_bracket_pairs(structure)
    return 2.0 * len(pairs) / len(structure)


def structure_state_identity(left: str, right: str) -> float:
    """Fraction of positions whose paired/unpaired state is preserved."""
    if len(left) != len(right):
        raise ValueError("structure-state identity requires equal-length structures")
    if not left:
        return float("nan")
    return sum((a == ".") == (b == ".") for a, b in zip(left, right)) / len(left)


def pairing_profile_distance(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if len(left_values) != len(right_values):
        raise ValueError("pairing profiles must have equal length")
    if not left_values:
        return float("nan")
    return fmean(abs(a - b) for a, b in zip(left_values, right_values))


def ires_crosstalk_ratio(result: FoldResult, ires_length: int) -> float:
    """Expected IRES--cargo base pairs divided by IRES-domain length."""
    if not 0 < ires_length < len(result.sequence):
        raise ValueError("ires_length must split a non-empty IRES and cargo context")
    cross_probability = sum(
        probability
        for left, right, probability in result.base_pair_probabilities
        if left < ires_length <= right
    )
    return cross_probability / ires_length


def context_structure_consistency(reference: FoldResult, context: FoldResult) -> float:
    """One minus mean absolute pairing-probability drift on the IRES domain."""
    ires_length = len(reference.sequence)
    if len(context.sequence) <= ires_length:
        raise ValueError("context fold must append a non-empty cargo to the IRES")
    distance = pairing_profile_distance(
        reference.paired_probabilities,
        context.paired_probabilities[:ires_length],
    )
    return 1.0 - distance


def fold_ensemble(sequence: str, *, min_pair_probability: float = 1e-6) -> FoldResult:
    """Fold a sequence with ViennaRNA and retain ensemble-level quantities.

    Import is intentionally delayed so that lightweight audits remain usable
    without ViennaRNA.  A broken binary installation fails closed instead of
    silently falling back to MFE-only heuristics.
    """
    sequence = normalize_rna(sequence)
    if not sequence or set(sequence) - set("ACGU"):
        raise ValueError("ViennaRNA folding requires a non-empty canonical RNA sequence")
    try:
        import RNA
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "ViennaRNA Python bindings are unavailable or binary-incompatible; "
            "repair the pinned environment before formal structure scoring"
        ) from exc

    compound = RNA.fold_compound(sequence)
    mfe_structure, mfe = compound.mfe()
    compound.exp_params_rescale(mfe)
    _, ensemble_free_energy = compound.pf()
    centroid_structure, centroid_distance = compound.centroid()
    bpp = compound.bpp()
    paired = [0.0] * len(sequence)
    sparse_pairs: list[tuple[int, int, float]] = []
    for left in range(1, len(sequence) + 1):
        for right in range(left + 1, len(sequence) + 1):
            probability = float(bpp[left][right])
            if probability <= 0.0:
                continue
            paired[left - 1] += probability
            paired[right - 1] += probability
            if probability >= min_pair_probability:
                sparse_pairs.append((left - 1, right - 1, probability))
    if any(not math.isfinite(value) or value < 0.0 or value > 1.000001 for value in paired):
        raise RuntimeError("ViennaRNA returned an invalid positional pairing profile")
    return FoldResult(
        sequence=sequence,
        mfe_structure=str(mfe_structure),
        mfe_kcal_mol=float(mfe),
        ensemble_free_energy_kcal_mol=float(ensemble_free_energy),
        centroid_structure=str(centroid_structure),
        centroid_distance=float(centroid_distance),
        ensemble_diversity=float(compound.mean_bp_distance()),
        paired_probabilities=tuple(paired),
        base_pair_probabilities=tuple(sparse_pairs),
        viennarna_version=str(RNA.__version__),
    )


def compare_seed_and_variant(seed: FoldResult, variant: FoldResult) -> dict[str, float]:
    if len(seed.sequence) != len(variant.sequence):
        raise ValueError("seed and variant must have equal length")
    return {
        "mfe_delta_kcal_mol": variant.mfe_kcal_mol - seed.mfe_kcal_mol,
        "mfe_abs_delta_kcal_mol": abs(variant.mfe_kcal_mol - seed.mfe_kcal_mol),
        "mfe_base_pair_distance": float(base_pair_distance(seed.mfe_structure, variant.mfe_structure)),
        "mfe_structure_state_identity": structure_state_identity(
            seed.mfe_structure, variant.mfe_structure
        ),
        "pairing_profile_l1": pairing_profile_distance(
            seed.paired_probabilities, variant.paired_probabilities
        ),
        "ensemble_diversity_delta": variant.ensemble_diversity - seed.ensemble_diversity,
    }
