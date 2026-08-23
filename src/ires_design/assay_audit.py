"""Cross-assay overlap audit for legacy DNA-reporter and direct-RNA IRES data."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .prediction import PredictionRecord, load_ires_ai_records


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_direct_sequences(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _best_kmer_neighbors(
    legacy: list[PredictionRecord],
    direct: list[dict[str, str]],
    *,
    n_neighbors: int = 5,
    query_batch_size: int = 64,
) -> list[list[tuple[int, float]]]:
    """Return candidate neighbors using binary 6-mer cosine similarity.

    This is a candidate-retrieval stage, not the reported identity.  The final
    rows are aligned explicitly with Biopython.
    """
    try:
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import normalize
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("numpy and scikit-learn are required for the cross-assay audit") from exc

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), binary=True, dtype=np.float32)
    legacy_matrix = normalize(vectorizer.fit_transform([row.sequence for row in legacy]), copy=False)
    direct_matrix = normalize(vectorizer.transform([row["sequence"] for row in direct]), copy=False)
    result: list[list[tuple[int, float]]] = []
    keep = min(n_neighbors, len(legacy))
    for start in range(0, len(direct), query_batch_size):
        similarities = (direct_matrix[start : start + query_batch_size] @ legacy_matrix.T).toarray()
        for scores in similarities:
            indices = np.argpartition(scores, -keep)[-keep:]
            ordered = indices[np.argsort(scores[indices])[::-1]]
            result.append([(int(index), float(scores[index])) for index in ordered])
    return result


def _local_alignment_stats(left: str, right: str) -> tuple[float, float, int]:
    try:
        from Bio.Align import PairwiseAligner
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("biopython is required for the cross-assay audit") from exc

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(left, right)
    try:
        alignment = next(iter(alignments))
    except StopIteration:
        return 0.0, 0.0, 0
    counts = alignment.counts()
    aligned = int(counts.identities + counts.mismatches)
    gaps = int(counts.gaps)
    denominator = aligned + gaps
    identity = counts.identities / denominator if denominator else 0.0
    coverage = aligned / min(len(left), len(right)) if left and right else 0.0
    return float(identity), float(coverage), int(counts.identities)


def cross_assay_overlap_rows(
    legacy: list[PredictionRecord], direct: list[dict[str, str]], *, n_neighbors: int = 5
) -> list[dict[str, Any]]:
    candidates = _best_kmer_neighbors(legacy, direct, n_neighbors=n_neighbors)
    legacy_exact: dict[str, list[int]] = {}
    legacy_by_length: dict[int, dict[str, list[int]]] = {}
    for index, row in enumerate(legacy):
        legacy_exact.setdefault(row.sequence, []).append(index)
        legacy_by_length.setdefault(len(row.sequence), {}).setdefault(row.sequence, []).append(index)

    output: list[dict[str, Any]] = []
    for direct_row, neighbor_rows in zip(direct, candidates):
        sequence = direct_row["sequence"]
        exact_indices = legacy_exact.get(sequence, [])
        candidate_indices = {index for index, _ in neighbor_rows} | set(exact_indices)
        # Exact containment is cheap and biologically informative when the old
        # 174-nt oligo is a fragment of a longer direct-RNA construct.
        contained_indices: set[int] = set()
        for legacy_length, sequence_index in legacy_by_length.items():
            if legacy_length > len(sequence):
                continue
            for start in range(len(sequence) - legacy_length + 1):
                contained_indices.update(
                    sequence_index.get(sequence[start : start + legacy_length], ())
                )
        candidate_indices |= contained_indices

        scored: list[tuple[float, float, int, int]] = []
        for index in candidate_indices:
            identity, coverage, identities = _local_alignment_stats(sequence, legacy[index].sequence)
            scored.append((identity * coverage, identity, coverage, index))
        _, identity, coverage, best_index = max(scored, default=(0.0, 0.0, 0.0, neighbor_rows[0][0]))
        best = legacy[best_index]
        kmer_score = next((score for index, score in neighbor_rows if index == best_index), 1.0)
        output.append(
            {
                "direct_sequence_id": direct_row["sequence_id"],
                "direct_sequence": sequence,
                "direct_length": len(sequence),
                "direct_consensus_status": direct_row["consensus_status"],
                "direct_consensus_label": direct_row["consensus_label"],
                "direct_is_control": direct_row["is_control"],
                "direct_construct_names": direct_row["construct_names"],
                "direct_groups": direct_row["groups"],
                "legacy_sequence_id": best.sequence_id,
                "legacy_length": best.length,
                "legacy_label": best.label,
                "legacy_source": best.source,
                "exact_sequence_match": bool(exact_indices),
                "legacy_contained_in_direct": bool(contained_indices),
                "sixmer_cosine": kmer_score,
                "local_alignment_identity": identity,
                "local_alignment_coverage_min_length": coverage,
                "high_similarity_90_80": identity >= 0.90 and coverage >= 0.80,
            }
        )
    return output


def summarize_cross_assay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [row for row in rows if row["direct_consensus_status"] in {"active", "inactive"}]
    by_status: dict[str, dict[str, int]] = {}
    for status in ("active", "inactive", "unlabeled_or_ambiguous", "conflict"):
        subset = [row for row in rows if row["direct_consensus_status"] == status]
        by_status[status] = {
            "n": len(subset),
            "exact": sum(row["exact_sequence_match"] for row in subset),
            "contained": sum(row["legacy_contained_in_direct"] for row in subset),
            "high_similarity_90_80": sum(row["high_similarity_90_80"] for row in subset),
            "nearest_legacy_negative": sum(row["legacy_label"] == 0 for row in subset),
            "nearest_legacy_positive": sum(row["legacy_label"] == 1 for row in subset),
        }
    contained_labelled = [
        row
        for row in labelled
        if row["legacy_contained_in_direct"]
    ]
    agreement = sum(
        (row["direct_consensus_status"] == "active") == bool(row["legacy_label"])
        for row in labelled
    )
    contained_agreement = sum(
        (row["direct_consensus_status"] == "active") == bool(row["legacy_label"])
        for row in contained_labelled
    )
    return {
        "schema_version": 1,
        "n_direct_sequences": len(rows),
        "n_direct_labelled": len(labelled),
        "n_exact_sequence_matches": sum(row["exact_sequence_match"] for row in rows),
        "n_legacy_contained_in_direct": sum(row["legacy_contained_in_direct"] for row in rows),
        "n_high_similarity_90_80": sum(row["high_similarity_90_80"] for row in rows),
        "nearest_legacy_label_agreement_rate": agreement / len(labelled) if labelled else None,
        "contained_label_agreement_rate": (
            contained_agreement / len(contained_labelled) if contained_labelled else None
        ),
        "nearest_legacy_label_counts": dict(sorted(Counter(str(row["legacy_label"]) for row in rows).items())),
        "overlap_by_direct_status": by_status,
        "interpretation": (
            "Similarity quantifies domain overlap, not assay agreement. Functional transfer must be "
            "evaluated with frozen legacy scorers on direct-RNA labels."
        ),
    }


def run_cross_assay_overlap_audit(
    legacy_path: Path, direct_path: Path, output_dir: Path
) -> dict[str, Any]:
    legacy = load_ires_ai_records(legacy_path)
    direct = _load_direct_sequences(direct_path)
    rows = cross_assay_overlap_rows(legacy, direct)
    summary = summarize_cross_assay(rows)
    summary["inputs"] = {
        "legacy_path": str(legacy_path),
        "legacy_sha256": sha256_file(legacy_path),
        "direct_path": str(direct_path),
        "direct_sha256": sha256_file(direct_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "nearest_legacy_overlap.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
