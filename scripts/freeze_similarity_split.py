#!/usr/bin/env python3
"""Freeze a deterministic cluster-disjoint train/validation/test manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ires_design.prediction import load_ires_ai_records  # noqa: E402


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--edge-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=174)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_tie_key(sequence_ids: list[str], seed: int) -> bytes:
    payload = f"{seed}\x1f{min(sequence_ids)}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def assign_clusters(
    clusters: list[list[int]],
    labels: list[int],
    fractions: dict[str, float],
    sequence_ids: list[str],
    seed: int,
) -> tuple[dict[int, str], dict[str, Counter[str]]]:
    total_by_key = Counter({"total": len(labels), "positive": sum(labels)})
    total_by_key["negative"] = total_by_key["total"] - total_by_key["positive"]
    targets = {
        split: {key: fraction * value for key, value in total_by_key.items()}
        for split, fraction in fractions.items()
    }
    current = {split: Counter() for split in fractions}
    assignment: dict[int, str] = {}
    ordered = sorted(
        clusters,
        key=lambda cluster: (-len(cluster), stable_tie_key([sequence_ids[i] for i in cluster], seed)),
    )
    split_order = {"validation": 0, "test": 1, "train": 2}
    for cluster in ordered:
        counts = Counter(
            {
                "total": len(cluster),
                "positive": sum(labels[index] for index in cluster),
            }
        )
        counts["negative"] = counts["total"] - counts["positive"]

        def placement_score(split: str) -> tuple[float, float, int]:
            relative_need = 0.0
            overflow = 0.0
            for key in ("total", "positive", "negative"):
                target = max(targets[split][key], 1.0)
                deficit = (target - current[split][key]) / target
                weight = counts[key] / max(total_by_key[key], 1)
                relative_need += deficit * weight
                overflow += max(current[split][key] + counts[key] - target, 0.0) / target
            return relative_need - 2.0 * overflow, -current[split]["total"], -split_order[split]

        selected = max(fractions, key=placement_score)
        for index in cluster:
            assignment[index] = selected
        current[selected].update(counts)
    return assignment, current


def main() -> int:
    args = parse_args()
    fractions = {
        "train": args.train_fraction,
        "validation": args.validation_fraction,
        "test": args.test_fraction,
    }
    if abs(sum(fractions.values()) - 1.0) > 1e-9 or any(value <= 0 for value in fractions.values()):
        raise ValueError("split fractions must be positive and sum to one")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    records = [row for row in load_ires_ai_records(args.dataset) if row.length == args.length]
    record_index = {row.sequence_id: index for index, row in enumerate(records)}
    union_find = UnionFind(len(records))
    edges: list[tuple[int, int]] = []
    with gzip.open(args.edge_file, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            left = record_index[row["left_sequence_id"]]
            right = record_index[row["right_sequence_id"]]
            union_find.union(left, right)
            edges.append((left, right))

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[union_find.find(index)].append(index)
    clusters = list(grouped.values())
    assignment, split_counts = assign_clusters(
        clusters,
        [row.label for row in records],
        fractions,
        [row.sequence_id for row in records],
        args.seed,
    )
    cluster_id_by_index: dict[int, str] = {}
    for cluster in clusters:
        cluster_id = f"cluster_{min(records[index].sequence_id for index in cluster)}"
        for index in cluster:
            cluster_id_by_index[index] = cluster_id
    crossing_edges = sum(assignment[left] != assignment[right] for left, right in edges)
    if crossing_edges:
        raise RuntimeError(f"cluster leakage detected across {crossing_edges} identity edges")

    output_rows = []
    for index, record in enumerate(records):
        output_rows.append(
            {
                "sequence_id": record.sequence_id,
                "sequence_sha256": hashlib.sha256(record.sequence.encode("ascii")).hexdigest(),
                "label": record.label,
                "source": record.source,
                "length": record.length,
                "cluster_id": cluster_id_by_index[index],
                "split": assignment[index],
            }
        )
    output_rows.sort(key=lambda row: str(row["sequence_id"]))
    assignment_path = args.output_dir / "assignments.csv.gz"
    with gzip.open(assignment_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    cluster_sizes = sorted((len(cluster) for cluster in clusters), reverse=True)
    cluster_splits = Counter(assignment[cluster[0]] for cluster in clusters)
    summary = {
        "schema_version": 1,
        "protocol": "cluster-disjoint exact Hamming identity for equal-length sequences",
        "sequence_length": args.length,
        "n_sequences": len(records),
        "n_clusters": len(clusters),
        "n_singleton_clusters": sum(size == 1 for size in cluster_sizes),
        "n_non_singleton_sequences": sum(size for size in cluster_sizes if size > 1),
        "largest_cluster_sizes": cluster_sizes[:20],
        "n_identity_edges": len(edges),
        "n_cross_split_identity_edges": crossing_edges,
        "requested_fractions": fractions,
        "split_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "split_cluster_counts": dict(cluster_splits),
        "limitations": (
            "The graph is exact for ungapped equal-length Hamming identity. Gapped and "
            "cross-length relationships require the separate local-alignment audit."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "dataset_path": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "edge_file": str(args.edge_file),
        "edge_file_sha256": file_sha256(args.edge_file),
        "assignment_file": str(assignment_path),
        "assignment_file_sha256": file_sha256(assignment_path),
        "seed": args.seed,
        "test_labels_used_for_model_fitting_or_split_selection": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
