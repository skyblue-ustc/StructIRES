#!/usr/bin/env python3
"""Summarize matched seeded-design selections at the parent-by-run level."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


METRICS = (
    "lm_log_likelihood_per_token",
    "mfe_abs_delta_kcal_mol",
    "pairing_profile_l1",
    "mfe_structure_state_identity",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return (sum((value - center) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-deltas", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--left-method", default="robust_full")
    parser.add_argument("--right-method", default="score_only")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in args.inputs:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("no selected candidates supplied")

    available_metrics = tuple(metric for metric in METRICS if all(metric in row for row in rows))
    if not available_metrics:
        raise ValueError("none of the requested summary metrics are available")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["run_seed"], row["parent_id"], row["method"]), []).append(row)
    group_rows: list[dict[str, object]] = []
    for (run_seed, parent_id, method), group in sorted(grouped.items()):
        group_rows.append(
            {
                "run_seed": int(run_seed),
                "parent_id": parent_id,
                "method": method,
                "n_candidates": len(group),
                **{metric: mean([float(row[metric]) for row in group]) for metric in available_metrics},
            }
        )
    methods = sorted({str(row["method"]) for row in group_rows})
    summary: list[dict[str, object]] = []
    for method in methods:
        method_rows = [row for row in group_rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "n_parent_run_units": len(method_rows),
                **{
                    f"{metric}_mean": mean([float(row[metric]) for row in method_rows])
                    for metric in available_metrics
                },
                **{
                    f"{metric}_std": sample_std([float(row[metric]) for row in method_rows])
                    for metric in available_metrics
                },
            }
        )
    by_key = {(str(row["run_seed"]), str(row["parent_id"]), str(row["method"])): row for row in group_rows}
    deltas: list[dict[str, object]] = []
    for run_seed, parent_id in sorted({(run_seed, parent_id) for run_seed, parent_id, _ in grouped}):
        left = by_key.get((run_seed, parent_id, args.left_method))
        right = by_key.get((run_seed, parent_id, args.right_method))
        if left is None or right is None:
            continue
        deltas.append(
            {
                "run_seed": int(run_seed),
                "parent_id": parent_id,
                **{
                    f"{args.left_method}_minus_{args.right_method}_{metric}": float(left[metric]) - float(right[metric])
                    for metric in available_metrics
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with args.paired_deltas.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["run_seed", "parent_id"] + [
            f"{args.left_method}_minus_{args.right_method}_{metric}" for metric in available_metrics
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(deltas)
    manifest = {
        "schema_version": 1,
        "experiment": "seeded_design_matched_summary",
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in args.inputs],
        "n_selected_candidates": len(rows),
        "n_parent_run_units": len(group_rows),
        "paired_comparison": {"left": args.left_method, "right": args.right_method},
        "n_paired_units": len(deltas),
        "metrics": list(available_metrics),
        "scope": "computational structural and RNA-LM plausibility metrics only; not activity validation",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
