#!/usr/bin/env python3
"""Make matched random, score-only, structure-only and robust selections."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def percentile_ranks(rows: list[dict[str, object]], key: str, *, higher_is_better: bool) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=higher_is_better)
    denominator = max(1, len(ordered) - 1)
    return {str(row["candidate_id"]): index / denominator for index, row in enumerate(ordered)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--likelihood", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    with args.likelihood.open(newline="", encoding="utf-8") as handle:
        likelihood = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    with args.structure.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        score = likelihood.get(row["candidate_id"])
        if score is None:
            raise ValueError(f"missing likelihood score for {row['candidate_id']}")
        row.update(score)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run_seed"]), str(row["parent_id"]))].append(row)
    selected: list[dict[str, object]] = []
    for (run_seed, parent_id), group in sorted(grouped.items()):
        if len(group) < args.top_k:
            raise ValueError(f"{parent_id} has fewer candidates than top-k")
        lm_rank = percentile_ranks(group, "lm_log_likelihood_per_token", higher_is_better=True)
        energy_rank = percentile_ranks(group, "mfe_abs_delta_kcal_mol", higher_is_better=False)
        ensemble_rank = percentile_ranks(group, "pairing_profile_l1", higher_is_better=False)
        methods = {
            "random_mutation": random.Random(f"{run_seed}|{parent_id}|random").sample(group, args.top_k),
            "score_only": sorted(group, key=lambda row: lm_rank[row["candidate_id"]])[: args.top_k],
            "structure_only": sorted(
                group,
                key=lambda row: (energy_rank[row["candidate_id"]] + ensemble_rank[row["candidate_id"]]) / 2,
            )[: args.top_k],
            "robust_full": sorted(
                group,
                key=lambda row: (
                    lm_rank[row["candidate_id"]]
                    + energy_rank[row["candidate_id"]]
                    + ensemble_rank[row["candidate_id"]]
                )
                / 3,
            )[: args.top_k],
        }
        for method, method_rows in methods.items():
            for rank, row in enumerate(method_rows, start=1):
                selected.append({"method": method, "selection_rank": rank, **row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    metrics = ["lm_log_likelihood_per_token", "mfe_abs_delta_kcal_mol", "pairing_profile_l1", "mfe_structure_state_identity"]
    summary: list[dict[str, object]] = []
    for method in ("random_mutation", "score_only", "structure_only", "robust_full"):
        method_rows = [row for row in selected if row["method"] == method]
        values = {metric: sum(float(row[metric]) for row in method_rows) / len(method_rows) for metric in metrics}
        summary.append({"method": method, "n": len(method_rows), **values})
    with args.summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"n_selected": len(selected), "top_k": args.top_k, "methods": 4}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
