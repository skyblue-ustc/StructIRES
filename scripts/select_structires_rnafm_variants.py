#!/usr/bin/env python3
"""Select matched RNA-FM score-only and constraint-guided StructIRES variants."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def ranks(rows: list[dict[str, str]], key: str, higher_is_better: bool) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=higher_is_better)
    denominator = max(1, len(ordered) - 1)
    return {row["candidate_id"]: index / denominator for index, row in enumerate(ordered)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    score_rows = {row["candidate_id"]: row for row in csv.DictReader(args.scores.open())}
    rows = list(csv.DictReader(args.structure.open()))
    for row in rows:
        score = score_rows.get(row["candidate_id"])
        if score is None:
            raise ValueError(f"missing RNA-FM score: {row['candidate_id']}")
        row.update(score)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_seed"], row["parent_id"])].append(row)
    selected: list[dict[str, str | int]] = []
    for _, group in sorted(grouped.items()):
        if len(group) < args.top_k:
            raise ValueError("candidate pool is smaller than top-k")
        function = ranks(group, "rnafm_probability_fold0", True)
        energy = ranks(group, "mfe_abs_delta_kcal_mol", False)
        ensemble = ranks(group, "pairing_profile_l1", False)
        objectives = {
            "rnafm_score_only": lambda row: function[row["candidate_id"]],
            "rnafm_plus_energy": lambda row: (function[row["candidate_id"]] + energy[row["candidate_id"]]) / 2,
            "rnafm_plus_ensemble": lambda row: (function[row["candidate_id"]] + ensemble[row["candidate_id"]]) / 2,
            "structires": lambda row: (function[row["candidate_id"]] + energy[row["candidate_id"]] + ensemble[row["candidate_id"]]) / 3,
        }
        for method, objective in objectives.items():
            for rank, row in enumerate(sorted(group, key=objective)[: args.top_k], start=1):
                selected.append({"method": method, "selection_rank": rank, **row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print({"n_selected": len(selected), "methods": 4, "top_k": args.top_k})


if __name__ == "__main__":
    main()
