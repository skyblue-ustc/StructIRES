#!/usr/bin/env python3
"""Select matched IRES-LM and structure-constrained candidates from frozen pools."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def percentile_rank(rows: list[dict[str, str]], key: str, *, higher: bool) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=higher)
    denominator = max(1, len(ordered) - 1)
    return {row["candidate_id"]: index / denominator for index, row in enumerate(ordered)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--structure", type=Path, nargs="+", required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--functional-risk", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    scores = {row["candidate_id"]: row for row in csv.DictReader(args.scores.open(encoding="utf-8", newline=""))}
    anchors = {row["candidate_id"]: row for row in csv.DictReader(args.anchors.open(encoding="utf-8", newline=""))}
    functional_risk = (
        {row["candidate_id"]: row for row in csv.DictReader(args.functional_risk.open(encoding="utf-8", newline=""))}
        if args.functional_risk else {}
    )
    rows: list[dict[str, str]] = []
    for path in args.structure:
        # Keep provenance of the frozen pool.  The selected CSV intentionally
        # stores no sequence, so downstream evaluators recover it from this
        # immutable source rather than relying on an ambiguous sibling path.
        for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
            row["pool_dir"] = str(path.parent)
            rows.append(row)
    identifiers = {row["candidate_id"] for row in rows}
    if identifiers != set(scores) or identifiers != set(anchors):
        raise ValueError("candidate IDs differ across score, structure and anchor inputs")
    for row in rows:
        row.update(scores[row["candidate_id"]])
        row.update(anchors[row["candidate_id"]])
        if row["candidate_id"] in functional_risk:
            row.update(functional_risk[row["candidate_id"]])
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_seed"], row["parent_id"])].append(row)
    selected: list[dict[str, str | int]] = []
    for _, group in sorted(grouped.items()):
        if len(group) < args.top_k:
            raise ValueError("candidate pool smaller than requested top-k")
        function = percentile_rank(group, "ireslm_probability_mean", higher=True)
        energy = percentile_rank(group, "mfe_abs_delta_kcal_mol", higher=False)
        ensemble = percentile_rank(group, "pairing_profile_l1", higher=False)
        anchors_rank = percentile_rank(group, "parent_ensemble_anchor_loss", higher=False)
        objectives = {
            "ireslm_score_only": lambda row: function[row["candidate_id"]],
            "ireslm_plus_energy": lambda row: (function[row["candidate_id"]] + energy[row["candidate_id"]]) / 2.0,
            "ireslm_plus_ensemble": lambda row: (function[row["candidate_id"]] + ensemble[row["candidate_id"]]) / 2.0,
            "ireslm_plus_anchor": lambda row: (function[row["candidate_id"]] + anchors_rank[row["candidate_id"]]) / 2.0,
            "structires": lambda row: (
                function[row["candidate_id"]]
                + energy[row["candidate_id"]]
                + ensemble[row["candidate_id"]]
                + anchors_rank[row["candidate_id"]]
            ) / 4.0,
        }
        if group[0]["parent_id"] == "IAPV" and functional_risk:
            mpra = percentile_rank(group, "iapv_mpra_mutational_risk", higher=False)
            objectives["structires_plus_iapv_mpra_guardrail"] = lambda row: (
                function[row["candidate_id"]] + energy[row["candidate_id"]]
                + ensemble[row["candidate_id"]] + anchors_rank[row["candidate_id"]]
                + mpra[row["candidate_id"]]
            ) / 5.0
        for method, objective in objectives.items():
            for rank, row in enumerate(sorted(group, key=objective)[: args.top_k], start=1):
                selected.append({"method": method, "selection_rank": rank, **row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        # The IAPV-only MPRA guardrail introduces additional provenance columns.
        # Use the union so ordinary parents remain represented with blank fields.
        fieldnames = list(selected[0]) + sorted(set().union(*(row.keys() for row in selected)) - set(selected[0]))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    print({"n_selected": len(selected), "methods": len(objectives), "top_k": args.top_k})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
