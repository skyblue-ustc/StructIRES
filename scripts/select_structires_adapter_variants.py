#!/usr/bin/env python3
"""Select frozen StructIRES-Adapter and structure-constrained candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_rank(rows: list[dict[str, str]], key: str, *, higher: bool) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=higher)
    denominator = max(1, len(ordered) - 1)
    return {row["candidate_id"]: index / denominator for index, row in enumerate(ordered)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--structure", type=Path, nargs="+", required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    scores = {
        row["candidate_id"]: row
        for row in csv.DictReader(args.scores.open(encoding="utf-8", newline=""))
    }
    anchors = {
        row["candidate_id"]: row
        for row in csv.DictReader(args.anchors.open(encoding="utf-8", newline=""))
    }
    rows: list[dict[str, str]] = []
    for path in args.structure:
        for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
            row["pool_dir"] = str(path.parent.resolve())
            rows.append(row)
    identifiers = {row["candidate_id"] for row in rows}
    if identifiers != set(scores) or identifiers != set(anchors):
        raise ValueError("candidate IDs differ across score, structure, and anchor inputs")
    for row in rows:
        row.update(scores[row["candidate_id"]])
        row.update(anchors[row["candidate_id"]])

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_seed"], row["parent_id"])].append(row)
    selected: list[dict[str, object]] = []
    method_names = (
        "adapter_score_only",
        "adapter_plus_energy",
        "adapter_plus_ensemble",
        "adapter_plus_anchor",
        "structires_adapter_rank",
    )
    for _, group in sorted(grouped.items()):
        if len(group) < args.top_k:
            raise ValueError("candidate pool smaller than requested top-k")
        function = percentile_rank(
            group, "structires_adapter_probability_mean", higher=True
        )
        energy = percentile_rank(group, "mfe_abs_delta_kcal_mol", higher=False)
        ensemble = percentile_rank(group, "pairing_profile_l1", higher=False)
        anchor = percentile_rank(group, "parent_ensemble_anchor_loss", higher=False)
        objectives = {
            method_names[0]: lambda row: function[row["candidate_id"]],
            method_names[1]: lambda row: (
                function[row["candidate_id"]] + energy[row["candidate_id"]]
            ) / 2.0,
            method_names[2]: lambda row: (
                function[row["candidate_id"]] + ensemble[row["candidate_id"]]
            ) / 2.0,
            method_names[3]: lambda row: (
                function[row["candidate_id"]] + anchor[row["candidate_id"]]
            ) / 2.0,
            method_names[4]: lambda row: (
                function[row["candidate_id"]]
                + energy[row["candidate_id"]]
                + ensemble[row["candidate_id"]]
                + anchor[row["candidate_id"]]
            ) / 4.0,
        }
        for method, objective in objectives.items():
            for rank, row in enumerate(
                sorted(group, key=lambda item: (objective(item), item["candidate_id"]))[
                    : args.top_k
                ],
                start=1,
            ):
                selected.append({"method": method, "selection_rank": rank, **row})

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "selected.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "schema_version": 1,
        "experiment": "structires_adapter_proposal_score_matched_selection",
        "score_column": "structires_adapter_probability_mean",
        "methods": list(method_names),
        "top_k_per_parent_pool": args.top_k,
        "n_parent_pool_units": len(grouped),
        "n_selected": len(selected),
        "scores": {"path": str(args.scores.resolve()), "sha256": sha256(args.scores)},
        "structure": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.structure
        ],
        "anchors": {"path": str(args.anchors.resolve()), "sha256": sha256(args.anchors)},
        "uses_candidate_activity_labels": False,
        "tie_break": "candidate_id ascending after mean-rank objective",
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_selected": len(selected), "methods": len(method_names)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
