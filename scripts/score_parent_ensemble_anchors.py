#!/usr/bin/env python3
"""Quantify retention of parent-derived high-confidence ensemble anchors.

This script keeps the parent-derived anchors and candidate structural metrics
separate from any functional score.  It is therefore suitable for a matched
structure-constraint ablation, not for claiming measured IRES activity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ires_design.structure import ensemble_anchor_pairs, fold_ensemble, weighted_anchor_retention


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_one(payload: tuple[dict[str, object], tuple[tuple[int, int, float], ...]]) -> dict[str, object]:
    candidate, anchors = payload
    folded = fold_ensemble(str(candidate["sequence"]))
    retention = weighted_anchor_retention(anchors, folded.base_pair_probabilities)
    return {
        "candidate_id": candidate["candidate_id"],
        "parent_id": candidate["parent_id"],
        "run_seed": candidate["seed"],
        "sequence_sha256": hashlib.sha256(str(candidate["sequence"]).encode("ascii")).hexdigest(),
        "n_parent_ensemble_anchors": len(anchors),
        "parent_ensemble_anchor_retention": retention,
        "parent_ensemble_anchor_loss": 1.0 - retention,
        "viennarna_version": folded.viennarna_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-probability", type=float, default=0.50)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    panel = json.loads(args.seed_panel.read_text(encoding="utf-8"))
    seed_records = {str(row["seed_id"]): row for row in panel["records"]}
    anchors = {
        seed_id: ensemble_anchor_pairs(
            fold_ensemble(str(record["sequence"])), min_probability=args.min_probability
        )
        for seed_id, record in seed_records.items()
    }
    if any(not values for values in anchors.values()):
        empty = sorted(seed_id for seed_id, values in anchors.items() if not values)
        raise ValueError(f"no high-confidence ensemble anchors for seeds: {empty}")
    candidates: list[dict[str, object]] = []
    input_audits = []
    for path in args.inputs:
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        candidates.extend(rows)
        input_audits.append({"path": str(path), "sha256": sha256(path), "n_candidates": len(rows)})
    missing = sorted({str(row["parent_id"]) for row in candidates} - set(anchors))
    if missing:
        raise ValueError(f"candidates have missing parent anchors: {missing}")
    payloads = [(row, anchors[str(row["parent_id"])]) for row in candidates]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(score_one, payloads, chunksize=16))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "experiment": "parent_ensemble_anchor_retention",
        "scope": "parent-derived thermodynamic ensemble anchors; not experimentally mapped structural motifs or activity validation",
        "inputs": input_audits,
        "seed_panel": {"path": str(args.seed_panel), "sha256": sha256(args.seed_panel)},
        "min_pair_probability": args.min_probability,
        "anchor_counts_by_parent": {seed_id: len(values) for seed_id, values in anchors.items()},
        "n_candidates": len(rows),
        "workers": args.workers,
        "viennarna_version": rows[0]["viennarna_version"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
