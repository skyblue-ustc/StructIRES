#!/usr/bin/env python3
"""Aggregate frozen StructIRES-Adapter fold predictions by arithmetic mean."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if len(args.inputs) < 2:
        raise ValueError("at least two fold-score files are required")

    tables = []
    score_columns = []
    for path in args.inputs:
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        columns = [name for name in rows[0] if name.startswith("structires_adapter_probability_fold")]
        if len(columns) != 1:
            raise ValueError(f"expected one fold score in {path}, found {columns}")
        tables.append({row["candidate_id"]: row for row in rows})
        score_columns.append(columns[0])
    identifiers = list(tables[0])
    if any(set(table) != set(identifiers) for table in tables[1:]):
        raise ValueError("candidate IDs differ across fold score files")
    if len(set(score_columns)) != len(score_columns):
        raise ValueError("duplicate fold score columns")

    output = []
    for identifier in identifiers:
        base = tables[0][identifier]
        values = [float(table[identifier][column]) for table, column in zip(tables, score_columns)]
        row = {
            "candidate_id": identifier,
            "parent_id": base["parent_id"],
            "run_seed": base["run_seed"],
            "pool_dir": base["pool_dir"],
            "sequence_sha256": base["sequence_sha256"],
        }
        row.update({column: value for column, value in zip(score_columns, values)})
        row["structires_adapter_probability_mean"] = float(np.mean(values))
        row["structires_adapter_probability_std"] = float(np.std(values, ddof=1))
        output.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "structires_adapter_pool_scores.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    manifest = {
        "schema_version": 1,
        "experiment": "three_fold_structires_adapter_ensemble_on_frozen_shared_pools",
        "aggregation": "arithmetic mean of frozen fold probabilities",
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.inputs
        ],
        "fold_score_columns": score_columns,
        "n_models": len(score_columns),
        "n_candidates": len(output),
        "uses_candidate_activity_labels": False,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_candidates": len(output), "n_models": len(score_columns)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
