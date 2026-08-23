#!/usr/bin/env python3
"""Build a label-free, position-aware ViennaRNA profile cache for StructIRES.

Every profile position contains ensemble pairing probability, MFE paired state,
centroid paired state and normalized position.  It intentionally contains no
activity labels and is extracted before model fitting.  The output is kept
outside Git because it is a derived run asset.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ires_design.prediction import load_ires_ai_records
from ires_design.structure import fold_ensemble

CHANNELS = (
    "ensemble_pairing_probability",
    "mfe_paired_state",
    "centroid_paired_state",
    "normalized_position",
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True,
                        help="locked benchmark assignments; profiles are built only for these IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile(sequence: str) -> np.ndarray:
    folded = fold_ensemble(sequence)
    length = len(sequence)
    result = np.empty((length, len(CHANNELS)), dtype=np.float32)
    result[:, 0] = np.asarray(folded.paired_probabilities, dtype=np.float32)
    result[:, 1] = np.fromiter((symbol != "." for symbol in folded.mfe_structure), dtype=np.float32, count=length)
    result[:, 2] = np.fromiter((symbol != "." for symbol in folded.centroid_structure), dtype=np.float32, count=length)
    result[:, 3] = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    if not np.all(np.isfinite(result)) or result.shape != (length, len(CHANNELS)):
        raise RuntimeError("invalid structural profile")
    return result


def main() -> int:
    options = args()
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite existing profile cache: {options.output}")
    with gzip.open(options.assignments, "rt", encoding="utf-8", newline="") as handle:
        assigned = {row["sequence_id"]: row for row in csv.DictReader(handle)}
    records = sorted((row for row in load_ires_ai_records(options.dataset) if row.sequence_id in assigned),
                     key=lambda row: row.sequence_id)
    if len(records) != len(assigned):
        raise ValueError("locked assignments and canonical dataset IDs differ")
    for row in records:
        observed = hashlib.sha256(row.sequence.encode("ascii")).hexdigest()
        if observed != assigned[row.sequence_id]["sequence_sha256"]:
            raise ValueError(f"sequence hash mismatch: {row.sequence_id}")
    lengths = {len(row.sequence) for row in records}
    if len(lengths) != 1:
        raise ValueError(f"profile CNN currently requires a common sequence length, found {sorted(lengths)}")
    sequences = [row.sequence for row in records]
    print(f"folding {len(sequences)} sequences into {len(CHANNELS)}-channel position profiles with {options.workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=options.workers) as executor:
        for index, row in enumerate(executor.map(profile, sequences, chunksize=64), start=1):
            rows.append(row)
            if index % 1000 == 0 or index == len(sequences):
                print(f"folded {index}/{len(sequences)}", flush=True)
    values = np.stack(rows).astype(np.float32, copy=False)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(options.output, sequence_ids=np.asarray([row.sequence_id for row in records]), profiles=values, channel_names=np.asarray(CHANNELS))
    manifest = {
        "schema_version": 1,
        "feature_family": "ViennaRNA position-aware ensemble structural profiles",
        "channels": list(CHANNELS),
        "dataset_path": str(options.dataset),
        "dataset_sha256": sha256(options.dataset),
        "assignments_path": str(options.assignments),
        "assignments_sha256": sha256(options.assignments),
        "n_records": len(records),
        "sequence_length": int(values.shape[1]),
        "profile_shape": list(values.shape),
        "uses_activity_labels": False,
        "note": "The cache is a fixed structural input to a downstream train-only-normalized encoder; it is not a measurement of IRES activity.",
    }
    options.output.with_suffix(options.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(options.output), "shape": list(values.shape)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
