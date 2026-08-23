#!/usr/bin/env python3
"""Subset a canonical label-free MFE contact cache to a locked split order.

The input cache is keyed by the full released IRES benchmark.  This utility
creates an immutable, compact cache whose record order is exactly the order
used by ``train_structires_rnafm.py``.  Labels are never read or used.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ires_design.prediction import load_ires_ai_records


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True,
                        help="locked split assignment; only listed records are retained")
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    with gzip.open(args.assignments, "rt", encoding="utf-8", newline="") as handle:
        assigned = {row["sequence_id"] for row in csv.DictReader(handle)}
    records = sorted((row for row in load_ires_ai_records(args.dataset) if row.sequence_id in assigned),
                     key=lambda row: row.sequence_id)
    if len(records) != len(assigned):
        raise ValueError("locked assignments and canonical dataset differ")
    source_ids = np.load(args.contact_dir / "sequence_ids.npy", allow_pickle=False).astype(str)
    source_offsets = np.load(args.contact_dir / "pair_offsets.npy", mmap_mode="r")
    source_pairs = np.load(args.contact_dir / "pairs.npy", mmap_mode="r")
    if len(source_offsets) != len(source_ids) + 1:
        raise ValueError("contact offsets and IDs differ")
    source_index = {identifier: index for index, identifier in enumerate(source_ids.tolist())}
    ids = [record.sequence_id for record in records]
    if len(ids) != len(set(ids)) or any(identifier not in source_index for identifier in ids):
        raise ValueError("canonical records do not match contact cache IDs")
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    chunks: list[np.ndarray] = []
    for output_index, identifier in enumerate(ids, start=1):
        input_index = source_index[identifier]
        start, stop = int(source_offsets[input_index]), int(source_offsets[input_index + 1])
        edge = np.asarray(source_pairs[start:stop], dtype=np.int32)
        chunks.append(edge)
        offsets[output_index] = offsets[output_index - 1] + len(edge)
    pairs = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), dtype=np.int32)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "sequence_ids.npy", np.asarray(ids))
    np.save(args.output_dir / "pair_offsets.npy", offsets)
    np.save(args.output_dir / "pairs.npy", pairs)
    manifest = {
        "schema_version": 1,
        "feature_family": "ViennaRNA MFE sparse base-pair contacts",
        "edge_semantics": "zero-indexed undirected MFE base-pair endpoints; sequence adjacency is not included",
        "uses_activity_labels": False,
        "dataset_sha256": sha256(args.dataset),
        "assignments_sha256": sha256(args.assignments),
        "source_contact_manifest_sha256": sha256(args.contact_dir / "manifest.json"),
        "n_sequences": len(ids),
        "n_contacts": int(len(pairs)),
        "record_order": "canonical sequence_id ascending, matching train_structires_rnafm.py",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "n_sequences": len(ids), "n_contacts": int(len(pairs))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
