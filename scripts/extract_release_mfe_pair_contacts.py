#!/usr/bin/env python3
"""Build a label-free sparse MFE base-pair contact cache for StructIRES.

Unlike the existing per-position profile, this asset preserves *which two
positions* ViennaRNA predicts to pair.  It is a sparse structural graph input
for a later encoder, not an IRES-activity label or experimental measurement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=min(32, max(1, (os.cpu_count() or 2) // 2)))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_unique_sequences(dataset: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(dataset) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv") and "__MACOSX" not in name]
        if len(names) != 1:
            raise ValueError("expected one canonical CSV in benchmark archive")
        records: dict[str, str] = {}
        with archive.open(names[0]) as handle:
            reader = csv.DictReader((line.decode("utf-8") for line in handle))
            for row in reader:
                identifier = str(row["ID"])
                sequence = str(row["Sequence"]).upper().replace("T", "U")
                if not sequence or set(sequence) - set("ACGU"):
                    raise ValueError(f"non-canonical sequence: {identifier}")
                previous = records.setdefault(identifier, sequence)
                if previous != sequence:
                    raise ValueError(f"inconsistent repeated sequence for {identifier}")
    if len(records) != 46774:
        raise ValueError(f"expected 46774 unique sequences, found {len(records)}")
    return sorted(records.items())


def parse_pairs(dot_bracket: str) -> np.ndarray:
    openings = "([{<"
    closing = ")]}>"
    inverse = dict(zip(closing, openings))
    stacks = {symbol: [] for symbol in openings}
    pairs: list[tuple[int, int]] = []
    for index, symbol in enumerate(dot_bracket):
        if symbol in stacks:
            stacks[symbol].append(index)
        elif symbol in inverse:
            opening = inverse[symbol]
            if not stacks[opening]:
                raise ValueError("unbalanced dot-bracket output")
            pairs.append((stacks[opening].pop(), index))
        elif symbol != ".":
            raise ValueError(f"unsupported dot-bracket symbol: {symbol!r}")
    if any(stacks.values()):
        raise ValueError("unbalanced dot-bracket output")
    return np.asarray(pairs, dtype=np.int32).reshape((-1, 2))


def fold_contacts(task: tuple[str, int]) -> tuple[np.ndarray, int]:
    sequence, max_length = task
    sequence = sequence[:max_length]
    import RNA
    structure, _ = RNA.fold_compound(sequence).mfe()
    pair = parse_pairs(structure)
    if pair.size and (pair.min() < 0 or pair.max() >= len(sequence)):
        raise RuntimeError("contact index outside truncated sequence")
    return pair, len(sequence)


def main() -> int:
    args = arguments()
    if args.max_length < 1 or args.workers < 1:
        raise ValueError("max-length and workers must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    rows = load_unique_sequences(args.dataset)
    args.output_dir.mkdir(parents=True)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    contact_chunks: list[np.ndarray] = []
    lengths: list[int] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        tasks = ((sequence, args.max_length) for _, sequence in rows)
        for index, (pair, length) in enumerate(executor.map(fold_contacts, tasks, chunksize=64), start=1):
            contact_chunks.append(pair)
            offsets[index] = offsets[index - 1] + len(pair)
            lengths.append(length)
            if index % 1000 == 0 or index == len(rows):
                print(f"folded {index}/{len(rows)}; contacts={offsets[index]}", flush=True)
    contacts = np.concatenate(contact_chunks, axis=0) if contact_chunks else np.empty((0, 2), dtype=np.int32)
    np.save(args.output_dir / "sequence_ids.npy", np.asarray([identifier for identifier, _ in rows]))
    np.save(args.output_dir / "pair_offsets.npy", offsets)
    np.save(args.output_dir / "pairs.npy", contacts)
    manifest = {
        "schema_version": 1,
        "feature_family": "ViennaRNA MFE sparse base-pair contacts",
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "n_unique_sequences": len(rows),
        "n_contacts": int(len(contacts)),
        "max_length": args.max_length,
        "input_length_min": min(lengths),
        "input_length_max_after_truncation": max(lengths),
        "n_truncated": sum(len(sequence) > args.max_length for _, sequence in rows),
        "uses_activity_labels": False,
        "edge_semantics": "zero-indexed undirected MFE base-pair endpoints; sequence adjacency is not included",
        "viennarna_version": __import__("RNA").__version__,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "n_contacts": int(len(contacts))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
