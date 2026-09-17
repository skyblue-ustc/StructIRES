#!/usr/bin/env python3
"""Export a validated sequence CSV as FASTA with a small provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--sequence-column", required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--rna-to-dna", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.input_csv, usecols=[args.id_column, args.sequence_column])
    if data[args.id_column].isna().any() or data[args.id_column].duplicated().any():
        raise ValueError("FASTA identifiers must be non-null and unique")
    sequences = data[args.sequence_column].astype(str).str.upper()
    if args.rna_to_dna:
        sequences = sequences.str.replace("U", "T", regex=False)
    allowed = set("ACGT") if args.rna_to_dna else set("ACGTU")
    invalid = sequences.map(lambda sequence: set(sequence) - allowed)
    if invalid.map(bool).any():
        raise ValueError(f"invalid sequence alphabets: {invalid[invalid.map(bool)].head().tolist()}")
    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with args.output_fasta.open("w", encoding="ascii") as handle:
        for sample_id, sequence in zip(data[args.id_column].astype(str), sequences):
            handle.write(f">{sample_id}\n{sequence}\n")
    manifest = {
        "schema_version": 1,
        "experiment": "sequence_csv_fasta_export",
        "input_csv": str(args.input_csv.resolve()),
        "input_csv_sha256": sha256(args.input_csv),
        "id_column": args.id_column,
        "sequence_column": args.sequence_column,
        "rna_to_dna": args.rna_to_dna,
        "n": int(len(data)),
        "output_fasta_sha256": sha256(args.output_fasta),
    }
    args.output_fasta.with_suffix(args.output_fasta.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
