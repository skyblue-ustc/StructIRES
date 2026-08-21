#!/usr/bin/env python3
"""Export one FASTA record per unique sequence in the released IRES-LM dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    with zipfile.ZipFile(args.dataset_zip) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.endswith("v2_dataset_with_unified_stratified_shuffle_train_test_split.csv")
            and not name.startswith("__MACOSX")
        ]
        if len(members) != 1:
            raise ValueError(f"expected one canonical CSV member, found {members}")
        with archive.open(members[0]) as handle:
            data = pd.read_csv(handle, usecols=["idx", "Sequence", "IRES_class_600"])
    consistency = data.groupby("idx").agg(
        sequence_count=("Sequence", "nunique"), label_count=("IRES_class_600", "nunique")
    )
    if int(consistency.to_numpy().max()) != 1:
        raise ValueError("idx does not map to one sequence and label")
    unique = data.drop_duplicates("idx").sort_values("idx", kind="stable")
    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with args.output_fasta.open("w", encoding="ascii") as handle:
        for row in unique.itertuples(index=False):
            handle.write(f">idx_{int(row.idx)}\n{str(row.Sequence).upper()}\n")
    manifest = {
        "schema_version": 1,
        "experiment": "ireslm_unique_fasta_export",
        "dataset_zip": str(args.dataset_zip),
        "dataset_zip_sha256": sha256(args.dataset_zip),
        "n_unique": len(unique),
        "n_positive": int(unique["IRES_class_600"].sum()),
        "output_fasta_sha256": sha256(args.output_fasta),
    }
    args.output_fasta.with_suffix(args.output_fasta.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
