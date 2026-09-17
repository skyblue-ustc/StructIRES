#!/usr/bin/env python3
"""Freeze the three reporter-only cargo contexts from Chen et al. Table S1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REPORTERS = ("nano Luciferase-3×Flag", "firefly Luciferase-3×Flag", "mCherry-3×Flag")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-s1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_excel(args.table_s1, sheet_name="B. ivcRNA sequence", header=None)
    records = []
    for label in REPORTERS:
        matches = table[table.iloc[:, 0].astype(str) == label]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one Table S1 entry for {label!r}, found {len(matches)}")
        sequence = str(matches.iloc[0, 1]).upper().replace("T", "U")
        if not sequence or set(sequence) - set("ACGU"):
            raise ValueError(f"non-canonical reporter sequence for {label!r}")
        records.append({"cargo_id": label, "sequence": sequence})
    output = {
        "schema_version": 1,
        "source": {
            "citation": "Chen et al. (2026), Table S1, sheet B. ivcRNA sequence",
            "table_s1": str(args.table_s1),
            "table_s1_sha256": sha256(args.table_s1),
        },
        "scope": "fixed public reporter cargo contexts for computational RNA structure stress testing only",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n_reporter_cargoes": len(records), "lengths": [len(row["sequence"]) for row in records]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
