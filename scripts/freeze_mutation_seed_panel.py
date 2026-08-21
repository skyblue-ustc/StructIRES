#!/usr/bin/env python3
"""Freeze a balanced full-length IRES seed panel from direct-RNA measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


NAMED_PATTERNS = {
    "IAPV": "IAPV-IRES-positive control|IAPV-WT|IAPV-WT-positive control",
    "HCV": "HCV-IRES-5p-WT|HCV-IRES-positivecontrol",
    "CrPV": "CrPV-IRES-positivecontrol",
    "SV-A": "Senecavirus_A_DQ641257.1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def select_seed_panel(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    active = [row for row in rows if row["consensus_status"] == "active"]
    selected: list[tuple[str, dict[str, str], str]] = []
    for canonical_name, pattern in NAMED_PATTERNS.items():
        matches = [row for row in active if pattern in row["construct_names"]]
        if not matches:
            raise ValueError(
                f"expected at least one exact named seed for {canonical_name}, found none"
            )
        matches.sort(
            key=lambda row: (
                -(row["source_tables"].count("|") + 1),
                abs(int(row["length"]) - 213),
                row["sequence_id"],
            )
        )
        selected.append((canonical_name, matches[0], "experimentally supported named viral IRES"))

    selected_ids = {row["sequence_id"] for _, row, _ in selected}
    for ires_type in ("IV", "VI b"):
        candidates = [
            row
            for row in active
            if row["sequence_id"] not in selected_ids
            and row["source_tables"] == "S3"
            and row["ires_types"] == ires_type
            and row["is_control"].lower() == "false"
            and as_float(row["te_ires_max"]) is not None
        ]
        candidates.sort(key=lambda row: (-float(row["te_ires_max"]), row["sequence_id"]))
        for row in candidates[:3]:
            canonical_name = row["construct_names"].split("|")[0]
            selected.append(
                (
                    canonical_name,
                    row,
                    f"top-3 direct-RNA-active non-control {ires_type} candidate by TE - IRES",
                )
            )
            selected_ids.add(row["sequence_id"])

    if len(selected) != 10:
        raise ValueError(f"expected 10 frozen seeds, selected {len(selected)}")
    records: list[dict[str, object]] = []
    for canonical_name, row, rationale in selected:
        records.append(
            {
                "seed_id": canonical_name,
                "sequence_id": row["sequence_id"],
                "sequence": row["sequence"],
                "sequence_sha256": row["sequence_sha256"],
                "length": int(row["length"]),
                "construct_names": row["construct_names"],
                "direct_rna_consensus_label": int(row["consensus_label"]),
                "direct_rna_consensus_status": row["consensus_status"],
                "te_ires_max": as_float(row["te_ires_max"]),
                "te_ires_baseline_mean_unique": as_float(row["te_ires_baseline_mean_unique"]),
                "ires_type": row["ires_types"],
                "organism": row["organisms"],
                "source_tables": row["source_tables"],
                "is_positive_control": row["is_control"].lower() == "true",
                "selection_rationale": rationale,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = select_seed_panel(rows)
    payload = {
        "schema_version": 1,
        "frozen_at": "2026-08-21",
        "assay_context": "direct RNA A-cap/G-cap nascent-translation MPRA",
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "selection_policy": {
            "named_seeds": list(NAMED_PATTERNS),
            "additional_seeds": "top three active non-control S3 sequences in each of Type IV and VI b by TE - IRES",
            "labels_used_for_optimization": False,
            "note": "The panel anchors seeded mutation; direct-RNA labels remain evaluation evidence and are not mutation objectives.",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n_seeds": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
