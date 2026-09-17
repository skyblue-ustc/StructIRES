#!/usr/bin/env python3
"""Derive an IAPV positional edit-risk guardrail from public direct-RNA MPRA tiles.

The S1 assay measured overlapping 1--3 nt A-substitution tiles across IAPV.  This script does
not infer activity for new candidates.  It records the observed loss of IRES translation for each
parent position and converts it into a conservative edit penalty: an unmeasured position receives
the median observed risk, so a selector cannot exploit missing coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def substitutions(parent: str, sequence: str) -> list[int]:
    if len(parent) != len(sequence):
        raise ValueError("mutational tile length differs from IAPV parent")
    return [index for index, (left, right) in enumerate(zip(parent, sequence)) if left != right]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--pools", type=Path, nargs="+", required=True)
    parser.add_argument("--position-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.position_output, args.candidate_output, args.manifest):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    rows = list(csv.DictReader(args.measurements.open(encoding="utf-8", newline="")))
    wild_types = [row for row in rows if row["construct_name"] == "IAPV-WT"]
    if not wild_types:
        raise ValueError("IAPV-WT not found")
    parent = wild_types[0]["sequence"]
    wt_te = median(float(row["te_ires"]) for row in wild_types if row["te_ires"])
    tiles = [
        row for row in rows
        if row["source_table"] == "S1" and row["construct_name"].startswith("IAPV_mut")
        and row["te_ires"]
    ]
    if len(tiles) < 100:
        raise ValueError("unexpectedly sparse IAPV direct-RNA tile series")
    losses: dict[int, list[float]] = {index: [] for index in range(len(parent))}
    tile_rows: list[dict[str, object]] = []
    for row in tiles:
        changed = substitutions(parent, row["sequence"])
        loss = max(0.0, wt_te - float(row["te_ires"]))
        for position in changed:
            losses[position].append(loss)
        tile_rows.append({"construct_name": row["construct_name"], "n_changed": len(changed), "loss": loss})
    observed = [median(values) for values in losses.values() if values]
    fallback = float(median(observed))
    positions = []
    for position, values in losses.items():
        positions.append({
            "parent_id": "IAPV", "position_0based": position, "parent_base": parent[position],
            "n_overlapping_mpra_tiles": len(values),
            "median_direct_rna_ires_loss": float(median(values)) if values else "",
            "conservative_edit_risk": float(median(values)) if values else fallback,
            "risk_source": "S1 direct-RNA IAPV A-substitution tiles" if values else "median-imputed from observed S1 tiles",
        })
    risk_by_position = {int(row["position_0based"]): float(row["conservative_edit_risk"]) for row in positions}
    candidates = []
    for pool in args.pools:
        with pool.open(encoding="utf-8") as handle:
            for row in map(json.loads, handle):
                if row["parent_id"] != "IAPV":
                    continue
                changed = [int(pos) for pos in row["metadata"]["mutated_positions_0based"]]
                values = [risk_by_position[position] for position in changed]
                candidates.append({
                    "candidate_id": row["candidate_id"], "parent_id": "IAPV", "run_seed": row["seed"],
                    "n_edits": len(changed), "n_positions_with_direct_mpra_coverage": sum(bool(losses[pos]) for pos in changed),
                    "iapv_mpra_mutational_risk": float(sum(values) / len(values)),
                    "scope": "observed direct-RNA mutational-scan edit-risk; not a new-candidate activity measurement",
                })
    args.position_output.parent.mkdir(parents=True, exist_ok=True)
    with args.position_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(positions[0])); writer.writeheader(); writer.writerows(positions)
    with args.candidate_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0])); writer.writeheader(); writer.writerows(candidates)
    manifest = {
        "schema_version": 1, "experiment": "iapv_direct_rna_mpra_mutational_risk",
        "measurement_input": {"path": str(args.measurements), "sha256": sha256(args.measurements)},
        "pool_inputs": [{"path": str(path), "sha256": sha256(path)} for path in args.pools],
        "parent": "IAPV", "parent_length": len(parent), "wt_te_ires": wt_te,
        "n_direct_rna_tiles": len(tiles), "n_observed_positions": sum(bool(v) for v in losses.values()),
        "unobserved_position_risk": fallback, "n_candidates": len(candidates),
        "scope": "observed direct-RNA MPRA mutational-scan guardrail; not validation of new candidate activity",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
