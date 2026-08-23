#!/usr/bin/env python3
"""Summarize the IAPV-only public-MPRA edit-risk guardrail ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = (
    "iapv_mpra_mutational_risk", "mfe_abs_delta_kcal_mol", "pairing_profile_l1",
    "mfe_structure_state_identity", "parent_ensemble_anchor_retention",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.paired_output, args.manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    rows = [
        row for row in csv.DictReader(args.input.open(encoding="utf-8", newline=""))
        if row["parent_id"] == "IAPV" and row["method"] in {"structires", "structires_plus_iapv_mpra_guardrail"}
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["run_seed"], row["method"]), []).append(row)
    unit_rows = []
    for (run_seed, method), group in sorted(grouped.items()):
        unit_rows.append({"run_seed": int(run_seed), "method": method, **{
            metric: float(np.mean([float(row[metric]) for row in group])) for metric in METRICS
        }})
    summary = []
    for method in sorted({row["method"] for row in unit_rows}):
        values = [row for row in unit_rows if row["method"] == method]
        summary.append({"method": method, "n_pools": len(values), **{
            f"{metric}_mean": float(np.mean([row[metric] for row in values])) for metric in METRICS
        }, **{
            f"{metric}_std": float(np.std([row[metric] for row in values], ddof=1)) for metric in METRICS
        }})
    base = {row["run_seed"]: row for row in unit_rows if row["method"] == "structires"}
    guard = {row["run_seed"]: row for row in unit_rows if row["method"] == "structires_plus_iapv_mpra_guardrail"}
    differences = []
    rng = np.random.default_rng(20260822)
    for metric in METRICS:
        delta = np.array([guard[seed][metric] - base[seed][metric] for seed in sorted(base)], dtype=float)
        bootstrap = np.mean(rng.choice(delta, size=(10000, len(delta)), replace=True), axis=1)
        differences.append({"metric": metric, "n_paired_pools": len(delta), "guardrail_minus_structires": float(np.mean(delta)),
                            "bootstrap_ci95_low": float(np.quantile(bootstrap, .025)), "bootstrap_ci95_high": float(np.quantile(bootstrap, .975))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with args.paired_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(differences[0])); writer.writeheader(); writer.writerows(differences)
    manifest = {"schema_version": 1, "experiment": "iapv_mpra_guardrail_summary", "input": {"path": str(args.input), "sha256": sha256(args.input)},
                "parent": "IAPV", "n_parent_run_units": 3,
                "scope": "public direct-RNA mutational-scan risk ablation; does not validate generated-candidate activity"}
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "paired": differences}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
