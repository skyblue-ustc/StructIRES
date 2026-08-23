#!/usr/bin/env python3
"""Compute ViennaRNA ensemble metrics for a frozen shared mutation pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ires_design.structure import compare_seed_and_variant, fold_ensemble


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_one(payload: tuple[dict[str, object], dict[str, object]]) -> dict[str, object]:
    candidate, seed = payload
    seed_fold = fold_ensemble(str(seed["sequence"]))
    variant_fold = fold_ensemble(str(candidate["sequence"]))
    comparison = compare_seed_and_variant(seed_fold, variant_fold)
    return {
        "candidate_id": candidate["candidate_id"],
        "parent_id": candidate["parent_id"],
        "run_seed": candidate["seed"],
        "sequence_sha256": hashlib.sha256(str(candidate["sequence"]).encode("ascii")).hexdigest(),
        "length": len(str(candidate["sequence"])),
        "seed_mfe_kcal_mol": seed_fold.mfe_kcal_mol,
        "mfe_kcal_mol": variant_fold.mfe_kcal_mol,
        "ensemble_free_energy_kcal_mol": variant_fold.ensemble_free_energy_kcal_mol,
        "ensemble_diversity": variant_fold.ensemble_diversity,
        **comparison,
        "viennarna_version": variant_fold.viennarna_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    panel = json.loads(args.seed_panel.read_text(encoding="utf-8"))
    seeds = {record["seed_id"]: record for record in panel["records"]}
    with args.input.open(encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle]
    if not candidates:
        raise ValueError("empty candidate pool")
    missing = {row["parent_id"] for row in candidates} - set(seeds)
    if missing:
        raise ValueError(f"candidates refer to absent seeds: {sorted(missing)}")
    payloads = [(row, seeds[row["parent_id"]]) for row in candidates]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        metrics = list(executor.map(fold_one, payloads, chunksize=16))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    manifest = {
        "schema_version": 1,
        "experiment": "viennarna_seeded_mutation_pool_folding",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "seed_panel": str(args.seed_panel),
        "seed_panel_sha256": sha256(args.seed_panel),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "n_candidates": len(metrics),
        "workers": args.workers,
        "viennarna_version": metrics[0]["viennarna_version"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
