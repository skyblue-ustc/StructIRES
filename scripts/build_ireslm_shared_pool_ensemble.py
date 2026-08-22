#!/usr/bin/env python3
"""Build the published-style IRES-LM score from frozen RNA-FM and UTR-LM runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnafm", type=Path, nargs="+", required=True)
    parser.add_argument("--utrlm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {args.output_dir}")
    rnafm: dict[str, list[float]] = defaultdict(list)
    metadata: dict[str, dict[str, str]] = {}
    rnafm_inputs = []
    for path in args.rnafm:
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        if not rows:
            raise ValueError(f"empty RNA-FM score file: {path}")
        for row in rows:
            candidate_id = row["candidate_id"]
            rnafm[candidate_id].append(float(row["rnafm_probability_fold0"]))
            metadata.setdefault(candidate_id, row)
        rnafm_inputs.append({"path": str(path), "sha256": sha256(path), "n_rows": len(rows)})
    expected_folds = len(args.rnafm)
    incomplete = [candidate_id for candidate_id, values in rnafm.items() if len(values) != expected_folds]
    if incomplete:
        raise ValueError(f"RNA-FM fold coverage incomplete for {len(incomplete)} candidates")
    utr_rows = {row["candidate_id"]: row for row in csv.DictReader(args.utrlm.open(encoding="utf-8", newline=""))}
    if set(utr_rows) != set(rnafm):
        raise ValueError("RNA-FM and UTR-LM candidate ID sets differ")
    output = []
    for candidate_id in sorted(rnafm):
        values = rnafm[candidate_id]
        utr = utr_rows[candidate_id]
        rna_mean = fmean(values)
        utr_mean = float(utr["utrlm_probability_mean"])
        base = metadata[candidate_id]
        output.append(
            {
                "candidate_id": candidate_id,
                "parent_id": base["parent_id"],
                "run_seed": base["run_seed"],
                "sequence_sha256": base["sequence_sha256"],
                "rnafm_probability_mean": rna_mean,
                "rnafm_probability_std": stdev(values),
                "utrlm_probability_mean": utr_mean,
                "utrlm_probability_std": utr["utrlm_probability_std"],
                "ireslm_probability_mean": (rna_mean + utr_mean) / 2.0,
                "n_rnafm_folds": len(values),
                "n_utrlm_folds": int(utr["n_folds"]),
            }
        )
    args.output_dir.mkdir(parents=True)
    score_path = args.output_dir / "ireslm_pool_scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    manifest = {
        "schema_version": 1,
        "experiment": "released_ireslm_ensemble_score_on_frozen_shared_pools",
        "scope": "published-style arithmetic mean of frozen released RNA-FM and UTR-LM classifier ensembles; no direct-RNA labels used for fitting or selection",
        "rnafm_inputs": rnafm_inputs,
        "utrlm_input": {"path": str(args.utrlm), "sha256": sha256(args.utrlm)},
        "combination": "0.5 * mean(RNA-FM folds) + 0.5 * mean(UTR-LM folds)",
        "n_candidates": len(output),
        "score_file_sha256": sha256(score_path),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_candidates": len(output), "mean_ireslm_score": fmean(row["ireslm_probability_mean"] for row in output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
