#!/usr/bin/env python3
"""Aggregate independently verified matched native StructIRES folds.

Inputs are output directories made by ``summarize_native_structires_pair.py``.
The script intentionally aggregates only the already independently recomputed
paired summaries; it does not read checkpoints or re-select any model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc", "ece10")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_pair(path: Path) -> dict:
    report = json.loads((path / "paired_summary.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or not report.get("test_metrics_independently_recomputed"):
        raise ValueError(f"not an independently verified pair: {path}")
    protocol = report["shared_protocol"]
    if protocol.get("test_labels_used_for_selection") is not False:
        raise ValueError(f"test-selected pair is not eligible: {path}")
    for key in ("sequence_only", "contact_fusion", "contact_minus_sequence"):
        if set(METRICS) - set(report[key]):
            raise ValueError(f"missing metrics in {path}: {key}")
    return report


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    reports = [load_pair(path) for path in args.pair_dirs]
    folds = [int(report["shared_protocol"]["fold"]) for report in reports]
    if len(folds) != len(set(folds)):
        raise ValueError(f"duplicate folds: {folds}")
    reports.sort(key=lambda report: int(report["shared_protocol"]["fold"]))
    reference = reports[0]["shared_protocol"]
    shared_keys = ("experiment", "seed", "dataset_sha256", "rnafm_base_sha256", "validation_fraction",
                   "selection", "training_objective", "mask_probability", "epochs_requested", "tokens_per_batch")
    for report in reports[1:]:
        current = report["shared_protocol"]
        mismatch = {key: (reference.get(key), current.get(key)) for key in shared_keys
                    if reference.get(key) != current.get(key)}
        if mismatch:
            raise ValueError(f"unmatched pair protocols: {mismatch}")
    args.output_dir.mkdir(parents=True)
    columns = ("fold", "n_test", "model", *METRICS)
    with (args.output_dir / "per_fold_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for report in reports:
            fold = int(report["shared_protocol"]["fold"])
            for model, key in (("author_style_sequence_only", "sequence_only"),
                               ("structires_mfe_contact_fusion", "contact_fusion")):
                writer.writerow({"fold": fold, "n_test": report["n_test"], "model": model,
                                 **{metric: report[key][metric] for metric in METRICS}})
    summary = {
        "schema_version": 1,
        "n_folds": len(reports),
        "folds": [int(report["shared_protocol"]["fold"]) for report in reports],
        "pair_directories": [str(path) for path in args.pair_dirs],
        "shared_protocol": {key: reference[key] for key in shared_keys},
        "test_metrics_independently_recomputed": True,
        "sequence_only": {metric: mean_std([report["sequence_only"][metric] for report in reports])
                          for metric in METRICS},
        "contact_fusion": {metric: mean_std([report["contact_fusion"][metric] for report in reports])
                           for metric in METRICS},
        "contact_minus_sequence": {metric: mean_std([report["contact_minus_sequence"][metric] for report in reports])
                                     for metric in METRICS},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
