#!/usr/bin/env python3
"""Summarize checkpoint-native StructIRES adapter folds with provenance.

The upstream release uses repeated, overlapping 90/10 holdouts.  This script
therefore keeps per-fold bootstrap intervals and reports mean/sample-standard
deviation over available holdouts rather than an invalid pooled test interval.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


METRICS = ("auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc", "ece10")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_values(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    negative = labels == 0
    ece = 0.0
    for low, high in zip(np.linspace(0., .9, 10), np.linspace(.1, 1., 10)):
        selected = (probability >= low) & ((probability <= high) if high == 1. else (probability < high))
        if selected.any():
            ece += float(selected.mean() * abs(probability[selected].mean() - labels[selected].mean()))
    return {
        "auc": float(roc_auc_score(labels, probability)),
        "aupr": float(average_precision_score(labels, probability)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "sensitivity": float(prediction[labels == 1].mean()),
        "specificity": float((~prediction[negative]).mean()),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "ece10": float(ece),
    }


def bootstrap(labels: np.ndarray, probability: np.ndarray, threshold: float, replicates: int, seed: int) -> dict[str, float]:
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("both classes are required for stratified bootstrap")
    rng = np.random.default_rng(seed)
    values = {metric: np.empty(replicates, dtype=float) for metric in METRICS}
    for replicate in range(replicates):
        index = np.concatenate((rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)))
        for metric, value in metric_values(labels[index], probability[index], threshold).items():
            values[metric][replicate] = value
    interval: dict[str, float] = {}
    for metric, value in values.items():
        interval[f"{metric}_ci_low"] = float(np.quantile(value, .025))
        interval[f"{metric}_ci_high"] = float(np.quantile(value, .975))
    return interval


def load_fold(path: Path, replicates: int, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest_path, metrics_path, prediction_path = path / "run_manifest.json", path / "metrics.json", path / "test_predictions.csv.gz"
    if not all(item.is_file() for item in (manifest_path, metrics_path, prediction_path)):
        raise FileNotFoundError(f"incomplete adapter run directory: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "structires_release_checkpoint_zero_initialized_structure_adapter":
        raise ValueError(f"unexpected experiment: {manifest_path}")
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(prediction_path, compression="gzip")
    if {"id", "label", "baseline_probability", "adapter_probability"}.difference(frame):
        raise ValueError(f"prediction columns missing: {prediction_path}")
    labels = frame.label.to_numpy(dtype=np.int64)
    baseline_threshold = float(report["baseline_released_checkpoint"]["threshold"])
    adapter_threshold = float(report["adapter_validation_selected"]["threshold"])
    fold = int(manifest["fold"])
    common = {
        "fold": fold, "n": int(len(frame)), "n_positive": int(labels.sum()),
        "source_run_dir": str(path.resolve()),
        "manifest_sha256": sha256(manifest_path), "metrics_sha256": sha256(metrics_path),
        "predictions_sha256": sha256(prediction_path), "checkpoint_sha256": manifest["checkpoint_sha256"],
    }
    rows: list[dict[str, object]] = []
    for model, column, threshold in (
        ("IRES-RNAFM released checkpoint", "baseline_probability", baseline_threshold),
        ("StructIRES release adapter", "adapter_probability", adapter_threshold),
    ):
        probability = frame[column].to_numpy(dtype=float)
        if not np.isfinite(probability).all() or not ((probability >= 0.) & (probability <= 1.)).all():
            raise ValueError(f"invalid probabilities for {model} fold {fold}")
        rows.append({"model": model, "threshold": threshold, **common, **metric_values(labels, probability, threshold), **bootstrap(labels, probability, threshold, replicates, seed + 1000 * fold + len(rows))})
    audit = {"fold": fold, "run_dir": str(path.resolve()), "manifest_sha256": common["manifest_sha256"], "metrics_sha256": common["metrics_sha256"], "predictions_sha256": common["predictions_sha256"]}
    return rows, audit


def main() -> int:
    args = arguments()
    if args.bootstrap_replicates < 1: raise ValueError("bootstrap-replicates must be positive")
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows: list[dict[str, object]] = []; audits: list[dict[str, object]] = []; seen: set[int] = set()
    for run_dir in args.run_dir:
        fold_rows, audit = load_fold(run_dir, args.bootstrap_replicates, args.seed)
        if audit["fold"] in seen: raise ValueError(f"duplicate fold {audit['fold']}")
        seen.add(audit["fold"]); rows.extend(fold_rows); audits.append(audit)
    fold_metrics = pd.DataFrame(rows).sort_values(["model", "fold"])
    aggregate_rows: list[dict[str, object]] = []
    for model, frame in fold_metrics.groupby("model", sort=True):
        row: dict[str, object] = {"model": model, "n_available_folds": int(len(frame)), "available_folds": ",".join(map(str, frame.fold.tolist())), "protocol": "released_native_repeated_holdout_validation_selected_adapter"}
        for metric in METRICS:
            row[metric] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1)) if len(frame) > 1 else 0.0
        aggregate_rows.append(row)
    args.output_dir.mkdir(parents=True)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output_dir / "input_audit.csv", index=False)
    output_manifest = {"schema_version": 1, "experiment": "structires_release_adapter_fold_summary", "available_folds": sorted(seen), "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.seed, "aggregation": "unweighted mean and sample standard deviation across overlapping released holdouts", "independence_warning": "The upstream 90/10 repeated holdouts overlap and are not independent; no pooled held-out CI is reported.", "input_runs": audits}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"folds": sorted(seen), "rows": len(fold_metrics), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
