#!/usr/bin/env python3
"""Build a provenance-checked summary of released RNA-FM native fold reruns.

The upstream benchmark uses ten repeated 90/10 holdouts, not independent folds.
This script therefore reports per-holdout stratified-bootstrap intervals and
mean +/- standard deviation across available holdouts without treating their
records as an independent pooled test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


METRIC_NAMES = (
    "auc",
    "aupr",
    "f1",
    "accuracy",
    "sensitivity",
    "specificity",
    "mcc",
    "ece10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if selected.any():
            value += float(selected.mean()) * abs(
                float(labels[selected].mean()) - float(probabilities[selected].mean())
            )
    return value


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    negatives = labels == 0
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float((predictions[negatives] == 0).mean()),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "ece10": expected_calibration_error(labels, probabilities),
    }


def stratified_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("stratified bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(replicates, dtype=float) for name in METRIC_NAMES}
    for replicate in range(replicates):
        indices = np.concatenate(
            (
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            )
        )
        values = compute_metrics(labels[indices], probabilities[indices])
        for name in METRIC_NAMES:
            samples[name][replicate] = values[name]
    intervals: dict[str, float] = {}
    for name in METRIC_NAMES:
        intervals[f"{name}_ci_low"] = float(np.quantile(samples[name], 0.025))
        intervals[f"{name}_ci_high"] = float(np.quantile(samples[name], 0.975))
    return intervals


def load_run(run_dir: Path) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    manifest_path = run_dir / "run_manifest.json"
    predictions_path = run_dir / "predictions.csv.gz"
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"incomplete RNA-FM run directory: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "released_ires_rnafm_native_checkpoint_rerun":
        raise ValueError(f"unexpected experiment in {manifest_path}")
    predictions = pd.read_csv(predictions_path)
    required = {"fold", "label", "probability", "prediction"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"missing prediction columns in {predictions_path}: {sorted(missing)}")
    audit = {
        "run_dir": str(run_dir.resolve()),
        "run_manifest_sha256": sha256(manifest_path),
        "predictions_sha256": sha256(predictions_path),
        "dataset_zip_sha256": manifest["dataset_zip_sha256"],
    }
    return manifest, predictions, audit


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    fold_rows: list[dict[str, object]] = []
    input_audits: list[dict[str, object]] = []
    seen_folds: set[int] = set()
    dataset_hash: str | None = None
    for run_index, run_dir in enumerate(args.run_dir):
        manifest, predictions, audit = load_run(run_dir)
        if dataset_hash is None:
            dataset_hash = str(manifest["dataset_zip_sha256"])
        elif dataset_hash != str(manifest["dataset_zip_sha256"]):
            raise ValueError("RNA-FM run directories use different dataset archives")
        checkpoint_by_fold = {
            int(row["fold"]): row for row in manifest.get("checkpoints", [])
        }
        for fold_value, frame in predictions.groupby("fold", sort=True):
            fold = int(fold_value)
            if fold in seen_folds:
                raise ValueError(f"fold {fold} occurs in more than one input run")
            seen_folds.add(fold)
            labels = frame["label"].to_numpy(dtype=int)
            probabilities = frame["probability"].to_numpy(dtype=float)
            if not np.isin(labels, [0, 1]).all():
                raise ValueError(f"non-binary labels in fold {fold}")
            if not np.isfinite(probabilities).all() or not (
                (probabilities >= 0.0) & (probabilities <= 1.0)
            ).all():
                raise ValueError(f"invalid probabilities in fold {fold}")
            recomputed_predictions = (probabilities >= 0.5).astype(int)
            if not np.array_equal(
                recomputed_predictions, frame["prediction"].to_numpy(dtype=int)
            ):
                raise ValueError(f"stored predictions differ from threshold 0.5 in fold {fold}")
            checkpoint = checkpoint_by_fold.get(fold)
            if checkpoint is None:
                raise ValueError(f"checkpoint provenance missing for fold {fold}")
            fold_rows.append(
                {
                    "model": "IRES-RNAFM",
                    "status": "released_checkpoint_rerun",
                    "protocol": "released_native_repeated_holdout",
                    "fold": fold,
                    "n": int(len(labels)),
                    "n_positive": int(labels.sum()),
                    "positive_fraction": float(labels.mean()),
                    "decision_threshold": 0.5,
                    **compute_metrics(labels, probabilities),
                    **stratified_bootstrap(
                        labels,
                        probabilities,
                        args.bootstrap_replicates,
                        args.seed + 1000 * run_index + fold,
                    ),
                    "bootstrap_replicates": args.bootstrap_replicates,
                    "checkpoint_sha256": checkpoint["sha256"],
                    "source_run_dir": str(run_dir.resolve()),
                    "predictions_sha256": audit["predictions_sha256"],
                }
            )
        input_audits.append(audit)

    fold_frame = pd.DataFrame(fold_rows).sort_values("fold")
    aggregate: dict[str, object] = {
        "model": "IRES-RNAFM",
        "status": "released_checkpoint_rerun",
        "protocol": "released_native_repeated_holdout",
        "n_available_folds": int(len(fold_frame)),
        "available_folds": ",".join(str(value) for value in fold_frame["fold"]),
        "n_evaluations_with_repeated_membership": int(fold_frame["n"].sum()),
    }
    for name in METRIC_NAMES:
        aggregate[f"{name}_mean"] = float(fold_frame[name].mean())
        aggregate[f"{name}_std"] = (
            float(fold_frame[name].std(ddof=1)) if len(fold_frame) > 1 else 0.0
        )

    fold_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    pd.DataFrame(input_audits).to_csv(args.output_dir / "input_audit.csv", index=False)
    output_manifest = {
        "schema_version": 1,
        "experiment": "released_ires_rnafm_native_fold_summary",
        "dataset_zip_sha256": dataset_hash,
        "available_folds": sorted(seen_folds),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "aggregation": "unweighted mean and sample standard deviation across available holdouts",
        "independence_warning": (
            "The released folds are repeated 90/10 holdouts with overlapping test membership; "
            "their evaluations are not independent and are not pooled for a test-set CI."
        ),
        "input_runs": input_audits,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
