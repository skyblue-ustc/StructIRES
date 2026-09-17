#!/usr/bin/env python3
"""Evaluate static IRESfinder scores on the released repeated test folds."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from ires_design.classification import (
    binary_classification_metrics,
    stratified_bootstrap_intervals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
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
            data = pd.read_csv(
                handle,
                usecols=["fold", "type", "idx", "Sequence", "IRES_class_600"],
            )
    predictions = pd.read_csv(args.predictions)
    required_columns = {"id", "probability"}
    if not required_columns.issubset(predictions.columns):
        missing = required_columns - set(predictions.columns)
        raise ValueError(f"missing prediction columns: {missing}")
    predictions["idx"] = predictions["id"].str.removeprefix("idx_").astype(int)
    if predictions["idx"].duplicated().any():
        raise ValueError("duplicate prediction idx")
    if not predictions["probability"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("probabilities must be in [0, 1]")
    prediction_map = predictions.set_index("idx")["probability"]
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for fold in range(10):
        test = data[(data["fold"] == fold) & (data["type"] == "test")].copy()
        probabilities = test["idx"].map(prediction_map)
        if probabilities.isna().any():
            raise ValueError(f"missing IRESfinder predictions in fold {fold}")
        labels = test["IRES_class_600"].astype(int)
        metric_values = binary_classification_metrics(labels, probabilities, threshold=0.5)
        rows.append(
            {
                "model": "IRESfinder",
                "status": "released_model_rerun_py3_adapter",
                "protocol": "released_native_test_fold",
                "fold": fold,
                **metric_values,
                **stratified_bootstrap_intervals(
                    labels,
                    probabilities,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + fold,
                ),
            }
        )
        for idx, sequence, label, probability in zip(
            test["idx"], test["Sequence"], labels, probabilities
        ):
            canonical = str(sequence).upper().replace("T", "U")
            prediction_rows.append(
                {
                    "fold": fold,
                    "idx": int(idx),
                    "sequence_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
                    "label": int(label),
                    "probability": float(probability),
                    "prediction": int(probability >= 0.5),
                }
            )
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(
        args.output_dir / "predictions.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    metrics = pd.DataFrame(rows)
    summary = {
        "schema_version": 2,
        "experiment": "iresfinder_released_repeated_fold_rerun",
        "scope": "classification-only reproduction",
        "dataset_zip": str(args.dataset_zip.resolve()),
        "dataset_zip_sha256": sha256(args.dataset_zip),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256(args.predictions),
        "decision_threshold": 0.5,
        "n_folds": 10,
        "bootstrap_replicates_per_fold": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "repeated_holdout_non_independence_warning": (
            "Released test folds overlap; fold mean and standard deviation are descriptive, "
            "not independent-fold uncertainty."
        ),
        "metric_means": {
            name: float(metrics[name].mean())
            for name in (
                "auc",
                "aupr",
                "f1",
                "accuracy",
                "sensitivity",
                "specificity",
                "mcc",
                "ece10",
            )
        },
        "metric_standard_deviations": {
            name: float(metrics[name].std(ddof=1))
            for name in (
                "auc",
                "aupr",
                "f1",
                "accuracy",
                "sensitivity",
                "specificity",
                "mcc",
                "ece10",
            )
        },
        "compatibility_note": (
            "Original Perl features and released training data; Python 2.7 orchestration ported "
            "to the installed scikit-learn API."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
