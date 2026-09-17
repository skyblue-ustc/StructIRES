#!/usr/bin/env python3
"""Recompute complete metrics from a frozen repeated-holdout prediction file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ires_design.classification import (
    binary_classification_metrics,
    stratified_bootstrap_intervals,
)


METRICS = ("auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc", "ece10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-folds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
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
    predictions_path = args.run_dir / "predictions.csv.gz"
    source_manifest_path = args.run_dir / "run_manifest.json"
    if not predictions_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError(f"incomplete source run: {args.run_dir}")

    predictions = pd.read_csv(predictions_path)
    required = {"fold", "label", "probability"}
    if missing := required.difference(predictions.columns):
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    expected_folds = [int(value) for value in args.expected_folds.split(",") if value]
    observed_folds = sorted(int(value) for value in predictions["fold"].unique())
    if observed_folds != expected_folds:
        raise ValueError(f"observed folds {observed_folds} differ from expected {expected_folds}")

    rows: list[dict[str, object]] = []
    for fold, frame in predictions.groupby("fold", sort=True):
        labels = frame["label"].to_numpy(dtype=int)
        probabilities = frame["probability"].to_numpy(dtype=float)
        values = binary_classification_metrics(labels, probabilities, threshold=0.5)
        if "prediction" in frame:
            recomputed = (probabilities >= 0.5).astype(int)
            if not np.array_equal(recomputed, frame["prediction"].to_numpy(dtype=int)):
                raise ValueError(f"stored predictions differ from threshold 0.5 in fold {fold}")
        rows.append(
            {
                "model": args.model,
                "status": "released_checkpoint_rerun",
                "protocol": "released_native_repeated_holdout",
                "fold": int(fold),
                **values,
                **stratified_bootstrap_intervals(
                    labels,
                    probabilities,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + int(fold),
                ),
                "source_run_dir": str(args.run_dir.resolve()),
            }
        )

    fold_frame = pd.DataFrame(rows).sort_values("fold")
    aggregate: dict[str, object] = {
        "model": args.model,
        "status": "released_checkpoint_rerun",
        "protocol": "released_native_repeated_holdout",
        "n_available_folds": int(len(fold_frame)),
        "available_folds": ",".join(str(value) for value in fold_frame["fold"]),
        "n_evaluations_with_repeated_membership": int(fold_frame["n"].sum()),
    }
    for name in METRICS:
        aggregate[f"{name}_mean"] = float(fold_frame[name].mean())
        aggregate[f"{name}_std"] = float(fold_frame[name].std(ddof=1))

    fold_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    manifest = {
        "schema_version": 1,
        "experiment": "repeated_binary_prediction_summary",
        "model": args.model,
        "source_run_dir": str(args.run_dir.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_predictions_sha256": sha256(predictions_path),
        "available_folds": expected_folds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "aggregation": "unweighted mean and sample standard deviation across repeated holdouts",
        "independence_warning": (
            "Released test folds overlap; fold mean and standard deviation are descriptive, "
            "not independent-fold uncertainty."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
