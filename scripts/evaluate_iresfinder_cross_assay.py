#!/usr/bin/env python3
"""Evaluate frozen IRESfinder scores on labelled direct-RNA IRES-TrAPPr records."""

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
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--overlap-data", type=Path, required=True)
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


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    error = 0.0
    for index in range(10):
        upper = probabilities <= edges[index + 1] if index == 9 else probabilities < edges[index + 1]
        mask = (probabilities >= edges[index]) & upper
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(error)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predicted = probabilities >= 0.5
    negatives = labels == 0
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float((~predicted[negatives]).mean()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece10": expected_calibration_error(labels, probabilities),
        "decision_threshold": 0.5,
    }


def bootstrap(
    labels: np.ndarray, probabilities: np.ndarray, replicates: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    aucs = np.empty(replicates)
    auprs = np.empty(replicates)
    for index in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        aucs[index] = roc_auc_score(labels[sampled], probabilities[sampled])
        auprs[index] = average_precision_score(labels[sampled], probabilities[sampled])
    return {
        "auc_ci_low": float(np.quantile(aucs, 0.025)),
        "auc_ci_high": float(np.quantile(aucs, 0.975)),
        "aupr_ci_low": float(np.quantile(auprs, 0.025)),
        "aupr_ci_high": float(np.quantile(auprs, 0.975)),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    direct = pd.read_csv(args.direct_data)
    direct = direct[direct["consensus_status"].isin(["active", "inactive"])].copy()
    direct["label"] = direct["consensus_label"].astype(int)
    overlap = pd.read_csv(args.overlap_data)
    overlap = overlap.rename(columns={"direct_sequence_id": "sequence_id"})
    direct = direct.merge(
        overlap[["sequence_id", "legacy_contained_in_direct", "high_similarity_90_80"]],
        on="sequence_id",
        how="left",
        validate="one_to_one",
    )
    predictions = pd.read_csv(args.predictions)
    predictions = predictions.rename(columns={"id": "sequence_id"})
    direct = direct.merge(
        predictions[["sequence_id", "probability"]],
        on="sequence_id",
        how="left",
        validate="one_to_one",
    )
    if direct["probability"].isna().any():
        raise ValueError("missing IRESfinder predictions for labelled direct-RNA records")
    subset_masks = {
        "all_labelled": np.ones(len(direct), dtype=bool),
        "non_control": ~direct["is_control"].astype(bool).to_numpy(),
        "legacy_contained": direct["legacy_contained_in_direct"].fillna(False).astype(bool).to_numpy(),
        "high_similarity_90_80": direct["high_similarity_90_80"].fillna(False).astype(bool).to_numpy(),
    }
    rows: list[dict[str, object]] = []
    for offset, (subset, mask) in enumerate(subset_masks.items()):
        labels = direct.loc[mask, "label"].to_numpy(dtype=np.int8)
        probabilities = direct.loc[mask, "probability"].to_numpy(dtype=float)
        if len(np.unique(labels)) < 2:
            continue
        rows.append(
            {
                "model": "IRESfinder",
                "training_assay": "released IRESfinder training set",
                "evaluation_assay": "direct RNA A-cap/G-cap nascent-translation MPRA",
                "subset": subset,
                **metrics(labels, probabilities),
                **bootstrap(labels, probabilities, args.bootstrap_replicates, args.seed + offset),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    direct[["sequence_id", "label", "is_control", "probability"]].to_csv(
        args.output_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "iresfinder_legacy_to_direct_rna_transfer",
        "scope": "classification-only reproduction",
        "direct_data": str(args.direct_data.resolve()),
        "direct_data_sha256": sha256(args.direct_data),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256(args.predictions),
        "overlap_data": str(args.overlap_data.resolve()),
        "overlap_data_sha256": sha256(args.overlap_data),
        "direct_labels_used_for_training_or_thresholding": False,
        "decision_threshold": 0.5,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
