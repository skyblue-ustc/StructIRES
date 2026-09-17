#!/usr/bin/env python3
"""Evaluate legacy IRES classifiers on a completely held-out data source."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ires_design.prediction import composition_features, load_ires_ai_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-source", default="55k")
    parser.add_argument("--test-source", default="IRESite_exp")
    parser.add_argument("--models", default="composition,kmer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-folds", type=int, default=10)
    parser.add_argument("--hash-features", type=int, default=65536)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fold(sequence: str, n_folds: int) -> int:
    digest = hashlib.sha256(sequence.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        upper = probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        mask = (probabilities >= edges[index]) & upper
        if mask.any():
            error += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return error


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = probabilities >= threshold
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece10": expected_calibration_error(labels, probabilities),
        "threshold_from_training_source_validation": float(threshold),
    }


def bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    auc = np.empty(replicates)
    aupr = np.empty(replicates)
    for index in range(replicates):
        sampled = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        auc[index] = roc_auc_score(labels[sampled], probabilities[sampled])
        aupr[index] = average_precision_score(labels[sampled], probabilities[sampled])
    return {
        "auc_ci_low": float(np.quantile(auc, 0.025)),
        "auc_ci_high": float(np.quantile(auc, 0.975)),
        "aupr_ci_low": float(np.quantile(aupr, 0.025)),
        "aupr_ci_high": float(np.quantile(aupr, 0.975)),
    }


def build_estimator(model: str, seed: int, max_iter: int):
    if model == "composition":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=max_iter,
                random_state=seed,
                solver="liblinear",
            ),
        )
    if model == "kmer":
        from sklearn.linear_model import SGDClassifier

        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.05,
            class_weight="balanced",
            max_iter=max_iter,
            tol=1e-4,
            random_state=seed,
        )
    raise ValueError(model)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = load_ires_ai_records(args.dataset)
    train_records = [row for row in records if row.source == args.train_source]
    test_records = [row for row in records if row.source == args.test_source]
    if set(row.label for row in train_records) != {0, 1}:
        raise ValueError("training source must contain both labels")
    if set(row.label for row in test_records) != {0, 1}:
        raise ValueError("test source must contain both labels")

    train_sequences = [row.sequence for row in train_records]
    test_sequences = [row.sequence for row in test_records]
    train_labels = np.asarray([row.label for row in train_records], dtype=np.int8)
    test_labels = np.asarray([row.label for row in test_records], dtype=np.int8)
    validation_folds = np.asarray(
        [stable_fold(row.sequence, args.validation_folds) for row in train_records]
    )
    models = [model.strip() for model in args.models.split(",") if model.strip()]

    feature_sets: dict[str, tuple[object, object]] = {}
    if "composition" in models:
        feature_sets["composition"] = (
            composition_features(train_sequences),
            composition_features(test_sequences),
        )
    if "kmer" in models:
        from sklearn.feature_extraction.text import HashingVectorizer

        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            n_features=args.hash_features,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
            dtype=np.float32,
        )
        feature_sets["kmer"] = (
            vectorizer.transform(train_sequences),
            vectorizer.transform(test_sequences),
        )

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(models):
        train_features, test_features = feature_sets[model]
        probabilities: list[np.ndarray] = []
        thresholds: list[float] = []
        for validation_fold in range(args.validation_folds):
            validation_mask = validation_folds == validation_fold
            fit_mask = ~validation_mask
            estimator = build_estimator(model, args.seed + validation_fold, args.max_iter)
            estimator.fit(train_features[fit_mask], train_labels[fit_mask])
            validation_probability = estimator.predict_proba(train_features[validation_mask])[:, 1]
            thresholds.append(
                best_f1_threshold(train_labels[validation_mask], validation_probability)
            )
            probabilities.append(estimator.predict_proba(test_features)[:, 1])
            print(f"{model}: ensemble member {validation_fold + 1}/{args.validation_folds}", flush=True)
        probability = np.mean(np.vstack(probabilities), axis=0)
        threshold = float(np.mean(thresholds))
        metrics = classification_metrics(test_labels, probability, threshold)
        metric_rows.append(
            {
                "model": model,
                "training_source": args.train_source,
                "test_source": args.test_source,
                **metrics,
                **bootstrap_intervals(
                    test_labels,
                    probability,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + model_index,
                ),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )
        for record, score in zip(test_records, probability):
            prediction_rows.append(
                {
                    "model": model,
                    "sequence_id": record.sequence_id,
                    "source": record.source,
                    "label": record.label,
                    "length": record.length,
                    "probability": float(score),
                    "threshold": threshold,
                }
            )

    write_csv(args.output_dir / "metrics.csv", metric_rows)
    with gzip.open(args.output_dir / "predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "ires_ai_source_holdout_lightweight",
        "dataset_path": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "training_source": args.train_source,
        "test_source": args.test_source,
        "n_training": len(train_records),
        "n_test": len(test_records),
        "models": models,
        "validation_folds_within_training_source": args.validation_folds,
        "test_source_used_for_training_thresholding_or_model_choice": False,
        "seed": args.seed,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metric_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
