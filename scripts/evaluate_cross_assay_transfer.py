#!/usr/bin/env python3
"""Evaluate frozen legacy-report classifiers on direct-RNA IRES-TrAPPr labels."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--legacy-data", type=Path, required=True)
    parser.add_argument("--direct-data", type=Path, required=True)
    parser.add_argument("--overlap-data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="composition,kmer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-features", type=int, default=65536)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_direct(path: Path, overlap_path: Path | None) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    overlap: dict[str, dict[str, str]] = {}
    if overlap_path:
        with overlap_path.open(encoding="utf-8", newline="") as handle:
            overlap = {row["direct_sequence_id"]: row for row in csv.DictReader(handle)}
    output = []
    for row in rows:
        if row["consensus_status"] not in {"active", "inactive"}:
            continue
        linked = overlap.get(row["sequence_id"], {})
        output.append(
            {
                **row,
                "label": int(row["consensus_label"]),
                "is_control_bool": row["is_control"].lower() == "true",
                "legacy_contained": linked.get("legacy_contained_in_direct", "false").lower()
                == "true",
                "high_similarity": linked.get("high_similarity_90_80", "false").lower()
                == "true",
            }
        )
    return output


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(error) if total else float("nan")


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
    negatives = labels == 0
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float((~predictions[negatives]).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece10": expected_calibration_error(labels, probabilities),
        "threshold_from_legacy_validation": float(threshold),
    }


def build_estimator(model_name: str, seed: int, max_iter: int):
    if model_name == "composition":
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
    if model_name == "kmer":
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
    raise ValueError(model_name)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    legacy = load_ires_ai_records(args.legacy_data)
    direct = load_direct(args.direct_data, args.overlap_data)
    legacy_sequences = [row.sequence for row in legacy]
    direct_sequences = [str(row["sequence"]) for row in direct]
    legacy_labels = np.asarray([row.label for row in legacy], dtype=np.int8)
    legacy_folds = np.asarray([row.test_fold for row in legacy], dtype=np.int8)
    direct_labels = np.asarray([row["label"] for row in direct], dtype=np.int8)
    models = [name.strip() for name in args.models.split(",") if name.strip()]

    feature_sets: dict[str, tuple[object, object]] = {}
    if "composition" in models:
        feature_sets["composition"] = (
            composition_features(legacy_sequences),
            composition_features(direct_sequences),
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
            vectorizer.transform(legacy_sequences),
            vectorizer.transform(direct_sequences),
        )

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for model_name in models:
        legacy_features, direct_features = feature_sets[model_name]
        fold_predictions = []
        thresholds = []
        for test_fold in range(10):
            validation_fold = (test_fold + 1) % 10
            train_mask = (legacy_folds != test_fold) & (legacy_folds != validation_fold)
            validation_mask = legacy_folds == validation_fold
            estimator = build_estimator(model_name, args.seed + test_fold, args.max_iter)
            estimator.fit(legacy_features[train_mask], legacy_labels[train_mask])
            validation_probability = estimator.predict_proba(legacy_features[validation_mask])[:, 1]
            thresholds.append(best_f1_threshold(legacy_labels[validation_mask], validation_probability))
            fold_predictions.append(estimator.predict_proba(direct_features)[:, 1])
            print(f"{model_name}: trained legacy fold ensemble member {test_fold + 1}/10", flush=True)
        probabilities = np.mean(np.vstack(fold_predictions), axis=0)
        threshold = float(np.mean(thresholds))

        subset_masks = {
            "all_labelled": np.ones(len(direct), dtype=bool),
            "non_control": np.asarray([not row["is_control_bool"] for row in direct]),
            "legacy_contained": np.asarray([row["legacy_contained"] for row in direct]),
            "high_similarity_90_80": np.asarray([row["high_similarity"] for row in direct]),
        }
        for subset_name, mask in subset_masks.items():
            if mask.sum() == 0 or len(np.unique(direct_labels[mask])) < 2:
                continue
            metric_rows.append(
                {
                    "model": model_name,
                    "training_assay": "DNA/lentiviral bicistronic reporter",
                    "evaluation_assay": "direct RNA A-cap/G-cap nascent-translation MPRA",
                    "subset": subset_name,
                    **classification_metrics(direct_labels[mask], probabilities[mask], threshold),
                }
            )
        for row, probability in zip(direct, probabilities):
            prediction_rows.append(
                {
                    "model": model_name,
                    "sequence_id": row["sequence_id"],
                    "sequence": row["sequence"],
                    "label": row["label"],
                    "is_control": row["is_control_bool"],
                    "legacy_contained": row["legacy_contained"],
                    "high_similarity_90_80": row["high_similarity"],
                    "probability": float(probability),
                    "threshold_from_legacy_validation": threshold,
                }
            )

    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "predictions.csv", prediction_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "legacy_to_direct_rna_transfer",
        "legacy_data": str(args.legacy_data),
        "legacy_sha256": file_sha256(args.legacy_data),
        "direct_data": str(args.direct_data),
        "direct_sha256": file_sha256(args.direct_data),
        "overlap_data": str(args.overlap_data) if args.overlap_data else None,
        "models": models,
        "legacy_fold_members": 10,
        "direct_labels_used_for_training_or_thresholding": False,
        "seed": args.seed,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metric_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
