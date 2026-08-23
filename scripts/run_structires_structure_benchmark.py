#!/usr/bin/env python3
"""Leakage-aware IRES classification with thermodynamic ensemble features.

This utility deliberately separates feature extraction from fitting.  ViennaRNA
is available in the isolated ``ires-structure`` environment while scikit-learn
is available in ``rfamllama``; a versioned feature cache makes that split
explicit and avoids modifying either environment.  It does *not* use test
labels to construct features, select a threshold, or choose a model.

The first-pass model is deliberately lightweight: it establishes whether
global thermodynamic / ensemble features carry signal under the frozen 90%
identity split before any expensive foundation-model fine-tuning is attempted.
It must not be described as a foundation-model result.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ires_design.prediction import composition_features, load_ires_ai_records
from ires_design.structure import fold_ensemble, paired_fraction


FEATURE_NAMES = (
    "mfe_per_nt",
    "ensemble_free_energy_per_nt",
    "centroid_distance_per_nt",
    "ensemble_diversity_per_nt",
    "mfe_paired_fraction",
    "paired_probability_mean",
    "paired_probability_std",
    "paired_probability_min",
    "paired_probability_max",
    "paired_probability_q05",
    "paired_probability_q25",
    "paired_probability_q50",
    "paired_probability_q75",
    "paired_probability_q95",
    "paired_probability_lag1_autocorrelation",
    "paired_probability_lag2_autocorrelation",
    "paired_probability_lag4_autocorrelation",
    "paired_probability_lag8_autocorrelation",
    "paired_fraction_ge_020",
    "paired_fraction_ge_050",
    "paired_fraction_ge_080",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("featurize", "evaluate"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--hash-features", type=int, default=65536)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_records(dataset: Path, assignments: Path | None):
    if assignments is None:
        records = load_ires_ai_records(dataset)
        records.sort(key=lambda record: record.sequence_id)
        return records, None
    with gzip.open(assignments, "rt", encoding="utf-8", newline="") as handle:
        assignment_rows = list(csv.DictReader(handle))
    assignment_by_id = {row["sequence_id"]: row for row in assignment_rows}
    if len(assignment_by_id) != len(assignment_rows):
        raise ValueError("duplicate sequence IDs in locked assignments")
    records = [record for record in load_ires_ai_records(dataset) if record.sequence_id in assignment_by_id]
    if len(records) != len(assignment_rows):
        raise ValueError("assignment IDs do not match canonical dataset records")
    records.sort(key=lambda record: record.sequence_id)
    for record in records:
        expected = assignment_by_id[record.sequence_id]["sequence_sha256"]
        observed = hashlib.sha256(record.sequence.encode("ascii")).hexdigest()
        if observed != expected:
            raise ValueError(f"sequence hash mismatch for {record.sequence_id}")
    return records, assignment_by_id


def _safe_autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    left, right = values[:-lag], values[lag:]
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def fold_feature_row(sequence: str) -> np.ndarray:
    result = fold_ensemble(sequence)
    paired = np.asarray(result.paired_probabilities, dtype=np.float64)
    length = max(len(sequence), 1)
    values = [
        result.mfe_kcal_mol / length,
        result.ensemble_free_energy_kcal_mol / length,
        result.centroid_distance / length,
        result.ensemble_diversity / length,
        paired_fraction(result.mfe_structure),
        float(np.mean(paired)),
        float(np.std(paired)),
        float(np.min(paired)),
        float(np.max(paired)),
        *(float(np.quantile(paired, quantile)) for quantile in (0.05, 0.25, 0.50, 0.75, 0.95)),
        *(_safe_autocorrelation(paired, lag) for lag in (1, 2, 4, 8)),
        *(float(np.mean(paired >= threshold)) for threshold in (0.20, 0.50, 0.80)),
    ]
    if len(values) != len(FEATURE_NAMES) or not np.all(np.isfinite(values)):
        raise RuntimeError("invalid thermodynamic feature row")
    return np.asarray(values, dtype=np.float32)


def featurize(args: argparse.Namespace) -> int:
    records, _ = load_locked_records(args.dataset, args.assignments)
    expected_ids = [record.sequence_id for record in records]
    if args.feature_cache.exists():
        # A foreground job may be reclaimed after atomically finishing the NPZ
        # but before it writes the companion manifest.  Reuse only an exact,
        # readable cache; never overwrite an ambiguous partial artifact.
        existing = np.load(args.feature_cache, allow_pickle=False)
        if (list(existing["sequence_ids"].astype(str)) != expected_ids
                or existing["features"].shape != (len(records), len(FEATURE_NAMES))
                or list(existing["feature_names"].astype(str)) != list(FEATURE_NAMES)
                or not np.all(np.isfinite(existing["features"]))):
            raise FileExistsError(
                f"existing feature cache is incomplete or incompatible; isolate it before retrying: {args.feature_cache}"
            )
        features = np.asarray(existing["features"], dtype=np.float32)
        print(f"reusing verified feature cache: {args.feature_cache}", flush=True)
    else:
        sequences = [record.sequence for record in records]
        print(f"folding {len(sequences)} locked sequences with {args.workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = []
            for index, row in enumerate(executor.map(fold_feature_row, sequences, chunksize=64), start=1):
                rows.append(row)
                if index % 1000 == 0 or index == len(sequences):
                    print(f"folded {index}/{len(sequences)}", flush=True)
        features = np.vstack(rows).astype(np.float32, copy=False)
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.feature_cache,
            sequence_ids=np.asarray(expected_ids),
            features=features,
            feature_names=np.asarray(FEATURE_NAMES),
        )
    metadata = {
        "schema_version": 1,
        "feature_family": "ViennaRNA thermodynamic and ensemble summaries",
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "assignments_path": str(args.assignments) if args.assignments else None,
        "assignments_sha256": sha256(args.assignments) if args.assignments else None,
        "n_records": len(records),
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_generation_uses_labels": False,
        "note": "Global sequence features only; parent-relative anchors belong to the separate seeded-design task.",
    }
    args.feature_cache.with_suffix(args.feature_cache.suffix + ".manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"feature_cache": str(args.feature_cache), "shape": list(features.shape)}, indent=2))
    return 0


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, recall_score, roc_auc_score

    prediction = probabilities >= threshold
    negative = labels == 0
    specificity = float((~prediction[negative]).mean()) if negative.any() else float("nan")
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for index in range(10):
        upper = probabilities <= edges[index + 1] if index == 9 else probabilities < edges[index + 1]
        mask = (probabilities >= edges[index]) & upper
        if mask.any():
            ece += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return {
        "n": int(len(labels)), "n_positive": int(labels.sum()), "prevalence": float(labels.mean()),
        "auc": float(roc_auc_score(labels, probabilities)), "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, prediction, zero_division=0)), "accuracy": float(accuracy_score(labels, prediction)),
        "sensitivity": float(recall_score(labels, prediction, zero_division=0)), "specificity": specificity,
        "mcc": float(matthews_corrcoef(labels, prediction)), "ece10": float(ece), "threshold_from_validation": float(threshold),
    }


def bootstrap(labels: np.ndarray, probabilities: np.ndarray, *, replicates: int, seed: int) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    auc_values, aupr_values = np.empty(replicates), np.empty(replicates)
    for index in range(replicates):
        sample = np.concatenate((rng.choice(positives, len(positives), replace=True), rng.choice(negatives, len(negatives), replace=True)))
        auc_values[index] = roc_auc_score(labels[sample], probabilities[sample])
        aupr_values[index] = average_precision_score(labels[sample], probabilities[sample])
    return {"auc_ci_low": float(np.quantile(auc_values, .025)), "auc_ci_high": float(np.quantile(auc_values, .975)), "aupr_ci_low": float(np.quantile(aupr_values, .025)), "aupr_ci_high": float(np.quantile(aupr_values, .975))}


def evaluate(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        raise ValueError("--output-dir is required in evaluate mode")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output_dir}")
    if args.assignments is None:
        raise ValueError("--assignments is required in evaluate mode")
    records, assignments = load_locked_records(args.dataset, args.assignments)
    assert assignments is not None
    cache = np.load(args.feature_cache, allow_pickle=False)
    identifiers = list(cache["sequence_ids"].astype(str))
    expected = [record.sequence_id for record in records]
    if identifiers != expected:
        raise ValueError("feature cache IDs do not match locked record order")
    structure = np.asarray(cache["features"], dtype=np.float32)
    if list(cache["feature_names"].astype(str)) != list(FEATURE_NAMES):
        raise ValueError("unexpected feature-cache schema")

    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray([record.label for record in records], dtype=np.int8)
    split = np.asarray([assignments[record.sequence_id]["split"] for record in records])
    train, validation, test = split == "train", split == "validation", split == "test"
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(3, 6), n_features=args.hash_features, alternate_sign=False, norm="l2", lowercase=False, dtype=np.float32)
    kmer = vectorizer.transform([record.sequence for record in records])
    composition = composition_features([record.sequence for record in records])
    standardized_structure = StandardScaler().fit_transform(structure[train])
    # Refit from train only for every matrix: this prevents validation/test distribution leakage.
    structure_all = StandardScaler().fit(structure[train]).transform(structure)
    composition_all = StandardScaler().fit(composition[train]).transform(composition)
    matrices = {
        "sequence_composition": csr_matrix(composition_all),
        "structure_ensemble": csr_matrix(structure_all),
        "composition_plus_structure": hstack((csr_matrix(composition_all), csr_matrix(structure_all)), format="csr"),
        "kmer_sequence": kmer,
        "kmer_plus_structure": hstack((kmer, csr_matrix(structure_all)), format="csr"),
    }
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    rows, prediction_rows = [], []
    for model_index, (model_name, matrix) in enumerate(matrices.items()):
        probabilities, thresholds = [], []
        for seed in seeds:
            estimator = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed, solver="liblinear")
            estimator.fit(matrix[train], labels[train])
            validation_probability = estimator.predict_proba(matrix[validation])[:, 1]
            thresholds.append(best_f1_threshold(labels[validation], validation_probability))
            probabilities.append(estimator.predict_proba(matrix[test])[:, 1])
        probability = np.mean(np.vstack(probabilities), axis=0)
        threshold = float(np.mean(thresholds))
        row = {"model": model_name, "protocol": "locked_hamming90_len174_70_15_15", "n_seeds": len(seeds), **metrics(labels[test], probability, threshold), **bootstrap(labels[test], probability, replicates=args.bootstrap_replicates, seed=seeds[0] + model_index), "bootstrap_replicates": args.bootstrap_replicates}
        rows.append(row)
        for index, score in zip(np.flatnonzero(test), probability):
            record = records[int(index)]
            prediction_rows.append({"model": model_name, "sequence_id": record.sequence_id, "cluster_id": assignments[record.sequence_id]["cluster_id"], "label": record.label, "probability": float(score), "threshold_from_validation": threshold})
        print(f"{model_name}: AUC={row['auc']:.4f} AUPR={row['aupr']:.4f} F1={row['f1']:.4f}", flush=True)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with gzip.open(args.output_dir / "predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    manifest = {"schema_version": 1, "experiment": "structires_lightweight_sequence_structure_classifier", "status": "exploratory_project_owned_baseline", "dataset_path": str(args.dataset), "dataset_sha256": sha256(args.dataset), "assignments_path": str(args.assignments), "assignments_sha256": sha256(args.assignments), "feature_cache": str(args.feature_cache), "feature_cache_sha256": sha256(args.feature_cache), "n_train": int(train.sum()), "n_validation": int(validation.sum()), "n_test": int(test.sum()), "seeds": seeds, "test_labels_used_for_feature_generation_thresholding_or_model_choice": False, "note": "This run does not use foundation-model embeddings; it is a low-cost structure-signal gate before a foundation-model fusion run."}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()
    return featurize(args) if args.mode == "featurize" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
