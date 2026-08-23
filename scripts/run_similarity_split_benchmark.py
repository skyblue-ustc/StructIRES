#!/usr/bin/env python3
"""Run lightweight baselines on a frozen cluster-disjoint split."""

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

from run_source_holdout_benchmark import (  # noqa: E402
    best_f1_threshold,
    bootstrap_intervals,
    build_estimator,
    classification_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="composition,kmer")
    parser.add_argument("--seeds", default="42,43,44")
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.assignments, "rt", encoding="utf-8", newline="") as handle:
        assignment_rows = list(csv.DictReader(handle))
    assignment_by_id = {row["sequence_id"]: row for row in assignment_rows}
    records = [
        row for row in load_ires_ai_records(args.dataset) if row.sequence_id in assignment_by_id
    ]
    if len(records) != len(assignment_rows):
        raise ValueError("assignment IDs do not match unique dataset records")
    for record in records:
        expected_hash = assignment_by_id[record.sequence_id]["sequence_sha256"]
        observed_hash = hashlib.sha256(record.sequence.encode("ascii")).hexdigest()
        if expected_hash != observed_hash:
            raise ValueError(f"sequence hash mismatch for {record.sequence_id}")

    sequences = [row.sequence for row in records]
    labels = np.asarray([row.label for row in records], dtype=np.int8)
    splits = np.asarray([assignment_by_id[row.sequence_id]["split"] for row in records])
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]

    feature_sets: dict[str, object] = {}
    if "composition" in models:
        feature_sets["composition"] = composition_features(sequences)
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
        feature_sets["kmer"] = vectorizer.transform(sequences)

    per_seed_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(models):
        features = feature_sets[model]
        test_probabilities: list[np.ndarray] = []
        thresholds: list[float] = []
        for seed in seeds:
            estimator = build_estimator(model, seed, args.max_iter)
            estimator.fit(features[train_mask], labels[train_mask])
            validation_probability = estimator.predict_proba(features[validation_mask])[:, 1]
            threshold = best_f1_threshold(labels[validation_mask], validation_probability)
            test_probability = estimator.predict_proba(features[test_mask])[:, 1]
            row = {
                "model": model,
                "seed": seed,
                **classification_metrics(labels[test_mask], test_probability, threshold),
            }
            per_seed_rows.append(row)
            thresholds.append(threshold)
            test_probabilities.append(test_probability)
            print(
                f"{model} seed={seed} AUC={row['auc']:.4f} "
                f"AUPR={row['aupr']:.4f} F1={row['f1']:.4f}",
                flush=True,
            )

        ensemble_probability = np.mean(np.vstack(test_probabilities), axis=0)
        ensemble_threshold = float(np.mean(thresholds))
        final_rows.append(
            {
                "model": model,
                "protocol": "cluster_disjoint_hamming90_len174_70_15_15",
                "n_seeds": len(seeds),
                **classification_metrics(
                    labels[test_mask], ensemble_probability, ensemble_threshold
                ),
                **bootstrap_intervals(
                    labels[test_mask],
                    ensemble_probability,
                    replicates=args.bootstrap_replicates,
                    seed=seeds[0] + model_index,
                ),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )
        test_indices = np.flatnonzero(test_mask)
        for index, probability in zip(test_indices, ensemble_probability):
            record = records[int(index)]
            assignment = assignment_by_id[record.sequence_id]
            prediction_rows.append(
                {
                    "model": model,
                    "sequence_id": record.sequence_id,
                    "cluster_id": assignment["cluster_id"],
                    "label": record.label,
                    "probability": float(probability),
                    "ensemble_threshold": ensemble_threshold,
                }
            )

    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(args.output_dir / "metrics.csv", final_rows)
    with gzip.open(args.output_dir / "predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "ires_ai_similarity_split_lightweight",
        "dataset_path": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "assignments_path": str(args.assignments),
        "assignments_sha256": file_sha256(args.assignments),
        "models": models,
        "seeds": seeds,
        "n_train": int(train_mask.sum()),
        "n_validation": int(validation_mask.sum()),
        "n_test": int(test_mask.sum()),
        "test_labels_used_for_training_thresholding_or_model_choice": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
