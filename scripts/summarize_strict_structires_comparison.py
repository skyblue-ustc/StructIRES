#!/usr/bin/env python3
"""Verify and aggregate the locked strict StructIRES seed comparison.

The inputs are final, already-selected run directories produced by
``train_structires_rnafm.py``.  This script never loads checkpoints or trains
models: it recomputes each reported test metric from the prediction ledgers,
checks that both arms use the same locked split, and reports seed-wise paired
differences plus mean +/- sample standard deviation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ("auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc", "ece10")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--fusion-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve
    precision, recall, values = precision_recall_curve(labels, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(values[int(np.nanargmax(f1))]) if len(values) else .5


def metrics(labels: np.ndarray, probability: np.ndarray, value: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
    prediction = probability >= value
    negative = labels == 0
    ece = 0.0
    for low, high in zip(np.linspace(0.0, .9, 10), np.linspace(.1, 1.0, 10)):
        selected = (probability >= low) & ((probability <= high) if high == 1.0 else (probability < high))
        if selected.any():
            ece += float(selected.mean() * abs(probability[selected].mean() - labels[selected].mean()))
    return {
        "auc": float(roc_auc_score(labels, probability)), "aupr": float(average_precision_score(labels, probability)),
        "f1": float(f1_score(labels, prediction, zero_division=0)), "accuracy": float(accuracy_score(labels, prediction)),
        "sensitivity": float(prediction[labels == 1].mean()), "specificity": float((~prediction[negative]).mean()),
        "mcc": float(matthews_corrcoef(labels, prediction)), "ece10": float(ece), "threshold": float(value),
    }


def read_run(path: Path, expected_variant: str) -> tuple[int, dict, dict[str, float], list[dict[str, object]]]:
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("variant") != expected_variant or manifest.get("test_evaluation_skipped"):
        raise ValueError(f"not a final {expected_variant} test run: {path}")
    rows = list(csv.DictReader((path / "test_metrics.csv").open(encoding="utf-8", newline="")))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one seed in {path}")
    stored = {key: float(rows[0][key]) for key in (*METRICS, "threshold")}
    seed = int(rows[0]["seed"])
    with gzip.open(path / "test_predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.int64)
    probability = np.asarray([float(row["probability"]) for row in predictions], dtype=np.float64)
    recomputed = metrics(labels, probability, stored["threshold"])
    for name in (*METRICS, "threshold"):
        if not np.isclose(stored[name], recomputed[name], atol=1e-12, rtol=0.0):
            raise ValueError(f"metric mismatch in {path}: {name}")
    return seed, manifest, stored, predictions


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if len(args.sequence_dirs) != len(args.fusion_dirs):
        raise ValueError("both arms require the same number of seed directories")
    sequence = [read_run(path, "sequence_only") for path in args.sequence_dirs]
    fusion = [read_run(path, "gated_profile") for path in args.fusion_dirs]
    sequence_by_seed, fusion_by_seed = {row[0]: row for row in sequence}, {row[0]: row for row in fusion}
    if set(sequence_by_seed) != set(fusion_by_seed):
        raise ValueError("sequence and fusion seed sets differ")
    reference = sequence[0][1]
    invariant = ("dataset_sha256", "assignments_sha256", "rnafm_base_sha256", "epochs", "unfreeze_last_layers",
                 "pooling", "mask_prob_training_only", "mlm_loss_weight", "cls_loss_weight")
    for _, manifest, _, _ in (*sequence, *fusion):
        mismatch = {key: (reference.get(key), manifest.get(key)) for key in invariant
                    if reference.get(key) != manifest.get(key)}
        if mismatch: raise ValueError(f"unmatched locked protocol: {mismatch}")
    for seed in sorted(sequence_by_seed):
        sequence_index_label = [(row["sample_index"], row["label"]) for row in sequence_by_seed[seed][3]]
        fusion_index_label = [(row["sample_index"], row["label"]) for row in fusion_by_seed[seed][3]]
        if sequence_index_label != fusion_index_label:
            raise ValueError(f"test prediction ledgers are not paired for seed {seed}")
    args.output_dir.mkdir(parents=True)
    rows = []
    for seed in sorted(sequence_by_seed):
        for arm, (_, _, report, _) in (("sequence_only", sequence_by_seed[seed]), ("gated_profile", fusion_by_seed[seed])):
            rows.append({"seed": seed, "model": arm, **{metric: report[metric] for metric in METRICS}})
    with (args.output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "model", *METRICS)); writer.writeheader(); writer.writerows(rows)
    summary = {"schema_version": 1, "n_seeds": len(sequence_by_seed), "seeds": sorted(sequence_by_seed),
               "test_metrics_independently_recomputed": True,
               "sequence_only": {metric: mean_std([sequence_by_seed[seed][2][metric] for seed in sorted(sequence_by_seed)]) for metric in METRICS},
               "gated_profile": {metric: mean_std([fusion_by_seed[seed][2][metric] for seed in sorted(fusion_by_seed)]) for metric in METRICS},
               "gated_minus_sequence": {metric: mean_std([fusion_by_seed[seed][2][metric] - sequence_by_seed[seed][2][metric] for seed in sorted(sequence_by_seed)]) for metric in METRICS}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "experiment": "strict_structires_locked_seed_summary", "seeds": summary["seeds"],
                "shared_protocol": {key: reference[key] for key in invariant},
                "input_run_manifest_sha256": {"sequence_only": [sha256(path / "run_manifest.json") for path in args.sequence_dirs], "gated_profile": [sha256(path / "run_manifest.json") for path in args.fusion_dirs]},
                "test_metrics_independently_recomputed": True}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
