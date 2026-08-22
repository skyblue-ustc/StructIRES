#!/usr/bin/env python3
"""Independently verify and summarize a matched native StructIRES pair.

The paired sequence/contact runs must have identical source-fold provenance and
an untouched test set.  This script re-computes all reported test metrics from
saved probabilities before producing a compact table; it never reuses a stored
number as evidence for a comparison.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-run", type=Path, required=True)
    parser.add_argument("--contact-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def metrics(label: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
    prediction = score >= threshold
    negative = label == 0
    ece = 0.0
    for low, high in zip(np.linspace(0., .9, 10), np.linspace(.1, 1., 10)):
        selected = (score >= low) & ((score <= high) if high == 1. else (score < high))
        if selected.any():
            ece += float(selected.mean() * abs(score[selected].mean() - label[selected].mean()))
    return {
        "auc": float(roc_auc_score(label, score)),
        "aupr": float(average_precision_score(label, score)),
        "f1": float(f1_score(label, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(label, prediction)),
        "sensitivity": float(prediction[label == 1].mean()),
        "specificity": float((~prediction[negative]).mean()),
        "mcc": float(matthews_corrcoef(label, prediction)),
        "ece10": float(ece),
        "threshold": float(threshold),
    }


def load_run(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    stored = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    threshold = float(stored["native_validation_selected"]["threshold"])
    with gzip.open(path / "test_predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    index = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    label = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    score = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    if not (len(index) and len(index) == len(np.unique(index))):
        raise ValueError(f"invalid or duplicate test indices in {path}")
    calculated = metrics(label, score, threshold)
    for key, value in calculated.items():
        if not np.isclose(value, float(stored["native_validation_selected"][key]), rtol=0., atol=1e-12):
            raise ValueError(f"stored metric mismatch for {path.name}: {key}")
    return manifest, index, label, score


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    seq_manifest, seq_index, label, seq_score = load_run(args.sequence_run)
    contact_manifest, contact_index, contact_label, contact_score = load_run(args.contact_run)
    shared = ("experiment", "fold", "seed", "dataset_sha256", "rnafm_base_sha256", "validation_fraction",
              "test_labels_used_for_selection", "selection", "training_objective", "mask_probability",
              "epochs_requested", "tokens_per_batch")
    mismatch = {key: (seq_manifest.get(key), contact_manifest.get(key)) for key in shared
                if seq_manifest.get(key) != contact_manifest.get(key)}
    if mismatch:
        raise ValueError(f"runs are not matched: {mismatch}")
    if not (np.array_equal(seq_index, contact_index) and np.array_equal(label, contact_label)):
        raise ValueError("runs do not share an identical test set and ordering")
    seq_threshold = json.loads((args.sequence_run / "metrics.json").read_text())["native_validation_selected"]["threshold"]
    contact_threshold = json.loads((args.contact_run / "metrics.json").read_text())["native_validation_selected"]["threshold"]
    sequence = metrics(label, seq_score, float(seq_threshold))
    contact = metrics(label, contact_score, float(contact_threshold))
    args.output_dir.mkdir(parents=True)
    fields = ("model", "auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc", "ece10", "threshold")
    with (args.output_dir / "paired_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow({"model": "author_style_sequence_only", **sequence})
        writer.writerow({"model": "structires_mfe_contact_fusion", **contact})
    delta = {key: float(contact[key] - sequence[key]) for key in sequence if key != "threshold"}
    report = {"schema_version": 1, "n_test": int(len(label)), "sequence_run": str(args.sequence_run),
              "contact_run": str(args.contact_run), "shared_protocol": {key: seq_manifest[key] for key in shared},
              "sequence_only": sequence, "contact_fusion": contact, "contact_minus_sequence": delta,
              "test_metrics_independently_recomputed": True}
    (args.output_dir / "paired_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
