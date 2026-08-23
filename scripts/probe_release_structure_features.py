#!/usr/bin/env python3
"""Validation-clean structure-only diagnostic on a released IRES-RNAFM fold.

This is deliberately a lightweight ablation, not a proposed StructIRES model.
It asks whether the existing label-free 21-dimensional ViennaRNA feature cache
contains independent IRES-classification signal under the same train/
validation/test protocol as the checkpoint-native adapters.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--c-values", type=float, nargs="+", default=(.01, .1, 1., 10., 100.))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fold_rows(path: Path, fold: int):
    import pandas as pd
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv") and "__MACOSX" not in name]
        if len(names) != 1:
            raise ValueError("expected one canonical CSV in dataset archive")
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle)
    frame = frame.loc[frame.fold == fold, ["ID", "IRES_class_600", "type"]].copy()
    if frame.ID.duplicated().any() or len(frame) != 46774:
        raise ValueError("unexpected fold record count or duplicate IDs")
    frame.rename(columns={"IRES_class_600": "label"}, inplace=True)
    return frame


def best_f1_threshold(label: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = precision_recall_curve(label, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(threshold[int(np.nanargmax(f1))]) if len(threshold) else .5


def metrics(label: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
    prediction = probability >= threshold
    negative = label == 0
    ece = 0.
    for low, high in zip(np.linspace(0., .9, 10), np.linspace(.1, 1., 10)):
        mask = (probability >= low) & ((probability <= high) if high == 1. else (probability < high))
        if mask.any():
            ece += float(mask.mean() * abs(probability[mask].mean() - label[mask].mean()))
    return {
        "auc": float(roc_auc_score(label, probability)),
        "aupr": float(average_precision_score(label, probability)),
        "f1": float(f1_score(label, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(label, prediction)),
        "sensitivity": float(prediction[label == 1].mean()),
        "specificity": float((~prediction[negative]).mean()),
        "mcc": float(matthews_corrcoef(label, prediction)),
        "ece10": float(ece),
        "threshold": float(threshold),
    }


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output_dir}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    frame = load_fold_rows(args.dataset, args.fold)
    cache = np.load(args.feature_cache, allow_pickle=False)
    feature_by_id = {identifier: index for index, identifier in enumerate(cache["sequence_ids"].astype(str))}
    if set(frame.ID.astype(str)) != set(feature_by_id):
        raise ValueError("dataset and feature-cache IDs differ")
    frame["feature_index"] = [feature_by_id[identifier] for identifier in frame.ID.astype(str)]
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    features = np.asarray(cache["features"], dtype=np.float64)[frame.feature_index.to_numpy(dtype=np.int64)]
    labels = frame.label.to_numpy(dtype=np.int64)
    upstream_train = np.flatnonzero(frame.type.to_numpy() == "train")
    test = np.flatnonzero(frame.type.to_numpy() == "test")
    split = StratifiedShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    local_train, local_validation = next(split.split(upstream_train, labels[upstream_train]))
    train, validation = upstream_train[local_train], upstream_train[local_validation]
    scaler = StandardScaler().fit(features[train])
    x_train, x_validation, x_test = scaler.transform(features[train]), scaler.transform(features[validation]), scaler.transform(features[test])
    selected = None
    curve = []
    for c_value in args.c_values:
        model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=args.seed)
        model.fit(x_train, labels[train])
        probability = model.predict_proba(x_validation)[:, 1]
        aupr = float(average_precision_score(labels[validation], probability))
        curve.append({"C": float(c_value), "validation_aupr": aupr})
        if selected is None or aupr > selected[0]:
            selected = (aupr, float(c_value))
    assert selected is not None
    _, c_value = selected
    model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=args.seed)
    model.fit(x_train, labels[train])
    validation_probability = model.predict_proba(x_validation)[:, 1]
    test_probability = model.predict_proba(x_test)[:, 1]
    threshold = best_f1_threshold(labels[validation], validation_probability)
    report = metrics(labels[test], test_probability, threshold)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"fold": args.fold, "structure_only_validation_selected": report,
                   "selected_C": c_value, "best_validation_aupr": selected[0],
                   "validation_curve": curve}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with gzip.open(args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "label", "probability"))
        writer.writeheader()
        for index, probability in zip(test, test_probability):
            writer.writerow({"id": frame.ID.iloc[int(index)], "label": int(labels[index]), "probability": float(probability)})
    manifest = {"schema_version": 1, "experiment": "structure_only_logistic_diagnostic",
                "fold": args.fold, "seed": args.seed, "validation_fraction": args.validation_fraction,
                "feature_names": cache["feature_names"].astype(str).tolist(),
                "dataset_sha256": sha256(args.dataset), "feature_cache_sha256": sha256(args.feature_cache),
                "selection": "validation AUPR", "test_labels_used_for_selection": False}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
