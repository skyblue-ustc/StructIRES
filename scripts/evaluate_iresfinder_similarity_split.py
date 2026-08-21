#!/usr/bin/env python3
"""Evaluate released IRESfinder scores on a frozen similarity-disjoint split."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
                handle, usecols=["idx", "ID", "Sequence", "IRES_class_600"]
            ).drop_duplicates("ID")
    if data["idx"].duplicated().any():
        raise ValueError("idx is not unique after canonicalization")
    data["sequence_sha256"] = data["Sequence"].str.upper().map(
        lambda sequence: hashlib.sha256(sequence.replace("T", "U").encode("ascii")).hexdigest()
    )

    assignments = pd.read_csv(args.assignments)
    assignments = assignments.rename(columns={"sequence_id": "ID"})
    merged = assignments.merge(data, on="ID", how="left", validate="one_to_one")
    if merged["idx"].isna().any():
        raise ValueError("assignment IDs missing from canonical dataset")
    if not (merged["sequence_sha256_x"] == merged["sequence_sha256_y"]).all():
        raise ValueError("assignment sequence hash mismatch")
    if not (merged["label"] == merged["IRES_class_600"]).all():
        raise ValueError("assignment label mismatch")

    predictions = pd.read_csv(args.predictions)
    predictions["idx"] = predictions["id"].str.removeprefix("idx_").astype(int)
    if predictions["idx"].duplicated().any():
        raise ValueError("duplicate prediction idx")
    merged = merged.merge(
        predictions[["idx", "probability"]], on="idx", how="left", validate="one_to_one"
    )
    if merged["probability"].isna().any():
        raise ValueError("missing IRESfinder probabilities")

    test = merged[merged["split"] == "test"].copy()
    predicted = (test["probability"] >= 0.5).astype(int)
    metrics = {
        "model": "IRESfinder",
        "status": "released_model_rerun_py3_adapter",
        "protocol": "frozen_hamming90_len174_cluster_disjoint_test",
        "n": int(len(test)),
        "n_positive": int(test["label"].sum()),
        "auc": float(roc_auc_score(test["label"], test["probability"])),
        "aupr": float(average_precision_score(test["label"], test["probability"])),
        "f1": float(f1_score(test["label"], predicted, zero_division=0)),
        "accuracy": float(accuracy_score(test["label"], predicted)),
        "decision_threshold": 0.5,
    }
    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    test[["ID", "idx", "label", "source", "cluster_id", "probability"]].to_csv(
        args.output_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "iresfinder_similarity_split_rerun",
        "scope": "classification-only reproduction",
        "dataset_zip": str(args.dataset_zip.resolve()),
        "dataset_zip_sha256": sha256(args.dataset_zip),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256(args.predictions),
        "assignments": str(args.assignments.resolve()),
        "assignments_sha256": sha256(args.assignments),
        "test_labels_used_for_training_thresholding_or_model_choice": False,
        "metrics": metrics,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
