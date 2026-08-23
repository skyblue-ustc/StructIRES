#!/usr/bin/env python3
"""Audit exact overlap between IRESfinder training data and project benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iresfinder-root", type=Path, required=True)
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


def compute_metrics(data: pd.DataFrame) -> dict[str, float | int]:
    labels = data["IRES_class_600"].astype(int)
    probabilities = data["probability"].astype(float)
    return {
        "n": int(len(data)),
        "n_positive": int(labels.sum()),
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, probabilities >= 0.5, zero_division=0)),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    positive_path = args.iresfinder_root / "dataset" / "training_dataset_ires.seq"
    negative_path = args.iresfinder_root / "dataset" / "training_dataset_nonires.seq"
    training_rows: list[dict[str, object]] = []
    for path, label in ((positive_path, 1), (negative_path, 0)):
        for sequence in path.read_text(encoding="ascii").splitlines():
            training_rows.append(
                {"sequence_dna": sequence.strip().upper().replace("U", "T"), "training_label": label}
            )
    training = pd.DataFrame(training_rows)
    if training["sequence_dna"].duplicated().any():
        raise ValueError("duplicate IRESfinder training sequence")

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
            long_data = pd.read_csv(
                handle,
                usecols=["fold", "type", "idx", "ID", "Sequence", "IRES_class_600", "Source"],
            )
    canonical = long_data.drop_duplicates("idx").copy()
    canonical["sequence_dna"] = canonical["Sequence"].str.upper().str.replace("U", "T", regex=False)
    canonical = canonical.merge(training, on="sequence_dna", how="left", validate="one_to_one")
    canonical["iresfinder_training_overlap"] = canonical["training_label"].notna()
    overlaps = canonical[canonical["iresfinder_training_overlap"]].copy()
    if not (overlaps["IRES_class_600"] == overlaps["training_label"]).all():
        raise ValueError("label disagreement in exact IRESfinder training overlaps")

    predictions = pd.read_csv(args.predictions)
    predictions["idx"] = predictions["id"].str.removeprefix("idx_").astype(int)
    canonical = canonical.merge(
        predictions[["idx", "probability"]], on="idx", how="left", validate="one_to_one"
    )
    if canonical["probability"].isna().any():
        raise ValueError("missing IRESfinder probability")
    long_data = long_data.merge(
        canonical[["idx", "probability", "iresfinder_training_overlap"]],
        on="idx",
        how="left",
        validate="many_to_one",
    )

    metric_rows: list[dict[str, object]] = []
    for fold in range(10):
        native_test = long_data[(long_data["fold"] == fold) & (long_data["type"] == "test")]
        for status, subset in (
            ("all", native_test),
            ("training_overlap_removed", native_test[~native_test["iresfinder_training_overlap"]]),
        ):
            metric_rows.append(
                {
                    "protocol": "released_native_test_fold",
                    "overlap_status": status,
                    "fold": fold,
                    **compute_metrics(subset),
                }
            )

    assignments = pd.read_csv(args.assignments).rename(columns={"sequence_id": "ID"})
    cluster_test = canonical.merge(
        assignments[["ID", "split", "cluster_id"]], on="ID", validate="one_to_one"
    )
    cluster_test = cluster_test[cluster_test["split"] == "test"]
    for status, subset in (
        ("all", cluster_test),
        ("training_overlap_removed", cluster_test[~cluster_test["iresfinder_training_overlap"]]),
    ):
        metric_rows.append(
            {
                "protocol": "frozen_hamming90_len174_cluster_disjoint_test",
                "overlap_status": status,
                "fold": "",
                **compute_metrics(subset),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    overlaps[
        ["idx", "ID", "Source", "IRES_class_600", "training_label", "sequence_dna"]
    ].to_csv(args.output_dir / "exact_overlaps.csv.gz", index=False, compression="gzip")
    native = metrics[metrics["protocol"] == "released_native_test_fold"]
    native_summary = (
        native.groupby("overlap_status")[["n", "n_positive", "auc", "aupr", "f1"]]
        .mean()
        .to_dict(orient="index")
    )
    summary = {
        "schema_version": 1,
        "experiment": "iresfinder_training_overlap_audit",
        "n_iresfinder_training": int(len(training)),
        "n_exact_benchmark_overlaps": int(len(overlaps)),
        "exact_overlap_fraction_of_iresfinder_training": float(len(overlaps) / len(training)),
        "exact_overlap_label_agreement": 1.0,
        "exact_overlap_source_counts": overlaps["Source"].value_counts().to_dict(),
        "native_repeated_fold_means": native_summary,
        "cluster_test": metrics[
            metrics["protocol"] == "frozen_hamming90_len174_cluster_disjoint_test"
        ].to_dict(orient="records"),
        "inputs": {
            "positive_training_sha256": sha256(positive_path),
            "negative_training_sha256": sha256(negative_path),
            "dataset_zip_sha256": sha256(args.dataset_zip),
            "predictions_sha256": sha256(args.predictions),
            "assignments_sha256": sha256(args.assignments),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
