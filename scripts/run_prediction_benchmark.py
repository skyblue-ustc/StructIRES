#!/usr/bin/env python3
"""Run leakage-aware lightweight baselines on reconstructed IRES-AI folds."""

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

from ires_design.prediction import (  # noqa: E402
    composition_features,
    dataset_audit,
    load_ires_ai_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models", default="composition,kmer", help="comma-separated: composition,kmer"
    )
    parser.add_argument(
        "--protocol", choices=("reconstructed_nested10",), default="reconstructed_nested10"
    )
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-features", type=int, default=65536)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prediction = probability >= threshold
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "aupr": float(average_precision_score(y_true, probability)),
        "f1": float(f1_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, probability)),
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
    raise ValueError(f"unknown model: {model_name}")


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for model_name in sorted({str(row["model"]) for row in rows}):
        selected = [row for row in rows if row["model"] == model_name]
        summary: dict[str, object] = {"model": model_name, "n_folds": len(selected)}
        for key in ("auc", "aupr", "f1", "precision", "recall", "balanced_accuracy", "brier"):
            values = np.asarray([float(row[key]) for row in selected])
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_audit(audit: dict[str, object], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_counts = audit["source_by_label"]
    sources = list(source_counts)
    negative = [source_counts[source]["0"] for source in sources]
    positive = [source_counts[source]["1"] for source in sources]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    positions = np.arange(len(sources))
    axes[0].bar(positions, negative, label="negative", color="#6B8EAD")
    axes[0].bar(positions, positive, bottom=negative, label="positive", color="#D97757")
    axes[0].set_yscale("symlog", linthresh=100)
    axes[0].set_xticks(positions, sources, rotation=35, ha="right")
    axes[0].set_ylabel("Sequence count (symlog)")
    axes[0].set_title("IRES-AI source and label composition")
    axes[0].legend(frameon=False)

    length_counts = audit["length_group_by_label"]
    order = ["<174", "174", "175-200", ">200"]
    neg_length = [length_counts.get(group, {}).get("0", 0) for group in order]
    pos_length = [length_counts.get(group, {}).get("1", 0) for group in order]
    axes[1].bar(order, neg_length, label="negative", color="#6B8EAD")
    axes[1].bar(order, pos_length, bottom=neg_length, label="positive", color="#D97757")
    axes[1].set_yscale("symlog", linthresh=100)
    axes[1].set_ylabel("Sequence count (symlog)")
    axes[1].set_title("Length composition exposes assay shift")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = load_ires_ai_records(args.dataset)
    audit = dataset_audit(records)
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_audit(audit, args.output_dir / "dataset_audit.png")

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    sequences = [record.sequence for record in records]
    labels = np.asarray([record.label for record in records], dtype=np.int8)
    test_folds = np.asarray([record.test_fold for record in records], dtype=np.int8)

    composition = composition_features(sequences) if "composition" in models else None
    hashed = None
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
        hashed = vectorizer.transform(sequences)

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for model_name in models:
        features = composition if model_name == "composition" else hashed
        if features is None:
            raise RuntimeError(f"features were not initialized for {model_name}")
        for test_fold in folds:
            test_mask = test_folds == test_fold
            validation_fold = (test_fold + 1) % 10
            validation_mask = test_folds == validation_fold
            train_mask = ~(test_mask | validation_mask)

            estimator = build_estimator(model_name, args.seed + test_fold, args.max_iter)
            estimator.fit(features[train_mask], labels[train_mask])
            validation_probability = estimator.predict_proba(features[validation_mask])[:, 1]
            threshold = best_f1_threshold(labels[validation_mask], validation_probability)
            probability = estimator.predict_proba(features[test_mask])[:, 1]
            row: dict[str, object] = {
                "model": model_name,
                "protocol": args.protocol,
                "test_fold": test_fold,
                "validation_fold": "" if validation_fold is None else validation_fold,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "threshold": threshold,
            }
            row.update(metrics(labels[test_mask], probability, threshold))
            fold_rows.append(row)
            for index, score in zip(np.flatnonzero(test_mask), probability):
                record = records[int(index)]
                prediction_rows.append(
                    {
                        "model": model_name,
                        "sequence_id": record.sequence_id,
                        "test_fold": test_fold,
                        "label": record.label,
                        "probability": float(score),
                    }
                )
            print(
                f"{model_name} fold={test_fold} "
                f"AUC={row['auc']:.4f} AUPR={row['aupr']:.4f} F1={row['f1']:.4f}",
                flush=True,
            )

    summary_rows = summarize(fold_rows)
    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    with gzip.open(args.output_dir / "predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    manifest = {
        "schema_version": 1,
        "experiment": "ires_ai_lightweight_prediction_baselines",
        "protocol": args.protocol,
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "models": models,
        "folds": folds,
        "seed": args.seed,
        "test_labels_used_for_tuning": False,
        "notes": (
            "The released random split is ten repeated holdouts, not mutually exclusive K-fold. "
            "This pilot uses deterministic label/source/length-stratified project folds; "
            "thresholds are selected on a separate validation fold. Cluster/family evaluation pending."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
