#!/usr/bin/env python3
"""Paper-aligned repeated-holdout StructIRES feature benchmark.

The released IRES-AI table has ten overlapping 90/10 holdouts.  This script
uses those exact test assignments to give a fast, clearly labelled comparison
against the reported RNA-FM/UTR-LM/IRES-LM numbers.  It is a protocol-aligned
reference, not a replacement for the locked cluster-disjoint result.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ires_design.prediction import composition_features, load_ires_ai_records
from run_structires_structure_benchmark import FEATURE_NAMES, metrics


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hash-features", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output_dir}")
    records = load_ires_ai_records(args.dataset)
    records.sort(key=lambda item: item.sequence_id)
    cache = np.load(args.feature_cache, allow_pickle=False)
    if list(cache["sequence_ids"].astype(str)) != [item.sequence_id for item in records]:
        raise ValueError("cache does not match canonical record order")
    if list(cache["feature_names"].astype(str)) != list(FEATURE_NAMES):
        raise ValueError("unexpected structure-cache schema")
    structure = np.asarray(cache["features"], dtype=np.float32)
    labels = np.asarray([item.label for item in records], dtype=np.int8)

    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(3, 6), n_features=args.hash_features, alternate_sign=False, norm="l2", lowercase=False, dtype=np.float32)
    kmer = vectorizer.transform([item.sequence for item in records])
    composition = composition_features([item.sequence for item in records])
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for fold in range(10):
        test = np.asarray([fold in item.published_test_folds for item in records])
        train = ~test
        if not test.any() or train.sum() == 0:
            raise ValueError(f"published fold {fold} is empty")
        structure_all = StandardScaler().fit(structure[train]).transform(structure)
        composition_all = StandardScaler().fit(composition[train]).transform(composition)
        matrices = {
            "kmer_sequence": kmer,
            "structure_ensemble": csr_matrix(structure_all),
            "kmer_plus_structure": hstack((kmer, csr_matrix(structure_all)), format="csr"),
            "composition_plus_structure": csr_matrix(np.hstack((composition_all, structure_all))),
        }
        for offset, (name, matrix) in enumerate(matrices.items()):
            estimator = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=1e-5, l1_ratio=.05, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=args.seed + fold * 10 + offset)
            estimator.fit(matrix[train], labels[train])
            probability = estimator.predict_proba(matrix[test])[:, 1]
            result = {"model": name, "protocol": "paper_aligned_repeated_90_10", "test_fold": fold, "n_train": int(train.sum()), "n_test": int(test.sum()), **metrics(labels[test], probability, threshold=.5)}
            rows.append(result)
            for index, score in zip(np.flatnonzero(test), probability):
                prediction_rows.append({"model": name, "sequence_id": records[int(index)].sequence_id, "test_fold": fold, "label": int(labels[int(index)]), "probability": float(score)})
            print(f"{name} fold={fold} AUC={result['auc']:.4f} AUPR={result['aupr']:.4f} F1={result['f1']:.4f}", flush=True)
    summary = []
    for name in sorted({str(row["model"]) for row in rows}):
        selected = [row for row in rows if row["model"] == name]
        entry: dict[str, object] = {"model": name, "protocol": "paper_aligned_repeated_90_10", "n_folds": len(selected)}
        for key in ("auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc", "ece10"):
            values = np.asarray([float(row[key]) for row in selected])
            entry[f"{key}_mean"] = float(values.mean())
            entry[f"{key}_std"] = float(values.std(ddof=1))
        summary.append(entry)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "fold_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    with gzip.open(args.output_dir / "predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    manifest = {"schema_version": 1, "experiment": "structires_paper_aligned_repeated_holdout", "dataset_path": str(args.dataset), "dataset_sha256": digest(args.dataset), "feature_cache": str(args.feature_cache), "feature_cache_sha256": digest(args.feature_cache), "test_assignment": "released overlapping 10x 90/10 holdouts", "threshold": 0.5, "test_labels_used_for_training_or_model_selection": False, "interpretation": "protocol-aligned reference only; overlapping test sets make mean±std non-independent and upstream published models selected checkpoints using test AUPR."}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
