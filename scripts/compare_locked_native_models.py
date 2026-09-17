#!/usr/bin/env python3
"""Compare locked structure and RNA-FM runs with paired OOF statistics."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PRIMARY_METRICS = ("auc", "aupr")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-folds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_runs(paths: list[Path], expected_variant: str | None) -> dict[int, dict]:
    runs: dict[int, dict] = {}
    for path in paths:
        manifest_path = path / "run_manifest.json"
        metrics_path = path / "metrics.json"
        predictions_path = path / "test_predictions.csv.gz"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment") != "native_rnafm_structires_locked_checkpoint_evaluation":
            raise ValueError(f"not a locked-checkpoint evaluation: {path}")
        if manifest.get("official_test_fold_evaluated_once_after_lock") is not True:
            raise ValueError(f"run does not certify one-shot official evaluation: {path}")
        if manifest.get("test_labels_used_for_selection") is not False:
            raise ValueError(f"test-selected run is ineligible: {path}")
        variant = str(manifest["variant"])
        if expected_variant is not None and variant != expected_variant:
            raise ValueError(f"expected {expected_variant}, found {variant}: {path}")
        fold = int(manifest["fold"])
        if fold in runs:
            raise ValueError(f"duplicate fold {fold}")
        with gzip.open(predictions_path, "rt", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        index = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
        label = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        score = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
        if len(index) != len(np.unique(index)):
            raise ValueError(f"duplicate sample indices in fold {fold}: {path}")
        if set(np.unique(label)) != {0, 1}:
            raise ValueError(f"fold {fold} does not contain both classes")
        runs[fold] = {
            "path": path,
            "manifest": manifest,
            "index": index,
            "label": label,
            "score": score,
            "hashes": {
                "manifest_sha256": sha256(manifest_path),
                "metrics_sha256": sha256(metrics_path),
                "predictions_sha256": sha256(predictions_path),
            },
        }
    return runs


def metric_values(label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(label, score)),
        "aupr": float(average_precision_score(label, score)),
    }


def exact_sign_flip_pvalue(delta: np.ndarray) -> float:
    """Exact two-sided sign-flip test for a paired mean difference."""
    observed = abs(float(delta.mean()))
    extreme = 0
    total = 1 << len(delta)
    for signs in itertools.product((-1.0, 1.0), repeat=len(delta)):
        statistic = abs(float(np.mean(delta * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
    return float(extreme / total)


def paired_stratified_bootstrap(
    label: np.ndarray,
    structure_score: np.ndarray,
    baseline_score: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(label == 1)
    negative = np.flatnonzero(label == 0)
    values = {metric: np.empty(replicates, dtype=np.float64) for metric in PRIMARY_METRICS}
    for replicate in range(replicates):
        sampled = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        y = label[sampled]
        structure = metric_values(y, structure_score[sampled])
        baseline = metric_values(y, baseline_score[sampled])
        for metric in PRIMARY_METRICS:
            values[metric][replicate] = structure[metric] - baseline[metric]
    return {
        metric: {
            "lower_95": float(np.quantile(values[metric], 0.025)),
            "median": float(np.quantile(values[metric], 0.5)),
            "upper_95": float(np.quantile(values[metric], 0.975)),
        }
        for metric in PRIMARY_METRICS
    }


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    expected = sorted(int(value) for value in args.expected_folds.split(",") if value.strip())
    structure = load_locked_runs(args.structure_runs, expected_variant=None)
    baseline = load_locked_runs(args.baseline_runs, expected_variant="sequence")
    if sorted(structure) != expected or sorted(baseline) != expected:
        raise ValueError(
            f"fold coverage differs from expected {expected}: "
            f"structure={sorted(structure)}, baseline={sorted(baseline)}"
        )
    structure_variants = {value["manifest"]["variant"] for value in structure.values()}
    if len(structure_variants) != 1 or "sequence" in structure_variants:
        raise ValueError(f"expected one non-sequence structure variant: {structure_variants}")
    lock_hashes = {
        value["manifest"]["selection_lock_sha256"]
        for collection in (structure, baseline)
        for value in collection.values()
    }
    if len(lock_hashes) != 1:
        raise ValueError("structure and baseline runs do not share one selection lock")

    per_fold = []
    oof_index = []
    oof_label = []
    oof_structure = []
    oof_baseline = []
    for fold in expected:
        left, right = structure[fold], baseline[fold]
        if not np.array_equal(left["index"], right["index"]):
            raise ValueError(f"test sample order differs in fold {fold}")
        if not np.array_equal(left["label"], right["label"]):
            raise ValueError(f"test labels differ in fold {fold}")
        structure_metrics = metric_values(left["label"], left["score"])
        baseline_metrics = metric_values(right["label"], right["score"])
        per_fold.append(
            {
                "fold": fold,
                "n_test": len(left["label"]),
                **{f"structure_{key}": value for key, value in structure_metrics.items()},
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                **{
                    f"delta_{key}": structure_metrics[key] - baseline_metrics[key]
                    for key in PRIMARY_METRICS
                },
            }
        )
        oof_index.append(left["index"])
        oof_label.append(left["label"])
        oof_structure.append(left["score"])
        oof_baseline.append(right["score"])

    indices = np.concatenate(oof_index)
    if len(indices) != len(np.unique(indices)):
        raise ValueError("official folds are not disjoint; OOF sample IDs repeat")
    label = np.concatenate(oof_label)
    structure_score = np.concatenate(oof_structure)
    baseline_score = np.concatenate(oof_baseline)
    oof_structure_metrics = metric_values(label, structure_score)
    oof_baseline_metrics = metric_values(label, baseline_score)
    fold_statistics = {}
    for metric in PRIMARY_METRICS:
        delta = np.asarray([row[f"delta_{metric}"] for row in per_fold])
        fold_statistics[metric] = {
            "mean_delta": float(delta.mean()),
            "std_delta": float(delta.std(ddof=1)),
            "positive_folds": int((delta > 0).sum()),
            "exact_sign_flip_pvalue": exact_sign_flip_pvalue(delta),
        }
    bootstrap = paired_stratified_bootstrap(
        label,
        structure_score,
        baseline_score,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    summary = {
        "schema_version": 1,
        "structure_variant": next(iter(structure_variants)),
        "baseline_variant": "sequence",
        "folds": expected,
        "n_oof": int(len(label)),
        "oof": {
            "structure": oof_structure_metrics,
            "baseline": oof_baseline_metrics,
            "delta": {
                metric: oof_structure_metrics[metric] - oof_baseline_metrics[metric]
                for metric in PRIMARY_METRICS
            },
        },
        "fold_paired_statistics": fold_statistics,
        "paired_stratified_oof_bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "delta_confidence_interval": bootstrap,
        },
    }
    args.output_dir.mkdir(parents=True)
    fields = (
        "fold", "n_test", "structure_auc", "baseline_auc", "delta_auc",
        "structure_aupr", "baseline_aupr", "delta_aupr",
    )
    with (args.output_dir / "paired_per_fold.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_fold)
    (args.output_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "locked_native_structure_vs_rnafm_paired_oof",
        "selection_lock_sha256": next(iter(lock_hashes)),
        "structure_inputs": [structure[fold]["hashes"] for fold in expected],
        "baseline_inputs": [baseline[fold]["hashes"] for fold in expected],
        "test_metrics_independently_recomputed": True,
        "test_labels_used_for_selection": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
