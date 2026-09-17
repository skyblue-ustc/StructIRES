#!/usr/bin/env python3
"""Audit the released IRES-LM fold metrics without claiming a model rerun."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path


METRIC_COLUMNS = {
    "auc": "test_AUC",
    "aupr": "test_AUPR",
    "f1": "test_F1",
    "precision": "test_precision",
    "recall": "test_recall",
    "accuracy": "test_accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-folds", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def read_released_metrics(path: Path, expected_folds: int) -> tuple[list[dict[str, float]], dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < expected_folds + 2:
        raise ValueError(f"{path} does not contain {expected_folds} folds plus mean/std")
    fold_rows = rows[:expected_folds]
    released_mean = rows[expected_folds]
    released_std = rows[expected_folds + 1]
    parsed = [{key: float(row[column]) for key, column in METRIC_COLUMNS.items()} for row in fold_rows]
    for key, column in METRIC_COLUMNS.items():
        observed_mean = statistics.fmean(row[key] for row in parsed)
        observed_std = statistics.stdev(row[key] for row in parsed)
        if abs(observed_mean - float(released_mean[column])) > 1e-8:
            raise ValueError(f"released mean mismatch for {path.name}:{column}")
        if abs(observed_std - float(released_std[column])) > 1e-8:
            raise ValueError(f"released std mismatch for {path.name}:{column}")
    return parsed, {key: statistics.fmean(row[key] for row in parsed) for key in METRIC_COLUMNS}


def selection_audit(script_path: Path) -> dict[str, object]:
    source = script_path.read_text(encoding="utf-8")
    uses_test_each_epoch = "test_metrics, test_loss, test_result = predict_step(test_dataloader, model)" in source
    selects_on_test_aupr = bool(
        re.search(r"if\s+test_metrics\[['\"]AUPR['\"]\].*?>\s*best_aupr", source)
    )
    return {
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "test_evaluated_each_training_epoch": uses_test_each_epoch,
        "checkpoint_selected_on_test_aupr": selects_on_test_aupr,
        "independent_blind_test": not (uses_test_each_epoch and selects_on_test_aupr),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    released = {
        "RNA-FM": args.upstream_root / "results" / "IRES_RNAFM_CLS2_10folds_AvgEmbFalse_BosEmbTrue_epoch10_nodes40_dropout30.5_finetuneESMTrue_finetuneLastLayerESMFalse_lr0.0001_metrics.csv",
        "UTR-LM": args.upstream_root / "results" / "IRES_UTRLM_FinetuneESM_lr1e-4_dr5_bos_CLS20_10folds_F0_AvgEmbFalse_BosEmbTrue_epoch100_nodes40_dropout30.5_finetuneESMTrue_finetuneLastLayerESMFalse_lr0.0001_metrics.csv",
    }
    scripts = {
        "RNA-FM": args.upstream_root / "Script" / "IRES_RNAFM.py",
        "UTR-LM": args.upstream_root / "Script" / "IRES_UTRLM.py",
    }
    per_fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    audits: dict[str, dict[str, object]] = {}
    for model, metrics_path in released.items():
        folds, means = read_released_metrics(metrics_path, args.expected_folds)
        audit = selection_audit(scripts[model])
        audits[model] = audit
        for fold, values in enumerate(folds):
            per_fold_rows.append({"model": model, "fold": fold, **values})
        summary_rows.append(
            {
                "model": model,
                "status": "released_metrics_recomputed_not_model_rerun",
                "protocol": "released_repeated_90_10_holdouts",
                "n_folds": args.expected_folds,
                **means,
                "checkpoint_selected_on_test_aupr": audit["checkpoint_selected_on_test_aupr"],
            }
        )
    write_csv(args.output_dir / "per_fold_metrics.csv", per_fold_rows)
    write_csv(args.output_dir / "metrics.csv", summary_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "published_ireslm_release_audit",
        "upstream_root": str(args.upstream_root),
        "upstream_commit": upstream_commit(args.upstream_root),
        "metrics_files": {model: {"path": str(path), "sha256": sha256(path)} for model, path in released.items()},
        "selection_audit": audits,
        "interpretation": "Numerical recomputation of released CSVs only; not a checkpoint rerun. The released training scripts select checkpoints using test AUPR.",
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
