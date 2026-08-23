#!/usr/bin/env python3
"""Audit cross-fold heterogeneity in frozen RNA-FM direct-RNA transfer."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_released_rnafm_cross_assay import bootstrap, metrics


FOLD_COLUMN = re.compile(r"probability_fold(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_summary(labels: np.ndarray, values: np.ndarray) -> dict[str, float]:
    return {
        "probability_min": float(values.min()),
        "probability_q25": float(np.quantile(values, 0.25)),
        "probability_median": float(np.median(values)),
        "probability_q75": float(np.quantile(values, 0.75)),
        "probability_max": float(values.max()),
        "probability_mean": float(values.mean()),
        "probability_std": float(values.std(ddof=1)),
        "probability_mean_negative": float(values[labels == 0].mean()),
        "probability_mean_positive": float(values[labels == 1].mean()),
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = args.transfer_run / "predictions.csv.gz"
    source_manifest_path = args.transfer_run / "run_manifest.json"
    if not predictions_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError(f"incomplete transfer run: {args.transfer_run}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(predictions_path)
    labels = frame["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("transfer predictions contain non-binary labels")

    fold_columns = sorted(
        (column for column in frame if FOLD_COLUMN.fullmatch(column)),
        key=lambda column: int(FOLD_COLUMN.fullmatch(column).group(1)),  # type: ignore[union-attr]
    )
    expected = [f"probability_fold{fold}" for fold in source_manifest["folds"]]
    if fold_columns != expected:
        raise ValueError(f"prediction fold columns {fold_columns} do not match manifest {expected}")
    evaluation_columns = [*fold_columns, "probability"]

    metric_rows: list[dict[str, object]] = []
    for offset, column in enumerate(evaluation_columns):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
            raise ValueError(f"invalid probabilities in {column}")
        match = FOLD_COLUMN.fullmatch(column)
        metric_rows.append(
            {
                "prediction_view": "fold" if match else "arithmetic_mean_ensemble",
                "fold": int(match.group(1)) if match else "ensemble",
                "probability_column": column,
                **metrics(labels, values),
                **probability_summary(labels, values),
                **bootstrap(
                    labels,
                    values,
                    args.bootstrap_replicates,
                    args.seed + offset,
                ),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "per_fold_metrics.csv", index=False)

    correlation_frame = frame[fold_columns].corr(method="spearman")
    pairwise_rows = []
    for left, right in itertools.combinations(fold_columns, 2):
        pairwise_rows.append(
            {
                "left_fold": int(FOLD_COLUMN.fullmatch(left).group(1)),  # type: ignore[union-attr]
                "right_fold": int(
                    FOLD_COLUMN.fullmatch(right).group(1)  # type: ignore[union-attr]
                ),
                "spearman": float(correlation_frame.loc[left, right]),
                "pearson": float(frame[left].corr(frame[right], method="pearson")),
            }
        )
    pd.DataFrame(pairwise_rows).to_csv(
        args.output_dir / "pairwise_checkpoint_correlations.csv", index=False
    )

    output_manifest = {
        "schema_version": 1,
        "experiment": "released_rnafm_cross_assay_checkpoint_heterogeneity_audit",
        "source_transfer_run": str(args.transfer_run.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_predictions_sha256": sha256(predictions_path),
        "folds": source_manifest["folds"],
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "direct_labels_used_for_training_thresholding_or_model_selection": False,
        "interpretation_boundary": (
            "Per-fold direct-RNA labels are used only for post hoc frozen evaluation. "
            "No checkpoint is selected and no probability direction is changed."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metric_rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
