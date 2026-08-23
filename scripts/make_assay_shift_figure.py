#!/usr/bin/env python3
"""Summarize the locked lightweight cross-assay transfer experiment for the manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL_LABELS = {"composition": "Composition", "kmer": "Hashed 3--6-mer"}
MODEL_COLORS = {"composition": "#668BA7", "kmer": "#DE7654"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-run", type=Path, required=True)
    parser.add_argument("--overlap-run", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("paper/figures/fig3_assay_shift.pdf")
    )
    parser.add_argument(
        "--table-output", type=Path, default=Path("paper/tables/assay_shift.tex")
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stratified_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    auc_values = np.empty(replicates, dtype=float)
    aupr_values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = np.concatenate(
            (
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            )
        )
        sampled_labels = labels[sampled]
        sampled_probabilities = probabilities[sampled]
        auc_values[index] = roc_auc_score(sampled_labels, sampled_probabilities)
        aupr_values[index] = average_precision_score(sampled_labels, sampled_probabilities)
    return {
        "auc_ci_low": float(np.quantile(auc_values, 0.025)),
        "auc_ci_high": float(np.quantile(auc_values, 0.975)),
        "aupr_ci_low": float(np.quantile(aupr_values, 0.025)),
        "aupr_ci_high": float(np.quantile(aupr_values, 0.975)),
    }


def calibration_points(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for index in range(bins):
        upper = probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        mask = (probabilities >= edges[index]) & upper
        if mask.any():
            predicted.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
            counts.append(int(mask.sum()))
    return np.asarray(predicted), np.asarray(observed), np.asarray(counts)


def write_bootstrap_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if row["subset"] in {"all_labelled", "high_similarity_90_80"}]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Frozen transfer from the legacy DNA/lentiviral reporter assay to direct-RNA IRES-TrAPPr labels. AUC and AUPR intervals are 95\% stratified-bootstrap intervals (2,000 replicates). The decision threshold was selected only on legacy validation folds.}",
        r"\label{tab:assay-shift}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Direct-RNA subset & $n$ (active) & AUC (95\% CI) & AUPR (95\% CI) & F1 \\",
        r"\midrule",
    ]
    for row in selected:
        subset = "All labelled" if row["subset"] == "all_labelled" else r"$\geq$90\% identity, $\geq$80\% coverage"
        lines.append(
            f"{MODEL_LABELS[str(row['model'])]} & {subset} & "
            f"{int(row['n'])} ({int(row['n_positive'])}) & "
            f"{float(row['auc']):.3f} ({float(row['auc_ci_low']):.3f}--{float(row['auc_ci_high']):.3f}) & "
            f"{float(row['aupr']):.3f} ({float(row['aupr_ci_low']):.3f}--{float(row['aupr_ci_high']):.3f}) & "
            f"{float(row['f1']):.3f} \\\\"  # noqa: E501
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    metric_rows = read_csv(args.transfer_run / "metrics.csv")
    predictions = read_csv(args.transfer_run / "predictions.csv")
    legacy_summary = {row["model"]: row for row in read_csv(args.legacy_run / "summary.csv")}
    overlap = json.loads((args.overlap_run / "summary.json").read_text(encoding="utf-8"))

    predictions_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        predictions_by_model[row["model"]].append(row)

    bootstrap_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(metric_rows):
        model_rows = predictions_by_model[row["model"]]
        if row["subset"] == "all_labelled":
            subset_rows = model_rows
        elif row["subset"] == "non_control":
            subset_rows = [item for item in model_rows if item["is_control"].lower() != "true"]
        elif row["subset"] == "legacy_contained":
            subset_rows = [item for item in model_rows if item["legacy_contained"].lower() == "true"]
        elif row["subset"] == "high_similarity_90_80":
            subset_rows = [item for item in model_rows if item["high_similarity_90_80"].lower() == "true"]
        else:
            raise ValueError(f"unknown subset: {row['subset']}")
        labels = np.asarray([int(item["label"]) for item in subset_rows])
        probabilities = np.asarray([float(item["probability"]) for item in subset_rows])
        bootstrap_rows.append(
            {
                **row,
                **stratified_bootstrap(
                    labels,
                    probabilities,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + row_index,
                ),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )

    write_bootstrap_csv(args.transfer_run / "bootstrap_metrics.csv", bootstrap_rows)
    write_latex_table(args.table_output, bootstrap_rows)

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    statuses = ("active", "inactive")
    positions = np.arange(len(statuses))
    width = 0.34
    for offset, (key, label, color) in zip(
        (-width / 2, width / 2),
        (
            ("contained", "Legacy fragment contained", "#668BA7"),
            ("high_similarity_90_80", "High similarity", "#DE7654"),
        ),
    ):
        values = [
            overlap["overlap_by_direct_status"][status][key]
            / overlap["overlap_by_direct_status"][status]["n"]
            for status in statuses
        ]
        axes[0].bar(positions + offset, values, width, label=label, color=color)
    axes[0].set_xticks(positions, ("Active (n=111)", "Inactive (n=2,068)"))
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of direct-RNA sequences")
    axes[0].set_title("a  Sequence overlap is label-skewed")
    axes[0].legend(frameon=False, fontsize=8)

    metrics = ("auc", "aupr")
    metric_positions = np.arange(len(metrics))
    for model_index, model in enumerate(("composition", "kmer")):
        legacy = legacy_summary[model]
        transfer = next(
            row for row in bootstrap_rows if row["model"] == model and row["subset"] == "all_labelled"
        )
        offset = (model_index - 0.5) * width
        within = [float(legacy[f"{metric}_mean"]) for metric in metrics]
        cross = [float(transfer[metric]) for metric in metrics]
        axes[1].bar(
            metric_positions + offset,
            within,
            width,
            color=MODEL_COLORS[model],
            alpha=0.95,
            label=f"{MODEL_LABELS[model]}: legacy CV",
        )
        axes[1].scatter(
            metric_positions + offset,
            cross,
            marker="D",
            s=42,
            facecolor="white",
            edgecolor="#222222",
            linewidth=1.1,
            zorder=3,
            label=f"{MODEL_LABELS[model]}: direct RNA",
        )
    axes[1].axhline(111 / 2179, color="#555555", ls="--", lw=1)
    axes[1].set_xticks(metric_positions, ("AUC", "AUPR"))
    axes[1].set_ylim(0, 0.8)
    axes[1].set_ylabel("Discrimination")
    axes[1].set_title("b  Legacy performance does not transfer")
    axes[1].legend(frameon=False, fontsize=7.4, loc="upper right")

    axes[2].plot((0, 1), (0, 1), color="#777777", ls="--", lw=1, label="Ideal")
    for model in ("composition", "kmer"):
        model_rows = predictions_by_model[model]
        labels = np.asarray([int(row["label"]) for row in model_rows])
        probabilities = np.asarray([float(row["probability"]) for row in model_rows])
        predicted, observed, counts = calibration_points(labels, probabilities)
        axes[2].plot(
            predicted,
            observed,
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        axes[2].scatter(predicted, observed, s=np.sqrt(counts) * 2.2, color=MODEL_COLORS[model])
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed active fraction")
    axes[2].set_title("c  Direct-RNA calibration failure")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
