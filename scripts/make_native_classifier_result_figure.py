#!/usr/bin/env python3
"""Plot independently recomputed paired native-fold classifier results."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from publication_style import INK, NAVY, STRUCTIRES, apply_style

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("/9950backfile/lant/data/ires-design-external-runs/") / \
    "structires_native_authorstyle_batchshuffle_multifold_0_2_4_v2_20260823/per_fold_metrics.csv"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "fig2_native_classifier_results"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    rows: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["model"]][int(row["fold"])] = {
                key: float(value) for key, value in row.items()
                if key not in {"model", "fold", "n_test"}
            }
    expected = {"author_style_sequence_only", "structires_mfe_contact_fusion"}
    if set(rows) != expected:
        raise ValueError(f"expected model rows {sorted(expected)}, got {sorted(rows)}")
    sequence_folds = set(rows["author_style_sequence_only"])
    contact_folds = set(rows["structires_mfe_contact_fusion"])
    if not sequence_folds or sequence_folds != contact_folds:
        raise ValueError("models must have one identical, non-empty set of independently verified folds")
    return rows


def paired_panel(ax, rows, folds, metric, title):
    x = np.arange(len(folds))
    sequence = rows["author_style_sequence_only"]
    contact = rows["structires_mfe_contact_fusion"]
    y0 = np.array([sequence[fold][metric] for fold in folds])
    y1 = np.array([contact[fold][metric] for fold in folds])
    for idx in range(len(folds)):
        ax.plot((x[idx], x[idx]), (y0[idx], y1[idx]), color="#B8C2CC", lw=.9, zorder=1)
    ax.scatter(x - .035, y0, s=28, color=NAVY, marker="o", zorder=3)
    ax.scatter(x + .035, y1, s=34, color=STRUCTIRES, marker="D", zorder=3)
    delta = (y1 - y0).mean()
    ax.text(.98, .91, f"Δ = {delta:+.3f}", transform=ax.transAxes, ha="right", fontsize=6.35, color=INK)
    ax.set_title(title, loc="left", fontsize=8.7, weight="bold", color=INK, pad=7)
    labels = [f"fold {fold}" for fold in folds]
    ax.set_xticks(x, labels, fontsize=6.5)
    if len(folds) > 5:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", color="#E6EBEF", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.8)
    low, high = float(min(y0.min(), y1.min())), float(max(y0.max(), y1.max()))
    pad = max(.003, .25 * (high - low))
    ax.set_ylim(low - pad, high + pad)


def main() -> None:
    args = arguments()
    rows = read_rows(args.input)
    folds = tuple(sorted(rows["author_style_sequence_only"]))
    apply_style(plt)
    fig, axes = plt.subplots(1, 4, figsize=(7.35, 2.75), layout="constrained")
    paired_panel(axes[0], rows, folds, "auc", "AUROC")
    paired_panel(axes[1], rows, folds, "aupr", "AUPR")
    paired_panel(axes[2], rows, folds, "f1", "F1")
    paired_panel(axes[3], rows, folds, "mcc", "MCC")
    fig.legend(handles=[Line2D([], [], color=NAVY, marker="o", linestyle="None", label="sequence-only"),
                        Line2D([], [], color=STRUCTIRES, marker="D", linestyle="None", label="MFE-contact fusion")],
               loc="lower center", ncol=2,
               bbox_to_anchor=(.5, -.08), fontsize=7.1, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
