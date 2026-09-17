#!/usr/bin/env python3
"""Plot final RNA-FM native and direct-RNA checkpoint heterogeneity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--transfer-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    native = pd.read_csv(args.native_summary / "fold_metrics.csv")
    transfer = pd.read_csv(args.transfer_audit / "per_fold_metrics.csv")
    pairs = pd.read_csv(args.transfer_audit / "pairwise_checkpoint_correlations.csv")
    manifest = json.loads(
        (args.transfer_audit / "run_manifest.json").read_text(encoding="utf-8")
    )
    folds = [int(value) for value in manifest["folds"]]
    native = native[native["fold"].isin(folds)].sort_values("fold")
    per_fold = transfer[transfer["prediction_view"] == "fold"].copy()
    per_fold["fold"] = per_fold["fold"].astype(int)
    per_fold = per_fold.sort_values("fold")
    if native["fold"].astype(int).tolist() != folds or per_fold["fold"].tolist() != folds:
        raise ValueError("native and transfer folds do not match the audit manifest")

    matrix = np.eye(len(folds), dtype=float)
    positions = {fold: index for index, fold in enumerate(folds)}
    for row in pairs.itertuples(index=False):
        left = positions[int(row.left_fold)]
        right = positions[int(row.right_fold)]
        matrix[left, right] = matrix[right, left] = float(row.spearman)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(folds)))
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.5), constrained_layout=True)

    native_auc = native["auc"].to_numpy(dtype=float)
    transfer_auc = per_fold["auc"].to_numpy(dtype=float)
    for index, fold in enumerate(folds):
        axes[0].plot(
            [0, 1],
            [native_auc[index], transfer_auc[index]],
            color=colors[index],
            alpha=0.8,
        )
        axes[0].scatter([0, 1], [native_auc[index], transfer_auc[index]], color=colors[index], s=28)
        axes[0].text(1.035, transfer_auc[index], str(fold), fontsize=7, va="center")
    ensemble = transfer[transfer["prediction_view"] == "arithmetic_mean_ensemble"].iloc[0]
    axes[0].scatter(1, float(ensemble["auc"]), marker="D", s=55, color="black", zorder=5)
    axes[0].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xticks([0, 1], ["Native holdout", "Direct-RNA"])
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.2, 0.86)
    axes[0].set_title("a  Per-checkpoint transfer")

    axes[1].scatter(native_auc, transfer_auc, c=colors, s=44, edgecolor="white", linewidth=0.6)
    for index, fold in enumerate(folds):
        axes[1].annotate(str(fold), (native_auc[index], transfer_auc[index]), xytext=(4, 3),
                         textcoords="offset points", fontsize=7)
    correlation = float(np.corrcoef(native_auc, transfer_auc)[0, 1])
    axes[1].text(0.03, 0.94, f"Pearson r = {correlation:.2f}", transform=axes[1].transAxes,
                 va="top", fontsize=8)
    axes[1].set_xlabel("Native-holdout AUROC")
    axes[1].set_ylabel("Direct-RNA AUROC")
    axes[1].set_title("b  Native score is not portability")

    image = axes[2].imshow(matrix, vmin=0.3, vmax=0.75, cmap="viridis", aspect="equal")
    axes[2].set_xticks(range(len(folds)), folds, fontsize=7)
    axes[2].set_yticks(range(len(folds)), folds, fontsize=7)
    axes[2].set_xlabel("Checkpoint fold")
    axes[2].set_ylabel("Checkpoint fold")
    axes[2].set_title("c  Direct-RNA rank correlation")
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label(r"Spearman $\rho$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    for axis in axes[:2]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    png_output = args.png_output or args.output.with_suffix(".png")
    figure.savefig(png_output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(
        json.dumps(
            {
                "folds": folds,
                "native_auc_mean": float(native_auc.mean()),
                "direct_auc_min": float(transfer_auc.min()),
                "direct_auc_max": float(transfer_auc.max()),
                "native_direct_pearson": correlation,
                "spearman_min": float(pairs["spearman"].min()),
                "spearman_max": float(pairs["spearman"].max()),
                "ensemble_auc": float(ensemble["auc"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
