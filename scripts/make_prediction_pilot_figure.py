#!/usr/bin/env python3
"""Create the manuscript dataset-audit and lightweight-pilot figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cluster-run", type=Path, default=None)
    parser.add_argument("--iresfinder-native-run", type=Path, default=None)
    parser.add_argument("--iresfinder-cluster-run", type=Path, default=None)
    parser.add_argument("--iresfinder-overlap-run", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("paper/figures/fig2_prediction_pilot.pdf")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = json.loads((args.run_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    with (args.run_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    cluster_summary = {}
    if args.cluster_run is not None:
        with (args.cluster_run / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            cluster_summary = {row["model"]: row for row in csv.DictReader(handle)}
    if args.iresfinder_native_run is not None:
        with (args.iresfinder_native_run / "metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            iresfinder_rows = list(csv.DictReader(handle))
        summary.append(
            {
                "model": "IRESfinder",
                **{
                    f"{metric}_{stat}": value
                    for metric in ("auc", "aupr", "f1")
                    for stat, value in (
                        ("mean", float(np.mean([float(row[metric]) for row in iresfinder_rows]))),
                        ("std", float(np.std([float(row[metric]) for row in iresfinder_rows]))),
                    )
                },
            }
        )
    if args.iresfinder_cluster_run is not None:
        with (args.iresfinder_cluster_run / "metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            cluster_summary["IRESfinder"] = next(csv.DictReader(handle))

    colors = {"negative": "#668BA7", "positive": "#DE7654"}
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.15))

    source_counts = audit["source_by_label"]
    sources = list(source_counts)
    negative = np.asarray([source_counts[source]["0"] for source in sources])
    positive = np.asarray([source_counts[source]["1"] for source in sources])
    positions = np.arange(len(sources))
    axes[0].bar(positions, negative, color=colors["negative"], label="negative")
    axes[0].bar(
        positions, positive, bottom=negative, color=colors["positive"], label="positive"
    )
    axes[0].set_yscale("symlog", linthresh=100)
    axes[0].set_xticks(positions, sources, rotation=32, ha="right")
    axes[0].set_ylabel("Sequence count (symlog)")
    axes[0].set_title("a  Source/label imbalance")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)

    metrics = ("auc", "aupr", "f1")
    labels = ("AUC", "AUPR", "F1")
    width = 0.72 / max(len(summary), 1)
    model_labels = {
        "composition": "Composition",
        "kmer": "Hashed 3–6-mer",
        "IRESfinder": "IRESfinder",
    }
    model_colors = {"composition": "#668BA7", "kmer": "#DE7654", "IRESfinder": "#5B9E6F"}
    for index, row in enumerate(summary):
        means = [float(row[f"{metric}_mean"]) for metric in metrics]
        errors = [float(row[f"{metric}_std"]) for metric in metrics]
        offset = (index - (len(summary) - 1) / 2) * width
        axes[1].bar(
            np.arange(len(metrics)) + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=model_labels.get(row["model"], row["model"]),
            color=model_colors.get(row["model"], "#777777"),
        )
        if row["model"] in cluster_summary:
            cluster = cluster_summary[row["model"]]
            axes[1].scatter(
                np.arange(len(metrics)) + offset,
                [float(cluster[metric]) for metric in metrics],
                marker="D",
                s=48,
                facecolor="white",
                edgecolor="#222222",
                linewidth=1.2,
                zorder=3,
            )
    if cluster_summary:
        axes[1].scatter(
            [],
            [],
            marker="D",
            s=48,
            facecolor="white",
            edgecolor="#222222",
            linewidth=1.2,
            label="90% cluster-disjoint test",
        )
    axes[1].set_xticks(np.arange(len(metrics)), labels)
    axes[1].set_ylim(0, 0.8)
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("b  Mixed-source versus cluster-disjoint")
    axes[1].legend(frameon=False, fontsize=9)
    if args.iresfinder_overlap_run is not None:
        overlap = json.loads(
            (args.iresfinder_overlap_run / "run_manifest.json").read_text(encoding="utf-8")
        )
        axes[1].text(
            0.98,
            0.04,
            f"IRESfinder exact train overlap:\n{overlap['n_exact_benchmark_overlaps']}/"
            f"{overlap['n_iresfinder_training']}",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

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
