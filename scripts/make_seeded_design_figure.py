#!/usr/bin/env python3
"""Render the first matched seeded-design trade-off figure."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


LABELS = {
    "random_mutation": "Random",
    "score_only": "Score-only",
    "utrlm_score_only": "Released UTR-LM\nscore-only",
    "structure_only": "Structure-only",
    "robust_full_first_pass": "Robust-full",
}
ORDER = list(LABELS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    colors = ["#7f8c8d", "#d95f02", "#7570b3", "#1b9e77", "#386cb0"]
    panels = [
        ("lm_log_likelihood_per_token", "RNA-LM likelihood\n(higher is better)"),
        ("mfe_abs_delta_kcal_mol", "$|\\Delta$MFE| (kcal/mol)\n(lower is better)"),
        ("pairing_profile_l1", "Pairing-profile $L_1$\n(lower is better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8), constrained_layout=True)
    for axis, (metric, title) in zip(axes, panels):
        values = [float(rows[key][f"{metric}_mean"]) for key in ORDER]
        errors = [float(rows[key][f"{metric}_std"]) for key in ORDER]
        axis.bar(range(len(ORDER)), values, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
        axis.set_xticks(range(len(ORDER)), [LABELS[key] for key in ORDER], rotation=25, ha="right", fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.25)
    axes[0].text(0.02, 0.98, "mean $\\pm$ SD across 30 parent--run units", transform=axes[0].transAxes, va="top", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
