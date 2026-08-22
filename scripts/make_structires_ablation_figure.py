#!/usr/bin/env python3
"""Render the matched released-IRES-LM constraint ablation figure."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


LABELS = {
    "ireslm_score_only": "Score",
    "ireslm_plus_energy": "+E",
    "ireslm_plus_ensemble": "+Ens",
    "ireslm_plus_anchor": "+Anc",
    "structires": "StructIRES",
}
ORDER = list(LABELS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    with args.structure.open(newline="", encoding="utf-8") as handle:
        structure = {row["method"]: row for row in csv.DictReader(handle)}
    with args.proxy.open(newline="", encoding="utf-8") as handle:
        proxy = {row["method"]: row for row in csv.DictReader(handle)}
    colors = ["#9e9e9e", "#e69f00", "#56b4e9", "#cc79a7", "#0072b2"]
    panels = [
        ("mfe_abs_delta_kcal_mol", r"$|\Delta$MFE| (kcal/mol)", "lower is better"),
        ("pairing_profile_l1", r"Pairing-profile $L_1$", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.7), constrained_layout=True)
    for axis, (metric, title, note) in zip(axes[:2], panels):
        values = [float(structure[key][f"{metric}_mean"]) for key in ORDER]
        errors = [float(structure[key][f"{metric}_std"]) for key in ORDER]
        axis.bar(range(len(ORDER)), values, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
        axis.set_title(title, fontsize=10)
        axis.text(0.02, 0.97, note, transform=axis.transAxes, va="top", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    metric = "parent_ensemble_anchor_retention"
    values = [float(structure[key][f"{metric}_mean"]) for key in ORDER]
    errors = [float(structure[key][f"{metric}_std"]) for key in ORDER]
    axes[2].bar(range(len(ORDER)), values, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Parent ensemble-anchor retention", fontsize=10)
    axes[2].text(0.02, 0.97, "higher is better", transform=axes[2].transAxes, va="top", fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)
    values = [float(proxy[key]["mean_direct_rna_s3_heldout_proxy"]) for key in ORDER]
    axes[3].bar(range(len(ORDER)), values, color=colors, edgecolor="black", linewidth=0.5)
    axes[3].set_title("Held-out direct-RNA proxy", fontsize=10)
    axes[3].text(0.02, 0.97, "higher is better", transform=axes[3].transAxes, va="top", fontsize=8)
    axes[3].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(range(len(ORDER)), [LABELS[key] for key in ORDER], fontsize=8)
    axes[0].text(0.02, -0.34, "Mean $\pm$ sample SD across 30 parent--run units; direct-RNA score is a held-out computational proxy.", transform=axes[0].transAxes, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
