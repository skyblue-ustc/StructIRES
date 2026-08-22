#!/usr/bin/env python3
"""Render the matched StructIRES-Rank constraint ablation in a common visual style."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from publication_style import METHOD_COLORS, apply_style, clean_axis, panel_label


LABELS = {
    "ireslm_score_only": "Score-only",
    "ireslm_plus_energy": "+ energy",
    "ireslm_plus_ensemble": "+ ensemble",
    "ireslm_plus_anchor": "+ anchors",
    "structires": "StructIRES-Rank",
}
ORDER = list(LABELS)


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def add_bars(axis, values, errors, title, note, *, higher=False):
    positions = range(len(ORDER))
    axis.bar(positions, values, yerr=errors, capsize=2.5,
             color=[METHOD_COLORS[key] for key in ORDER], edgecolor="white", linewidth=0.8, zorder=2)
    for x, value, error in zip(positions, values, errors):
        axis.text(x, value + error, f"{value:.3f}", ha="center", va="bottom", fontsize=6.4, color="#374151")
    axis.set_title(title, fontsize=9.5, loc="left", weight="bold")
    axis.text(0.0, 0.98, note, transform=axis.transAxes, va="top", fontsize=7.2, color="#4B5563")
    axis.set_xticks(list(positions), [LABELS[key] for key in ORDER], rotation=28, ha="right")
    if higher:
        axis.set_ylim(0, max(1.05, max(value + error for value, error in zip(values, errors)) * 1.17))
    else:
        axis.set_ylim(0, max(value + error for value, error in zip(values, errors)) * 1.28)
    clean_axis(axis)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt)
    structure, proxy = read_rows(args.structure), read_rows(args.proxy)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axes = axes.ravel()
    panels = [
        ("mfe_abs_delta_kcal_mol", r"$|\Delta\mathrm{MFE}|$ (kcal mol$^{-1}$)", "lower is better", False),
        ("pairing_profile_l1", r"Pairing-profile $L_1$", "lower is better", False),
        ("parent_ensemble_anchor_retention", "Parent-anchor retention", "higher is better", True),
    ]
    for index, (metric, title, note, higher) in enumerate(panels):
        values = [float(structure[key][f"{metric}_mean"]) for key in ORDER]
        errors = [float(structure[key][f"{metric}_std"]) for key in ORDER]
        add_bars(axes[index], values, errors, title, note, higher=higher)
        panel_label(axes[index], chr(65 + index))
    values = [float(proxy[key]["mean_direct_rna_s3_heldout_proxy"]) for key in ORDER]
    add_bars(axes[3], values, [0.0] * len(values), "Held-out direct-RNA proxy", "S3-held-out; post-selection only", higher=True)
    panel_label(axes[3], "D")
    fig.text(0.5, -0.02,
             "All arms select the top 50 candidates from the same 512-variant parent-by-pool set (30 matched units). Error bars: sample SD; proxy is computational, not an activity assay.",
             ha="center", fontsize=7.5, color="#4B5563")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
