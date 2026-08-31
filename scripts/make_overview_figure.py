#!/usr/bin/env python3
"""Render the high-level, print-readable StructIRES study overview."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from publication_style import ENERGY, INK, NAVY, RISK, STRUCTIRES, apply_style

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures" / "fig1_overview"
MUTED = "#52616F"


def box(ax, x, y, w, h, title, lines, colour, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012,rounding_size=.026",
                                linewidth=1.25, edgecolor=colour, facecolor=fill, zorder=2))
    ax.text(x + .025, y + h - .055, title, ha="left", va="top", fontsize=9.2,
            weight="bold", color=colour, zorder=3)
    ax.text(x + .025, y + h - .145, "\n".join(lines), ha="left", va="top", fontsize=7.1,
            color=INK, linespacing=1.42, zorder=3)


def arrow(ax, start, end, colour=NAVY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.45, color=colour, zorder=1))


def title(ax, tag, heading, subheading):
    ax.text(.02, .97, tag, fontsize=10.2, weight="bold", color="white", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=.18", facecolor=INK, edgecolor=INK))
    ax.text(.09, .975, heading, fontsize=10.2, weight="bold", color=INK, ha="left", va="top")
    ax.text(.09, .915, subheading, fontsize=6.8, color=MUTED, ha="left", va="top")


def main() -> None:
    apply_style(plt)
    # Sized at final IEEE two-column width: no downscaling of typography.
    fig = plt.figure(figsize=(7.35, 6.55), layout="constrained")
    grid = fig.add_gridspec(3, 1, hspace=.035)

    ax = fig.add_subplot(grid[0]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    title(ax, "A", "Recognize IRES-like sequences with structure-aware fusion",
          "Classifier training uses native source-train records only; structural contacts are label-free.")
    box(ax, .03, .24, .20, .42, "RNA sequence", ["candidate IRES", "up to 1,024 nt"], NAVY, "#F4F8FB")
    box(ax, .39, .24, .25, .42, "StructIRES-Classifier", ["RNA-FM sequence representation", "+ ViennaRNA MFE contacts", "+ gated residual fusion"], STRUCTIRES, "#EFF9F5")
    box(ax, .80, .24, .17, .42, "Output", ["IRES-like", "probability"], NAVY, "#F4F8FB")
    arrow(ax, (.23, .45), (.38, .45)); arrow(ax, (.64, .45), (.79, .45), STRUCTIRES)
    ax.text(.515, .125, "joint objective: weighted IRES classification + masked-LM regularization", fontsize=6.8,
            color=MUTED, ha="center")

    ax = fig.add_subplot(grid[1]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    title(ax, "B", "Optimize local IRES variants under matched structural constraints",
          "All methods select from the same length-preserving, fixed-budget candidate pool.")
    box(ax, .03, .24, .20, .42, "Parent IRES", ["full-length", "supported sequence"], NAVY, "#F4F8FB")
    box(ax, .31, .24, .20, .42, "Shared pool", ["512 substitutions", "at most 3% edits"], NAVY, "#F4F8FB")
    box(ax, .59, .16, .27, .58, "StructIRES-Rank", ["proposal score", "energy deviation", "ensemble-profile distance", "parent-anchor retention"], STRUCTIRES, "#EFF9F5")
    box(ax, .90, .24, .08, .42, "Top 50", ["per", "pool"], STRUCTIRES, "#EFF9F5")
    arrow(ax, (.23, .45), (.30, .45)); arrow(ax, (.51, .45), (.58, .45)); arrow(ax, (.86, .45), (.89, .45), STRUCTIRES)
    ax.text(.725, .095, "Score-only and each ablation have identical parents, pools, edit budget and retained-set size.", fontsize=6.55,
            color=MUTED, ha="center")

    ax = fig.add_subplot(grid[2]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    title(ax, "C", "Report structural preservation and assay-qualified external evidence",
          "Direct-RNA observations are held out from candidate selection and do not establish new-candidate activity.")
    box(ax, .03, .24, .26, .42, "Structural endpoints", [r"|ΔMFE| energy deviation", "pairing-profile distance", "anchor retention"], ENERGY, "#FFF8E9")
    box(ax, .37, .24, .26, .42, "Held-out proxy", ["direct-RNA 3–6-mer model", "test once after selection", "computational enrichment only"], NAVY, "#F4F8FB")
    box(ax, .71, .24, .26, .42, "IAPV MPRA guardrail", ["observed tile-loss by edit", "IAPV-specific risk rank", "not activity prediction"], RISK, "#FFF3EE")
    ax.text(.50, .10, "Claim boundary: predicted structural preservation and in-silico candidate enrichment; experimental validation remains future work.",
            fontsize=6.55, color=MUTED, ha="center")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
