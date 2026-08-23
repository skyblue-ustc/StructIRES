#!/usr/bin/env python3
"""Render the publication-style StructIRES-Rank overview (Figure 1)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from publication_style import ANCHOR, ENERGY, ENSEMBLE, INK, NAVY, RISK, STRUCTIRES, apply_style


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures" / "fig1_overview"


def rounded_box(axis, xy, width, height, title, lines, colour, *, fill="#FFFFFF"):
    x, y = xy
    axis.add_patch(FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.025",
                                  linewidth=1.25, edgecolor=colour, facecolor=fill, zorder=2))
    axis.text(x + 0.025, y + height - 0.06, title, ha="left", va="top", fontsize=7.8,
              weight="bold", color=colour, zorder=3)
    axis.text(x + 0.025, y + height - 0.15, "\n".join(lines), ha="left", va="top", fontsize=6.1,
              color=INK, linespacing=1.45, zorder=3)


def arrow(axis, start, end, *, colour=INK):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                   linewidth=1.0, color=colour, zorder=1))


def main() -> None:
    apply_style(plt)
    fig, axis = plt.subplots(figsize=(7.2, 5.25))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.text(0.02, 0.965, "Assay-aware, structure-robust full-length IRES optimization",
              fontsize=10.5, weight="bold", color=INK, va="top")
    axis.text(0.02, 0.915,
              "Frozen public evidence informs constraints; every selection arm ranks the same candidate pool under the same budget.",
              fontsize=6.8, color="#4B5563", va="top")
    rounded_box(axis, (0.025, 0.585), 0.205, 0.24, "1  Assay-qualified evidence", [
        "Legacy IRES-LM classifiers", "Direct-RNA IRES-TrAPPr labels", "IAPV mutational MPRA scan",
    ], NAVY, fill="#F6F9FC")
    rounded_box(axis, (0.275, 0.585), 0.205, 0.24, "2  Frozen design pool", [
        "10 supported full-length parents", "3 deterministic pools", "512 length-preserving variants / unit",
    ], STRUCTIRES, fill="#F3FBF8")
    rounded_box(axis, (0.525, 0.535), 0.23, 0.34, "3  StructIRES-Rank", [
        "Released IRES-LM ensemble", "Equal-rank structural objectives", "Energy, ensemble and parent anchors",
    ], STRUCTIRES, fill="#F3FBF8")
    rounded_box(axis, (0.80, 0.585), 0.175, 0.24, "4  Matched selection", [
        "Top 50 / parent / pool", "Same variants and query budget", "Five-arm ablation",
    ], NAVY, fill="#F6F9FC")
    arrow(axis, (0.23, 0.705), (0.27, 0.705), colour=NAVY)
    arrow(axis, (0.48, 0.705), (0.52, 0.705), colour=STRUCTIRES)
    arrow(axis, (0.755, 0.705), (0.795, 0.705), colour=NAVY)
    for index, (text, colour) in enumerate((("energy", ENERGY), ("ensemble", ENSEMBLE), ("parent anchors", ANCHOR))):
        x = 0.543 + index * 0.067
        axis.add_patch(FancyBboxPatch((x, 0.56), 0.06, 0.042, boxstyle="round,pad=0.008,rounding_size=0.014",
                                      linewidth=0.6, edgecolor=colour, facecolor="white", zorder=4))
        axis.text(x + 0.03, 0.581, text, fontsize=5.8, ha="center", va="center", color=colour, zorder=5)
    axis.text(0.025, 0.46, "Independent, assay-qualified evaluation", fontsize=8.3, weight="bold", color=INK)
    rounded_box(axis, (0.025, 0.12), 0.285, 0.26, "Structure preservation", [
        "|ΔMFE|; pairing-profile distance", "parent-anchor retention", "structure-state identity",
    ], ENERGY, fill="#FFF9ED")
    rounded_box(axis, (0.358, 0.12), 0.285, 0.26, "Post-selection direct-RNA proxy", [
        "Train: non-S3 provenance records", "Evaluate once: held-out S3", "Not used for selection or tuning",
    ], NAVY, fill="#F6F9FC")
    rounded_box(axis, (0.69, 0.12), 0.285, 0.26, "IAPV MPRA edit-risk guardrail", [
        "Observed mutational-loss map", "IAPV-only constrained selection", "Not a candidate activity measurement",
    ], RISK, fill="#FFF5F1")
    for start, end, colour in [((0.887, 0.585), (0.17, 0.38), ENERGY), ((0.887, 0.585), (0.50, 0.38), NAVY), ((0.887, 0.585), (0.83, 0.38), RISK)]:
        arrow(axis, start, end, colour=colour)
    axis.text(0.5, 0.035, "Scope: computational candidate enrichment and structural preservation; no new-candidate activity is claimed.",
              fontsize=6.2, color="#4B5563", ha="center")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
