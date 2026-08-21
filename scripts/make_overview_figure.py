#!/usr/bin/env python3
"""Create the manuscript's vector overview figure from repository-native code."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures" / "fig1_overview"


def box(axis, xy, width, height, title, lines, color):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
    )
    axis.add_patch(patch)
    x, y = xy
    axis.text(x + width / 2, y + height * 0.78, title, ha="center", va="center",
              fontsize=11, fontweight="bold", color=color)
    axis.text(x + width / 2, y + height * 0.40, "\n".join(lines), ha="center", va="center",
              fontsize=8.7, color="#263238", linespacing=1.35)


def arrow(axis, start, end):
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.3, color="#546E7A")
    )


def main() -> None:
    figure, axis = plt.subplots(figsize=(12.5, 4.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    box(axis, (0.02, 0.19), 0.20, 0.66, "Assay-labelled evidence",
        ["Legacy DNA/lentiviral MPRA", "Direct-RNA IRES-TrAPPr", "In-cell structure atlas",
         "IRES--cargo measurements"], "#315A7D")
    box(axis, (0.28, 0.19), 0.19, 0.66, "Task 0: audit",
        ["Native prediction benchmark", "Cluster/source holdouts", "Cross-assay calibration",
         "Applicability distance"], "#8A5A2B")
    box(axis, (0.53, 0.19), 0.20, 0.66, "Task 1: optimize",
        ["Full-length viral seeds", "Function lower confidence bound", "Ensemble preservation",
         "Cargo crosstalk + diversity"], "#7A3E65")
    box(axis, (0.79, 0.19), 0.19, 0.66, "Independent evaluation",
        ["Assay transfer", "Structural robustness", "Worst-case cargo context",
         "Validity, novelty, efficiency"], "#35735C")

    arrow(axis, (0.225, 0.52), (0.275, 0.52))
    arrow(axis, (0.475, 0.52), (0.525, 0.52))
    arrow(axis, (0.735, 0.52), (0.785, 0.52))
    axis.text(0.5, 0.06,
              "Optimization scorer is never the sole final evaluator; all comparisons use matched budgets.",
              ha="center", va="center", fontsize=9, color="#37474F")
    figure.tight_layout()
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
