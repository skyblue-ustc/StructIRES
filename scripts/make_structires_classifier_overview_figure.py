#!/usr/bin/env python3
"""Render a faithful overview of the validation-clean StructIRES pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from publication_style import ANCHOR, ENERGY, ENSEMBLE, INK, NAVY, STRUCTIRES, apply_style

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures" / "fig1_structires_classifier"


def box(ax, x, y, w, h, title, body, edge, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.024",
                                linewidth=1.2, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + .022, y + h - .052, title, fontsize=7.6, color=edge, weight="bold", va="top", zorder=3)
    ax.text(x + .022, y + h - .135, "\n".join(body), fontsize=5.9, color=INK, va="top", linespacing=1.45, zorder=3)


def arrow(ax, start, end, colour=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.05, color=colour, zorder=1))


def main() -> None:
    apply_style(plt)
    fig, ax = plt.subplots(figsize=(7.35, 4.65))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.text(.02, .968, "StructIRES: structure-aware IRES recognition", fontsize=9.5,
            color=INK, weight="bold", va="top")
    ax.text(.02, .922, "Public RNA-FM is initialized from generic pretraining and jointly fine-tuned with the author-style IRES objective.",
            fontsize=6.4, color="#4B5563", va="top")

    box(ax, .025, .57, .16, .20, "RNA input", ["IRES candidate", "up to 1,024 nt"], NAVY, "#F6F9FC")
    box(ax, .245, .60, .22, .20, "Sequence encoder", ["public RNA-FM t12", "BOS: 640 → 40"], NAVY, "#F6F9FC")
    box(ax, .245, .34, .22, .19, "Structure encoder", ["ViennaRNA MFE pair graph", "contextual pair pooling"], ENSEMBLE, "#F2FBFA")
    box(ax, .545, .54, .22, .22, "Gated fusion", ["trainable sequence + graph", "structural residual"], STRUCTIRES, "#F3FBF8")
    box(ax, .83, .57, .145, .20, "Output", ["functional-IRES", "probability"], NAVY, "#F6F9FC")
    arrow(ax, (.185, .68), (.24, .70), NAVY)
    arrow(ax, (.185, .63), (.24, .44), ENSEMBLE)
    arrow(ax, (.465, .70), (.54, .66), NAVY)
    arrow(ax, (.465, .44), (.54, .60), ENSEMBLE)
    arrow(ax, (.765, .65), (.825, .68), STRUCTIRES)
    ax.text(.655, .462, r"$g=\sigma(W[h_{seq};h_{str}])$", fontsize=6.7, color=STRUCTIRES, ha="center")
    ax.text(.655, .430, r"$h=h_{seq}+g\odot h_{str}$", fontsize=7.1, color=INK, ha="center")
    ax.text(.655, .397, "contact residual is zero-initialized", fontsize=5.9, color="#4B5563", ha="center")

    ax.plot([.025, .975], [.27, .27], color="#D1D5DB", linewidth=.8)
    ax.text(.025, .225, "Validation-clean optimisation and evaluation", fontsize=7.5, color=INK, weight="bold")
    box(ax, .025, .010, .23, .210, "Training records", ["native source-train only", "stratified validation"], NAVY, "#F6F9FC")
    box(ax, .395, .010, .26, .210, "Joint objective", ["2 × IRES classification", "+ 1 × masked-LM loss"], STRUCTIRES, "#F3FBF8")
    box(ax, .80, .010, .175, .210, "Locked test", ["AUROC, AUPR, F1", "one final evaluation"], ENERGY, "#FFF9ED")
    arrow(ax, (.255, .115), (.385, .115), NAVY)
    arrow(ax, (.655, .115), (.79, .115), STRUCTIRES)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
