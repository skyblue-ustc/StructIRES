"""Small, dependency-free visual system for manuscript figures.

The palette is colour-vision-friendly and method semantics are fixed across figures.
Keep this module local and explicit rather than importing an unpinned plotting theme.
"""
from __future__ import annotations

from typing import Iterable


INK = "#1F2937"
MUTED = "#8A8A8A"
GRID = "#D9E0E7"
PAPER = "#FFFFFF"
SCORE = "#8A8A8A"
ENERGY = "#E69F00"
ENSEMBLE = "#56B4E9"
ANCHOR = "#CC79A7"
STRUCTIRES = "#009E73"
RISK = "#D55E00"
NAVY = "#3B5B7A"

METHOD_COLORS = {
    "ireslm_score_only": SCORE,
    "ireslm_plus_energy": ENERGY,
    "ireslm_plus_ensemble": ENSEMBLE,
    "ireslm_plus_anchor": ANCHOR,
    "structires": STRUCTIRES,
}


def apply_style(plt) -> None:
    """Apply deterministic, journal-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.edgecolor": "#9AA7B5",
            "axes.linewidth": 0.65,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": PAPER,
            "savefig.transparent": False,
        }
    )


def clean_axis(axis, *, grid_y: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=3, width=0.6, labelsize=8)
    if grid_y:
        axis.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8, zorder=0)
        axis.set_axisbelow(True)


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.14,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )


def colour_for_methods(methods: Iterable[str]) -> list[str]:
    return [METHOD_COLORS.get(method, MUTED) for method in methods]
