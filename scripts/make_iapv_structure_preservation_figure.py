#!/usr/bin/env python3
"""Render a compact, provenance-locked IAPV structural-preservation case.

This main-text figure deliberately contains only the parent and two matched
rank-1 candidates.  It visualizes predicted ViennaRNA secondary structures and
parent-derived ensemble anchors; it does not assert that these are experimentally
mapped functional IRES motifs or that either candidate has measured activity.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ires_design.structure import dot_bracket_pairs, ensemble_anchor_pairs, fold_ensemble
from publication_style import INK, RISK, STRUCTIRES, apply_style, panel_label


METHODS = ("ireslm_score_only", "structires")
TITLES = {"ireslm_score_only": "IRES-LM score-only", "structires": "StructIRES-Rank"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_sequences(pool: Path) -> dict[str, str]:
    return {
        row["candidate_id"]: row["sequence"]
        for row in (json.loads(line) for line in (pool / "candidates.jsonl").open(encoding="utf-8"))
    }


def xy_for_structure(structure: str) -> np.ndarray:
    import RNA

    coordinates = RNA.naview_xy_coordinates(structure)
    return np.asarray(
        [(coordinates[index + 1].X, coordinates[index + 1].Y) for index in range(len(structure))],
        dtype=float,
    )


def draw_structure(axis, fold, anchors, mutations: set[int], *, parent: bool, title: str) -> None:
    """Draw MFE topology plus inherited ensemble-anchor state."""
    coords = xy_for_structure(fold.mfe_structure)
    axis.plot(coords[:, 0], coords[:, 1], color="#AAB4C0", linewidth=0.65, zorder=1)
    for left, right in dot_bracket_pairs(fold.mfe_structure):
        axis.plot(
            (coords[left, 0], coords[right, 0]),
            (coords[left, 1], coords[right, 1]),
            color="#CCD4DD",
            linewidth=0.52,
            zorder=1,
        )
    candidate_probabilities = {(left, right): value for left, right, value in fold.base_pair_probabilities}
    retained_positions: set[int] = set()
    reduced_positions: set[int] = set()
    for left, right, parent_probability in anchors:
        retained = parent or candidate_probabilities.get((left, right), 0.0) >= parent_probability
        (retained_positions if retained else reduced_positions).update((left, right))
        axis.plot(
            (coords[left, 0], coords[right, 0]),
            (coords[left, 1], coords[right, 1]),
            # Use a high-contrast teal rather than the global anchor palette here:
            # in this case figure, colour encodes retained versus reduced state.
            color=STRUCTIRES if retained else RISK,
            linestyle="-" if retained else (0, (2, 1.5)),
            alpha=0.95,
            linewidth=1.45,
            zorder=2,
        )
    colours = [RISK if index in mutations else "#FFFFFF" for index in range(len(fold.sequence))]
    edges = [RISK if index in mutations else "#536273" for index in range(len(fold.sequence))]
    axis.scatter(coords[:, 0], coords[:, 1], s=11, c=colours, edgecolors=edges, linewidths=0.38, zorder=3)
    # Chords alone can be hidden by dense nucleotide nodes; outline the paired
    # nucleotides as well so anchor state survives two-column downscaling.
    if retained_positions:
        retained_index = sorted(retained_positions)
        axis.scatter(
            coords[retained_index, 0],
            coords[retained_index, 1],
            s=23,
            facecolors="none",
            edgecolors=STRUCTIRES,
            linewidths=0.75,
            zorder=3.5,
        )
    if reduced_positions:
        reduced_index = sorted(reduced_positions)
        axis.scatter(
            coords[reduced_index, 0],
            coords[reduced_index, 1],
            s=16,
            marker="x",
            color=RISK,
            linewidths=0.75,
            zorder=4,
        )
    # Only label endpoints: denser numbering obscures the 213-nt layout at
    # two-column print scale without adding interpretive value.
    for index in (0, len(fold.sequence) - 1):
        axis.text(
            coords[index, 0],
            coords[index, 1],
            str(index + 1),
        fontsize=5.5,
            ha="center",
            va="center",
            color=INK,
            zorder=4,
        )
    axis.set_title(title, loc="left", fontsize=9.2, weight="bold", color=INK, pad=3)
    axis.set_aspect("equal")
    axis.margins(0.08)
    axis.axis("off")


def chosen_rows(selected: Path, parent: str, run_seed: str) -> dict[str, dict[str, str]]:
    rows = read_csv(selected)
    result: dict[str, dict[str, str]] = {}
    for method in METHODS:
        matches = [
            row
            for row in rows
            if row["parent_id"] == parent
            and row["run_seed"] == run_seed
            and row["method"] == method
            and row["selection_rank"] == "1"
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one rank-1 {method} row, got {len(matches)}")
        result[method] = matches[0]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--seed-panel", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", default="IAPV")
    parser.add_argument("--run-seed", default="42")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt)
    selected = chosen_rows(args.selected, args.parent, str(args.run_seed))
    seed_panel = json.loads(args.seed_panel.read_text(encoding="utf-8"))
    seed = next(row for row in seed_panel["records"] if row["seed_id"] == args.parent)
    sequences = read_sequences(args.pool)
    parent_fold = fold_ensemble(seed["sequence"])
    anchors = ensemble_anchor_pairs(parent_fold, min_probability=0.50)
    # Plot the highest-confidence subset so individual structural features
    # remain interpretable at conference two-column scale.  Quantitative
    # retention metrics below still use all parent anchors.
    displayed_anchors = sorted(anchors, key=lambda item: item[2], reverse=True)[:16]
    folds = {method: fold_ensemble(sequences[row["candidate_id"]]) for method, row in selected.items()}
    mutations = {
        method: {
            index
            for index, (parent_base, candidate_base) in enumerate(
                zip(seed["sequence"], sequences[row["candidate_id"]])
            )
            if parent_base != candidate_base
        }
        for method, row in selected.items()
    }

    # A deliberately tall full-width panel: the structures must remain readable
    # after IEEE two-column placement, rather than behaving like a thumbnail.
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 4.35), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.19, wspace=0.10)
    draw_structure(
        axes[0], parent_fold, displayed_anchors, set(), parent=True,
        title="Parent IAPV",
    )
    for axis, method in zip(axes[1:], METHODS):
        row = selected[method]
        title = TITLES[method]
        draw_structure(axis, folds[method], displayed_anchors, mutations[method], parent=False, title=title)
    for axis, label in zip(axes, ("A", "B", "C")):
        panel_label(axis, label)

    metric_text = (
        f"Parent  |  213 nt  |  {len(anchors)} anchors"
        f"     Score-only  |  IRES-LM {float(selected['ireslm_score_only']['ireslm_probability_mean']):.3f}"
        f"  |  |ΔMFE| {float(selected['ireslm_score_only']['mfe_abs_delta_kcal_mol']):.2f}"
        f"  |  anchor retention {float(selected['ireslm_score_only']['parent_ensemble_anchor_retention']):.3f}"
        f"     StructIRES-Rank  |  IRES-LM {float(selected['structires']['ireslm_probability_mean']):.3f}"
        f"  |  |ΔMFE| {float(selected['structires']['mfe_abs_delta_kcal_mol']):.2f}"
        f"  |  anchor retention {float(selected['structires']['parent_ensemble_anchor_retention']):.3f}"
    )
    fig.text(
        0.5,
        0.125,
        metric_text,
        ha="center",
        va="center",
        fontsize=6.15,
        color=INK,
    )
    fig.text(
        0.5,
        0.076,
        "Top 16 parent anchors by ensemble probability: teal outlines = retained; dashed orange / × = reduced; red nodes = edits",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#435466",
    )
    fig.text(
        0.5,
        0.030,
        "ViennaRNA 2.7.2 MFE layouts. Anchors are thermodynamic ensemble features (parent BPP ≥ 0.5), not experimentally mapped IRES motifs.",
        ha="center",
        va="center",
        fontsize=6.9,
        color="#5D6B7A",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.output.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
