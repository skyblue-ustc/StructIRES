#!/usr/bin/env python3
"""Draw a provenance-locked IAPV secondary-structure and MPRA guardrail figure.

The layouts are ViennaRNA NaviView layouts of each MFE structure.  The lower panel
uses the public IRES-TrAPPr IAPV A-substitution scan only as an edit-risk map; it
does not assign functional activity to generated candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ires_design.structure import dot_bracket_pairs, ensemble_anchor_pairs, fold_ensemble
from publication_style import ANCHOR, INK, RISK, SCORE, STRUCTIRES, apply_style, clean_axis, panel_label


METHODS = ("ireslm_score_only", "structires")
LABELS = {"ireslm_score_only": "IRES-LM score-only", "structires": "StructIRES-Rank"}


def read_sequences(pool: Path) -> dict[str, str]:
    return {row["candidate_id"]: row["sequence"] for row in
            (json.loads(line) for line in (pool / "candidates.jsonl").open(encoding="utf-8"))}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def xy_for_structure(structure: str) -> np.ndarray:
    import RNA
    coordinates = RNA.naview_xy_coordinates(structure)
    # ViennaRNA returns a sentinel at index 0 followed by 1-based nucleotide coordinates.
    return np.asarray([(coordinates[index + 1].X, coordinates[index + 1].Y)
                       for index in range(len(structure))], dtype=float)


def draw_structure(axis, fold, parent_anchors, *, title: str, mutations: set[int], parent: bool) -> None:
    coords = xy_for_structure(fold.mfe_structure)
    # Backbone and MFE base pairs.
    axis.plot(coords[:, 0], coords[:, 1], color="#B8C2CC", linewidth=0.55, zorder=1)
    for left, right in dot_bracket_pairs(fold.mfe_structure):
        axis.plot((coords[left, 0], coords[right, 0]), (coords[left, 1], coords[right, 1]),
                  color="#AAB5C0", linewidth=0.55, zorder=1)
    probabilities = {(left, right): value for left, right, value in fold.base_pair_probabilities}
    # Parent anchors are ensemble-derived (not known motifs).  A retained line has at least
    # the parent's anchor probability; a reduced line is rendered in orange/red.
    for left, right, probability in parent_anchors:
        retained = parent or probabilities.get((left, right), 0.0) >= probability
        colour = STRUCTIRES if retained else RISK
        style = "-" if retained else "--"
        axis.plot((coords[left, 0], coords[right, 0]), (coords[left, 1], coords[right, 1]),
                  color=colour, linestyle=style, alpha=0.50, linewidth=0.8, zorder=2)
    node_colours = [RISK if index in mutations else "#F7F9FB" for index in range(len(fold.sequence))]
    edge_colours = [RISK if index in mutations else "#6B7280" for index in range(len(fold.sequence))]
    axis.scatter(coords[:, 0], coords[:, 1], s=8, c=node_colours, edgecolors=edge_colours,
                 linewidths=0.35, zorder=3)
    # Labels remain sparse so 213 nt is legible at paper scale.
    for index in range(0, len(fold.sequence), 25):
        axis.text(coords[index, 0], coords[index, 1], str(index + 1), fontsize=5.5,
                  ha="center", va="center", color=INK, zorder=4)
    axis.set_title(title, loc="left", fontsize=9, weight="bold", color=INK, pad=5)
    axis.set_aspect("equal")
    axis.axis("off")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--seed-panel", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--position-risk", type=Path, required=True)
    parser.add_argument("--guardrail-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", default="IAPV")
    parser.add_argument("--run-seed", default="42")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt)
    rows = read_csv(args.selected)
    chosen = {}
    for method in METHODS:
        matches = [row for row in rows if row["parent_id"] == args.parent and row["run_seed"] == str(args.run_seed)
                   and row["method"] == method and row["selection_rank"] == "1"]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {method} illustrative row; got {len(matches)}")
        chosen[method] = matches[0]
    seed_panel = json.loads(args.seed_panel.read_text(encoding="utf-8"))
    seed = next(row for row in seed_panel["records"] if row["seed_id"] == args.parent)
    sequences = read_sequences(args.pool)
    parent_fold = fold_ensemble(seed["sequence"])
    anchors = ensemble_anchor_pairs(parent_fold, min_probability=0.50)
    folds = {method: fold_ensemble(sequences[row["candidate_id"]]) for method, row in chosen.items()}
    mutations = {method: {index for index, (left, right) in enumerate(zip(seed["sequence"], sequences[row["candidate_id"]])) if left != right}
                 for method, row in chosen.items()}

    fig = plt.figure(figsize=(7.2, 6.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(3.0, 1.35))
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    draw_structure(axes[0], parent_fold, anchors, title=f"Parent IAPV\n213 nt; {len(anchors)} ensemble anchors", mutations=set(), parent=True)
    for index, method in enumerate(METHODS, start=1):
        row = chosen[method]
        title = (f"{LABELS[method]}\nIRES-LM {float(row['ireslm_probability_mean']):.3f}; "
                 f"|ΔMFE| {float(row['mfe_abs_delta_kcal_mol']):.2f}; "
                 f"anchors {float(row['parent_ensemble_anchor_retention']):.3f}")
        draw_structure(axes[index], folds[method], anchors, title=title, mutations=mutations[method], parent=False)
    for index, label in enumerate(("A", "B", "C")):
        panel_label(axes[index], label)

    risk_axis = fig.add_subplot(grid[1, :2])
    risk_rows = [row for row in read_csv(args.position_risk) if row["parent_id"] == args.parent]
    x = np.asarray([int(row["position_0based"]) + 1 for row in risk_rows])
    y = np.asarray([float(row["conservative_edit_risk"]) for row in risk_rows])
    observed = np.asarray([row["risk_source"].startswith("S1 direct") for row in risk_rows])
    risk_axis.plot(x, y, color="#B6C1CC", linewidth=0.8, zorder=1)
    risk_axis.scatter(x[observed], y[observed], s=8, color=RISK, label="observed direct-RNA tile loss", zorder=2)
    risk_axis.scatter(x[~observed], y[~observed], s=7, color="#C9D1D9", label="median-imputed risk", zorder=2)
    for method, colour, offset in (("ireslm_score_only", SCORE, 0.0), ("structires", STRUCTIRES, 0.10)):
        for position in mutations[method]:
            risk_axis.vlines(position + 1 + offset, 0, y[position], color=colour, linewidth=1.0, alpha=0.9, zorder=3)
    risk_axis.set(title="IAPV public direct-RNA MPRA mutational-risk map", xlabel="IAPV position", ylabel="edit risk")
    risk_axis.text(0.99, 0.95, "vertical ticks: edits in illustrative candidates\ngray = score-only; teal = StructIRES-Rank",
                   transform=risk_axis.transAxes, ha="right", va="top", fontsize=7, color="#4B5563")
    risk_axis.legend(loc="upper left", fontsize=6.6, ncol=2)
    clean_axis(risk_axis)
    panel_label(risk_axis, "D")

    summary_axis = fig.add_subplot(grid[1, 2])
    summary = {row["method"]: row for row in read_csv(args.guardrail_summary)}
    methods = ("structires", "structires_plus_iapv_mpra_guardrail")
    names = ("StructIRES-Rank", "+ MPRA guardrail")
    risks = [float(summary[method]["iapv_mpra_mutational_risk_mean"]) for method in methods]
    errors = [float(summary[method]["iapv_mpra_mutational_risk_std"]) for method in methods]
    summary_axis.bar(range(2), risks, yerr=errors, capsize=2.5, color=[STRUCTIRES, RISK], edgecolor="white", zorder=2)
    for index, value in enumerate(risks):
        summary_axis.text(index, value + errors[index], f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    summary_axis.set_xticks(range(2), names, rotation=22, ha="right")
    summary_axis.set(title="Matched IAPV selection", ylabel="mean MPRA edit risk")
    summary_axis.text(0.0, 0.96, "lower is better; 3 frozen pools", transform=summary_axis.transAxes,
                      va="top", fontsize=6.8, color="#4B5563")
    clean_axis(summary_axis)
    panel_label(summary_axis, "E")
    fig.text(0.5, 0.005,
             "MFE layouts: ViennaRNA 2.7.2. Green lines denote retained parent-derived ensemble anchors; dashed orange lines denote reduced anchors; red nodes are edits. The case is illustrative and the MPRA map is an edit-risk guardrail, not a measurement of generated-candidate activity.",
             ha="center", fontsize=7.2, color="#4B5563")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
