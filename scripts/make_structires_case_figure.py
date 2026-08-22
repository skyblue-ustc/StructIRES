#!/usr/bin/env python3
"""Render a provenance-locked IAPV case study for score-only vs StructIRES.

This is an illustrative structural case, selected deterministically as rank 1 in the frozen
IAPV/seed-42 pool for each method. It is not a functional validation of either candidate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ires_design.structure import ensemble_anchor_pairs, fold_ensemble, weighted_anchor_retention


METHODS = ("ireslm_score_only", "structires")
LABELS = {"ireslm_score_only": "IRES-LM score-only", "structires": "StructIRES-Rank"}
COLORS = {"ireslm_score_only": "#7f7f7f", "structires": "#0072b2"}


def read_sequences(pool: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for line in (pool / "candidates.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        sequences[row["candidate_id"]] = row["sequence"]
    return sequences


def plot_pairs(axis, fold, parent_anchors, title: str, mutation_positions: list[int], *, candidate: bool) -> None:
    # Show the ensemble probability landscape sparsely, then overlay the parent-defined anchors.
    pairs = [(i, j, p) for i, j, p in fold.base_pair_probabilities if p >= 0.10]
    if pairs:
        x, y, c = zip(*pairs)
        scatter = axis.scatter(x, y, c=c, s=8, cmap="Blues", vmin=0.10, vmax=1.0, marker="s", linewidths=0)
        axis.figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03, label="base-pair probability")
    anchor_map = {(i, j): p for i, j, p in fold.base_pair_probabilities}
    retained_x, retained_y, lost_x, lost_y = [], [], [], []
    for i, j, probability in parent_anchors:
        if candidate and anchor_map.get((i, j), 0.0) < probability:
            lost_x.append(i); lost_y.append(j)
        else:
            retained_x.append(i); retained_y.append(j)
    axis.scatter(retained_x, retained_y, s=18, facecolors="none", edgecolors="#009e73", linewidths=0.8, label="parent anchor retained")
    if lost_x:
        axis.scatter(lost_x, lost_y, s=18, marker="x", color="#d55e00", linewidths=0.9, label="parent anchor reduced")
    for position in mutation_positions:
        axis.axvline(position, color="#cc79a7", alpha=0.28, linewidth=0.6)
    axis.set(xlim=(0, len(fold.sequence)), ylim=(len(fold.sequence), 0), xlabel="5′ nucleotide position", ylabel="3′ nucleotide position", title=title)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=7)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--seed-panel", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", default="IAPV")
    parser.add_argument("--run-seed", default="42")
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite case-study outputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    selected = list(csv.DictReader(args.selected.open(encoding="utf-8", newline="")))
    chosen = {}
    for method in METHODS:
        rows = [row for row in selected if row["parent_id"] == args.parent and row["run_seed"] == str(args.run_seed)
                and row["method"] == method and row["selection_rank"] == "1"]
        if len(rows) != 1:
            raise ValueError(f"expected one rank-1 {method} row; got {len(rows)}")
        chosen[method] = rows[0]
    panel = json.loads(args.seed_panel.read_text(encoding="utf-8"))
    seed = next(row for row in panel["records"] if row["seed_id"] == args.parent)
    sequences = read_sequences(args.pool)
    parent_fold = fold_ensemble(seed["sequence"])
    anchors = ensemble_anchor_pairs(parent_fold, min_probability=0.50)
    folds = {method: fold_ensemble(sequences[row["candidate_id"]]) for method, row in chosen.items()}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.3), constrained_layout=True)
    plot_pairs(axes[0], parent_fold, anchors, f"Parent IAPV ({len(anchors)} anchors)", [], candidate=False)
    for axis, method in zip(axes[1:], METHODS):
        row, fold = chosen[method], folds[method]
        sequence = sequences[row["candidate_id"]]
        mutations = [index for index, (a, b) in enumerate(zip(seed["sequence"], sequence)) if a != b]
        retention = weighted_anchor_retention(anchors, fold.base_pair_probabilities)
        title = (f"{LABELS[method]}\n"
                 f"IRES-LM={float(row['ireslm_probability_mean']):.3f}; |ΔMFE|={float(row['mfe_abs_delta_kcal_mol']):.2f}\n"
                 f"anchor retention={retention:.3f}; {len(mutations)} edits")
        plot_pairs(axis, fold, anchors, title, mutations, candidate=True)
    axes[1].legend(loc="lower left", fontsize=7, frameon=False)
    fig.text(0.02, 0.01, "Each panel shows ViennaRNA ensemble base-pair probabilities ≥0.10. Green circles mark retained parent-derived high-confidence anchors; orange crosses mark anchors whose probability decreased. Pink lines mark edited positions.", fontsize=8)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    manifest = {
        "schema_version": 1, "case_type": "deterministic rank-1 illustrative IAPV structural case",
        "selected": str(args.selected), "pool": str(args.pool), "seed_panel": str(args.seed_panel),
        "parent": args.parent, "run_seed": int(args.run_seed), "anchor_probability_threshold": 0.5,
        "chosen": {method: {"candidate_id": row["candidate_id"], "selection_rank": int(row["selection_rank"]),
                               "ireslm_probability_mean": float(row["ireslm_probability_mean"]),
                               "mfe_abs_delta_kcal_mol": float(row["mfe_abs_delta_kcal_mol"]),
                               "parent_ensemble_anchor_retention": float(row["parent_ensemble_anchor_retention"])}
                   for method, row in chosen.items()},
        "scope": "illustrative computational structure case only; not validation of candidate activity",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
