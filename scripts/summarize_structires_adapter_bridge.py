#!/usr/bin/env python3
"""Summarize the Panel-A-to-Panel-B StructIRES-Adapter bridge experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "structires_adapter_probability_mean",
    "mfe_abs_delta_kcal_mol",
    "pairing_profile_l1",
    "mfe_structure_state_identity",
    "parent_ensemble_anchor_retention",
    "direct_rna_s3_heldout_proxy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--proxy-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    selected = pd.read_csv(args.selected)
    proxy = pd.read_csv(args.proxy_scores)
    keys = ["method", "run_seed", "parent_id", "candidate_id"]
    if proxy.duplicated(keys).any() or selected.duplicated(keys).any():
        raise ValueError("selected or proxy rows are not unique")
    frame = selected.merge(proxy, on=keys, how="inner", validate="one_to_one")
    if len(frame) != len(selected):
        raise ValueError("proxy scores do not cover all selected candidates")
    required = {"adapter_score_only", "structires_adapter_rank"}
    if not required <= set(frame.method):
        raise ValueError("bridge comparison methods are missing")

    unit = frame.groupby(["method", "parent_id", "run_seed"], as_index=False)[
        list(METRICS)
    ].mean()
    method_summary = unit.groupby("method")[list(METRICS)].agg(["mean", "std"])
    method_summary.columns = [f"{metric}_{stat}" for metric, stat in method_summary.columns]
    method_summary = method_summary.reset_index()

    indexed = unit.set_index(["method", "parent_id", "run_seed"])
    full = indexed.xs("structires_adapter_rank", level="method")
    score = indexed.xs("adapter_score_only", level="method")
    delta = full[list(METRICS)] - score[list(METRICS)]
    parent_delta = delta.groupby("parent_id").mean()
    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = []
    for metric in METRICS:
        values = parent_delta[metric].to_numpy(dtype=np.float64)
        boot = rng.choice(
            values,
            size=(args.bootstrap_replicates, len(values)),
            replace=True,
        ).mean(axis=1)
        summaries.append(
            {
                "metric": metric,
                "n_parent_units": len(values),
                "mean_structires_adapter_rank_minus_adapter_score_only": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "parent_bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                "parent_bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                "n_parents_positive": int((values > 0).sum()),
                "n_parents_negative": int((values < 0).sum()),
                "n_parents_zero": int((values == 0).sum()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    method_path = args.output_dir / "method_summary.csv"
    paired_path = args.output_dir / "paired_parent_summary.csv"
    method_summary.to_csv(method_path, index=False)
    pd.DataFrame(summaries).to_csv(paired_path, index=False)
    manifest = {
        "schema_version": 1,
        "experiment": "panel_a_to_panel_b_structires_adapter_bridge_summary",
        "comparison": "structires_adapter_rank minus adapter_score_only",
        "selected": {"path": str(args.selected.resolve()), "sha256": sha256(args.selected)},
        "proxy_scores": {
            "path": str(args.proxy_scores.resolve()),
            "sha256": sha256(args.proxy_scores),
        },
        "n_selected_rows": len(frame),
        "n_methods": int(frame.method.nunique()),
        "n_parent_pool_units": len(unit),
        "n_independent_parent_units_for_ci": int(parent_delta.shape[0]),
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "scope": "computational proposal-score and structure-preservation selection; direct-RNA value is a held-out computational proxy, not candidate activity measurement",
        "outputs": [
            {"path": str(method_path.resolve()), "sha256": sha256(method_path)},
            {"path": str(paired_path.resolve()), "sha256": sha256(paired_path)},
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
