#!/usr/bin/env python3
"""Stress-test selected IRES candidates across a frozen cargo panel.

The panel must be an explicit JSON manifest with ``records`` containing
``cargo_id`` and canonical-RNA ``sequence``.  This script intentionally does
not infer or invent cargo sequences.  It reports ViennaRNA ensemble crosstalk
and context-consistency metrics only, never translation activity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ires_design.structure import (
    context_structure_consistency,
    fold_ensemble,
    ires_crosstalk_ratio,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidates(pools: list[Path]) -> dict[str, dict[str, object]]:
    candidates: dict[str, dict[str, object]] = {}
    for pool in pools:
        path = pool / "candidates.jsonl"
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                candidate_id = str(row["candidate_id"])
                if candidate_id in candidates:
                    raise ValueError(f"duplicate candidate_id across pools: {candidate_id}")
                candidates[candidate_id] = row
    return candidates


def worker(payload: tuple[dict[str, str], dict[str, object], list[dict[str, str]]]) -> list[dict[str, object]]:
    selected, candidate, cargoes = payload
    sequence = str(candidate["sequence"])
    reference = fold_ensemble(sequence)
    rows=[]
    for cargo in cargoes:
        context = fold_ensemble(sequence + cargo["sequence"])
        rows.append({
            "method": selected["method"],
            "selection_rank": int(selected["selection_rank"]),
            "run_seed": int(selected["run_seed"]),
            "parent_id": selected["parent_id"],
            "candidate_id": selected["candidate_id"],
            "candidate_sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "cargo_id": cargo["cargo_id"],
            "cargo_length": len(cargo["sequence"]),
            "ires_length": len(sequence),
            "ires_crosstalk_ratio": ires_crosstalk_ratio(context, len(sequence)),
            "context_structure_consistency": context_structure_consistency(reference, context),
            "viennarna_version": context.viennarna_version,
        })
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def bootstrap_paired(group_rows: list[dict[str, object]], left: str, right: str) -> list[dict[str, object]]:
    by_key = {(str(row["run_seed"]), str(row["parent_id"]), str(row["method"])): row for row in group_rows}
    pairs=[]
    for run_seed, parent_id in sorted({key[:2] for key in by_key}):
        a=by_key.get((run_seed,parent_id,left)); b=by_key.get((run_seed,parent_id,right))
        if a is not None and b is not None:
            pairs.append((a,b))
    if not pairs:
        return []
    rng=np.random.default_rng(20260822)
    output=[]
    for metric in ("mean_crosstalk_ratio", "worst_crosstalk_ratio", "mean_context_consistency", "worst_context_consistency"):
        values=np.asarray([float(a[metric])-float(b[metric]) for a,b in pairs])
        boot=np.mean(rng.choice(values,size=(10000,len(values)),replace=True),axis=1)
        output.append({"left_method":left,"right_method":right,"metric":metric,"n_parent_run_units":len(values),
                       "mean_left_minus_right":float(values.mean()),"bootstrap_ci95_low":float(np.quantile(boot,.025)),
                       "bootstrap_ci95_high":float(np.quantile(boot,.975))})
    return output


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--pool",type=Path,nargs="+",required=True)
    parser.add_argument("--selected",type=Path,nargs="+",required=True)
    parser.add_argument("--cargo-panel",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--top-k",type=int,default=10)
    parser.add_argument("--workers",type=int,default=16)
    args=parser.parse_args()
    if args.top_k < 1 or args.workers < 1:
        raise ValueError("top-k and workers must be positive")
    panel=json.loads(args.cargo_panel.read_text())
    cargoes=[{"cargo_id":str(row["cargo_id"]),"sequence":str(row["sequence"]).upper().replace("T","U")} for row in panel["records"]]
    if not cargoes or any(not row["sequence"] or set(row["sequence"])-set("ACGU") for row in cargoes):
        raise ValueError("cargo panel must contain non-empty canonical RNA sequences")
    if len({row["cargo_id"] for row in cargoes}) != len(cargoes):
        raise ValueError("cargo_id must be unique")
    candidates=load_candidates(args.pool)
    selected=[]
    for path in args.selected:
        with path.open(newline="") as handle:
            selected.extend(row for row in csv.DictReader(handle) if int(row["selection_rank"]) <= args.top_k)
    if not selected:
        raise ValueError("no selected candidates under top-k")
    missing={str(row["candidate_id"]) for row in selected}-set(candidates)
    if missing:
        raise ValueError(f"selected ids missing from frozen pools: {sorted(missing)[:5]}")
    payloads=[(row,candidates[str(row["candidate_id"])],cargoes) for row in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows=[result for batch in executor.map(worker,payloads,chunksize=1) for result in batch]
    grouped: dict[tuple[str,str,str],list[dict[str,object]]]=defaultdict(list)
    for row in rows:
        grouped[(str(row["run_seed"]),str(row["parent_id"]),str(row["method"]))].append(row)
    group_rows=[]
    for (run_seed,parent_id,method),values in sorted(grouped.items()):
        by_cargo: dict[str,list[dict[str,object]]]=defaultdict(list)
        for row in values: by_cargo[str(row["cargo_id"])].append(row)
        cargo_cross=[mean([float(row["ires_crosstalk_ratio"]) for row in group]) for group in by_cargo.values()]
        cargo_consistency=[mean([float(row["context_structure_consistency"]) for row in group]) for group in by_cargo.values()]
        group_rows.append({"run_seed":int(run_seed),"parent_id":parent_id,"method":method,"n_candidate_cargo_records":len(values),
                           "mean_crosstalk_ratio":mean(cargo_cross),"worst_crosstalk_ratio":max(cargo_cross),
                           "mean_context_consistency":mean(cargo_consistency),"worst_context_consistency":min(cargo_consistency)})
    methods=sorted({str(row["method"]) for row in group_rows})
    summary=[]
    for method in methods:
        subset=[row for row in group_rows if row["method"] == method]
        summary.append({"method":method,"n_parent_run_units":len(subset),
                        **{metric:mean([float(row[metric]) for row in subset]) for metric in ("mean_crosstalk_ratio","worst_crosstalk_ratio","mean_context_consistency","worst_context_consistency")}})
    comparisons=[]
    for right in ("score_only","rnafm_fold0_score_only","utrlm_score_only","random_mutation"):
        if "robust_full" in methods and right in methods:
            comparisons.extend(bootstrap_paired(group_rows,"robust_full",right))
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,data in (("candidate_cargo_metrics.csv",rows),("parent_run_summary.csv",group_rows),("method_summary.csv",summary),("paired_comparisons.csv",comparisons)):
        if not data:
            continue
        with (args.output_dir/name).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    manifest={"schema_version":1,"experiment":"cargo_context_stress_test","scope":"ViennaRNA ensemble crosstalk and context consistency; not experimental activity","top_k_per_parent_method":args.top_k,"workers":args.workers,
              "cargo_panel":{"path":str(args.cargo_panel),"sha256":sha256(args.cargo_panel),"n_cargoes":len(cargoes)},
              "pools":[{"path":str(path/"candidates.jsonl"),"sha256":sha256(path/"candidates.jsonl")} for path in args.pool],
              "selected":[{"path":str(path),"sha256":sha256(path)} for path in args.selected],"n_candidate_cargo_records":len(rows)}
    (args.output_dir/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(manifest,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
