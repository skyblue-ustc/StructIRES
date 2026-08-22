#!/usr/bin/env python3
"""Select a budget-matched score-only arm using frozen UTR-LM ensemble scores."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--scores",type=Path,required=True)
    parser.add_argument("--likelihood",type=Path,required=True)
    parser.add_argument("--structure",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--top-k",type=int,default=50)
    args=parser.parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    with args.scores.open(newline="") as handle:
        scores={row["candidate_id"]:row for row in csv.DictReader(handle)}
    with args.likelihood.open(newline="") as handle:
        likelihood={row["candidate_id"]:row for row in csv.DictReader(handle)}
    with args.structure.open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        if row["candidate_id"] not in scores:
            raise ValueError(f"missing UTR-LM score for {row['candidate_id']}")
        row.update(scores[row["candidate_id"]])
        if row["candidate_id"] not in likelihood:
            raise ValueError(f"missing RNA-LM likelihood for {row['candidate_id']}")
        row.update(likelihood[row["candidate_id"]])
    grouped: dict[tuple[str,str],list[dict[str,str]]]=defaultdict(list)
    for row in rows:
        grouped[(row["run_seed"],row["parent_id"])].append(row)
    selected=[]
    for (run_seed,parent_id), group in sorted(grouped.items()):
        if len(group) < args.top_k:
            raise ValueError(f"{parent_id} has fewer than {args.top_k} candidates")
        for rank,row in enumerate(sorted(group,key=lambda x:float(x["utrlm_probability_mean"]),reverse=True)[:args.top_k],start=1):
            selected.append({"method":"utrlm_score_only","selection_rank":rank,**row})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    print({"n_selected":len(selected),"n_units":len(grouped),"method":"utrlm_score_only"})
    return 0


if __name__=="__main__":
    raise SystemExit(main())
