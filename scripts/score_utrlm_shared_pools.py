#!/usr/bin/env python3
"""Score frozen mutation pools with the released IRES-UTRLM checkpoint ensemble.

This is a classification-score baseline only.  It does not use direct-RNA labels,
does not retrain a model, and does not alter the mutation pools.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictor(script_dir: Path):
    sys.path.insert(0, str(script_dir))
    module_path = script_dir / "UTRLM_Predictor.py"
    spec = importlib.util.spec_from_file_location("ires_ea_public_utrlm_predictor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, nargs="+", required=True,
                        help="Pool directories containing candidates.jsonl")
    parser.add_argument("--upstream-script-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--batch-toks", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidates: list[dict[str, object]] = []
    pool_audits=[]
    for pool in args.pool:
        path=pool / "candidates.jsonl"
        with path.open() as handle:
            rows=[json.loads(line) for line in handle]
        if not rows:
            raise ValueError(f"empty pool: {path}")
        for row in rows:
            row={"pool_dir":str(pool), **row}
            row["model_sequence"]=str(row["sequence"]).upper().replace("U", "T")
            candidates.append(row)
        pool_audits.append({"path":str(path),"sha256":sha256(path),"n":len(rows)})
    ids=[str(row["candidate_id"]) for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id must be unique across input pools")

    predictor=load_predictor(args.upstream_script_dir)
    predictor.batch_toks=args.batch_toks
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.device=device
    labels=["candidate"] * len(candidates)
    sequences=[str(row["model_sequence"]) for row in candidates]
    _, loader=predictor.generate_dataset_dataloader(labels, sequences)
    folds=[int(value) for value in args.folds.split(",") if value.strip()]
    all_prob={candidate_id:[] for candidate_id in ids}
    checkpoint_audits=[]
    for fold in folds:
        checkpoint=args.checkpoint_dir / f"IRES_UTRLM_best_model_fold{fold}.pt"
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        clean=OrderedDict((key.replace("module.",""),value) for key,value in state.items())
        model=predictor.CNN_linear().to(device)
        incompatible=model.load_state_dict(clean,strict=False)
        allowed={"esm2.supervised_linear.weight","esm2.supervised_linear.bias"}
        if incompatible.missing_keys or set(incompatible.unexpected_keys) != allowed:
            raise ValueError(f"checkpoint mismatch fold {fold}: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
        result=predictor.predict_step(loader,model,fold)["df"]
        queues: defaultdict[str, deque[str]]=defaultdict(deque)
        for candidate_id, sequence in zip(ids,sequences):
            queues[sequence].append(candidate_id)
        probs=result[f"Prob_U{fold}"].tolist()
        for sequence,probability in zip(result["sequence"].astype(str),probs):
            candidate_id=queues[sequence].popleft()
            all_prob[candidate_id].append(float(probability))
        if any(queue for queue in queues.values()):
            raise ValueError(f"prediction alignment incomplete for fold {fold}")
        checkpoint_audits.append({"fold":fold,"path":str(checkpoint),"sha256":sha256(checkpoint),"ignored_checkpoint_keys":sorted(allowed)})
        print(json.dumps({"fold":fold,"n":len(probs),"mean_probability":sum(probs)/len(probs)}),flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output=[]
    for row in candidates:
        values=all_prob[str(row["candidate_id"])]
        if len(values) != len(folds):
            raise ValueError("incomplete fold prediction")
        output.append({"candidate_id":row["candidate_id"],"parent_id":row["parent_id"],"run_seed":row.get("run_seed", row["seed"]),
                       "pool_dir":row["pool_dir"],"sequence_sha256":hashlib.sha256(str(row["sequence"]).encode()).hexdigest(),
                       "utrlm_probability_mean":sum(values)/len(values),
                       "utrlm_probability_std":float(torch.tensor(values).std(unbiased=True).item()),
                       "n_folds":len(values)})
    with (args.output_dir / "utrlm_pool_scores.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    manifest={"schema_version":1,"experiment":"released_utrlm_ensemble_score_on_frozen_shared_pools","scope":"classification-score baseline; no direct-RNA labels used for fitting or selection","pools":pool_audits,
              "upstream_predictor":str(args.upstream_script_dir / "UTRLM_Predictor.py"),"upstream_predictor_sha256":sha256(args.upstream_script_dir / "UTRLM_Predictor.py"),"folds":folds,"checkpoints":checkpoint_audits,"device":str(device),"torch_version":torch.__version__,"n_candidates":len(output)}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
