#!/usr/bin/env python3
"""Score frozen mutation pools with a validated released IRES-RNAFM checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, nargs="+", required=True)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-toks", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.predictor_dir))
    import RNAFM_Predictor as predictor

    candidates: list[dict[str, object]] = []
    pool_audits: list[dict[str, object]] = []
    for pool in args.pool:
        path = pool / "candidates.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if not rows:
            raise ValueError(f"empty candidate pool: {path}")
        candidates.extend({"pool_dir": str(pool), **row} for row in rows)
        pool_audits.append({"path": str(path), "sha256": sha256(path), "n": len(rows)})
    ids = [str(row["candidate_id"]) for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique across pools")

    predictor.batch_toks = args.batch_toks
    predictor.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = ["candidate"] * len(candidates)
    sequences = [str(row["sequence"]).upper().replace("T", "U") for row in candidates]
    _, dataloader = predictor.generate_dataset_dataloader(labels, sequences)
    model = predictor.RNAFM_linear().to(predictor.device)
    # Required by the full-fold compatibility audit against frozen released outputs.
    model.rnafm.args.token_dropout = True
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}; "
            f"unexpected={incompatible.unexpected_keys}"
        )
    result = predictor.predict_step(dataloader, model, args.fold)["df"]
    queues: defaultdict[str, deque[str]] = defaultdict(deque)
    for candidate_id, sequence in zip(ids, sequences):
        queues[sequence].append(candidate_id)
    scores: dict[str, float] = {}
    for sequence, probability in zip(result["sequence"].astype(str), result[f"Prob_R{args.fold}"]):
        scores[queues[sequence].popleft()] = float(probability)
    if any(queue for queue in queues.values()):
        raise ValueError("prediction alignment incomplete")

    rows = [
        {
            "candidate_id": row["candidate_id"],
            "parent_id": row["parent_id"],
            "run_seed": row.get("run_seed", row["seed"]),
            "pool_dir": row["pool_dir"],
            "sequence_sha256": hashlib.sha256(str(row["sequence"]).encode()).hexdigest(),
            "rnafm_probability_fold0": scores[str(row["candidate_id"])],
        }
        for row in candidates
    ]
    with (args.output_dir / "rnafm_pool_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "experiment": "released_rnafm_fold0_score_on_frozen_shared_pools",
        "scope": "released-classifier score-only baseline; no direct-RNA labels used for fitting or selection",
        "pools": pool_audits,
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
        "fold": args.fold,
        "batch_toks": args.batch_toks,
        "token_dropout": True,
        "predictor": str(args.predictor_dir / "RNAFM_Predictor.py"),
        "predictor_sha256": sha256(args.predictor_dir / "RNAFM_Predictor.py"),
        "device": str(predictor.device),
        "n_candidates": len(rows),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_candidates": len(rows), "mean_probability": sum(scores.values()) / len(scores)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
