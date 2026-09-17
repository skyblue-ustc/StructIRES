#!/usr/bin/env python3
"""Compare the public RNAFM predictor with frozen released predictions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--batch-toks", type=int, default=4096)
    parser.add_argument(
        "--token-dropout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the released-checkpoint evaluator's inferred RNA-FM setting.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.predictor_dir))
    import RNAFM_Predictor as predictor

    with zipfile.ZipFile(args.dataset_zip) as archive:
        member = next(
            name
            for name in archive.namelist()
            if name.endswith("v2_dataset_with_unified_stratified_shuffle_train_test_split.csv")
            and not name.startswith("__MACOSX")
        )
        data = pd.read_csv(archive.open(member))
    test = data[(data["fold"] == args.fold) & (data["type"] == "test")]
    test = test.sort_values("idx", kind="stable").iloc[: args.n]
    sequences = [value.replace("T", "U") for value in test["Sequence"].astype(str)]
    labels = test["IRES_class_600"].astype(int).tolist()

    with gzip.open(args.reference_predictions, "rt", encoding="utf-8") as handle:
        reference = pd.read_csv(handle)
    by_hash = dict(zip(reference["sequence_sha256"], reference["probability"]))
    reference_scores = np.asarray(
        [by_hash[hashlib.sha256(value.encode("ascii")).hexdigest()] for value in sequences]
    )

    model = predictor.RNAFM_linear().to(predictor.device)
    # The public backbone checkpoint omits this runtime option, whereas the
    # released classifier evaluator infers it as true from the training model.
    # It scales embeddings even when no tokens are masked, so it must match.
    model.rnafm.args.token_dropout = args.token_dropout
    # This legacy implementation is batch-sensitive.  Use the token budget of
    # the frozen released-checkpoint evaluation rather than its module default.
    predictor.batch_toks = args.batch_toks
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint incompatibility: missing={incompatible.missing_keys}; "
            f"unexpected={incompatible.unexpected_keys}"
        )

    _, dataloader = predictor.generate_dataset_dataloader(labels, sequences)
    predictions = predictor.predict_step(dataloader, model, args.fold)["df"]
    # FastaBatchedDataset groups records by length, so returned rows need to
    # be mapped back to the canonical input order before comparison.
    predicted_by_hash = {
        hashlib.sha256(sequence.encode("ascii")).hexdigest(): score
        for sequence, score in zip(predictions["sequence"], predictions[f"Prob_R{args.fold}"])
    }
    scores = np.asarray(
        [predicted_by_hash[hashlib.sha256(value.encode("ascii")).hexdigest()] for value in sequences]
    )
    delta = np.abs(scores - reference_scores)
    result = {
        "schema_version": 1,
        "purpose": "public_RNAFM_Predictor_checkpoint_compatibility",
        "fold": args.fold,
        "n_sequences": int(len(scores)),
        "device": str(predictor.device),
        "token_dropout": args.token_dropout,
        "batch_toks": args.batch_toks,
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
        "allclose_atol_1e-6": bool(np.allclose(scores, reference_scores, rtol=0, atol=1e-6)),
        "checkpoint": str(args.checkpoint),
        "reference_predictions": str(args.reference_predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
