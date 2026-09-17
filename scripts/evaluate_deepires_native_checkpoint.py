#!/usr/bin/env python3
"""Re-evaluate a frozen DeepIRES checkpoint on the canonical inner validation split."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=174)
    parser.add_argument("--crop", choices=("left", "right"), default="left")
    parser.add_argument("--padding", choices=("pre", "post"), default="post")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--unmasked-fixed-length", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_module():
    source = Path(__file__).with_name("train_deepires_native.py")
    spec = importlib.util.spec_from_file_location("deepires_checkpoint_eval", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    training = load_training_module()
    utility = training.utility_module()
    frame = utility.load_fold_rows(args.dataset, args.fold)
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    labels = frame.label.to_numpy(dtype=np.int64)
    upstream_train = np.flatnonzero(frame.type.to_numpy() == "train")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.validation_fraction, random_state=args.seed
    )
    _, validation_local = next(splitter.split(upstream_train, labels[upstream_train]))
    validation = upstream_train[validation_local]

    tokens, lengths = training.encode_sequences(
        frame.Sequence.astype(str).tolist(), args.max_length, args.crop, args.padding
    )
    label_tensor = torch.from_numpy(labels.astype(np.float32))
    loader = training.make_loader(
        tokens,
        lengths,
        label_tensor,
        validation,
        args.batch_size,
        False,
        args.seed,
        0,
    )
    device = torch.device(args.device)
    model = training.DeepIRES(args.dropout, args.unmasked_fixed_length).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if int(payload.get("fold", args.fold)) != args.fold:
        raise ValueError("checkpoint fold does not match requested fold")
    model.load_state_dict(payload["model"], strict=True)
    probability = training.probability(model, loader, device)
    if len(probability) != len(validation):
        raise RuntimeError("prediction count does not match validation split")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "validation_predictions.csv.gz"
    with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "label", "probability"))
        writer.writeheader()
        for index, score in zip(validation, probability):
            writer.writerow(
                {
                    "sample_index": int(index),
                    "label": int(labels[index]),
                    "probability": float(score),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment": "frozen_deepires_checkpoint_canonical_validation_evaluation",
        "fold": args.fold,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "n_validation": int(len(validation)),
        "n_positive": int(labels[validation].sum()),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "preprocessing": {
            "max_length": args.max_length,
            "crop": args.crop,
            "padding": args.padding,
            "unmasked_fixed_length": args.unmasked_fixed_length,
        },
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "training_performed": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
