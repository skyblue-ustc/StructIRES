#!/usr/bin/env python3
"""Rerun released IRES-UTRLM checkpoints on their native test folds."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import zipfile
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--upstream-script-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--batch-toks", type=int, default=4096)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.endswith("v2_dataset_with_unified_stratified_shuffle_train_test_split.csv")
            and not name.startswith("__MACOSX")
        ]
        if len(members) != 1:
            raise ValueError(f"expected one canonical CSV member, found {members}")
        with archive.open(members[0]) as handle:
            return pd.read_csv(
                handle,
                usecols=["fold", "type", "idx", "Sequence", "IRES_class_600"],
            )


def load_predictor(script_dir: Path):
    sys.path.insert(0, str(script_dir))
    module_path = script_dir / "UTRLM_Predictor.py"
    spec = importlib.util.spec_from_file_location("released_utrlm_predictor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    data = load_dataset(args.dataset_zip)
    predictor = load_predictor(args.upstream_script_dir)
    predictor.batch_toks = args.batch_toks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.device = device

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    checkpoint_audits: list[dict[str, object]] = []
    for fold in folds:
        test = data[(data["fold"] == fold) & (data["type"] == "test")].copy()
        test = test.sort_values("idx", kind="stable")
        labels = test["IRES_class_600"].astype(int).to_numpy()
        sequences = test["Sequence"].astype(str).tolist()
        checkpoint = args.checkpoint_dir / f"IRES_UTRLM_best_model_fold{fold}.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        clean_state = OrderedDict((key.replace("module.", ""), value) for key, value in state.items())
        model = predictor.CNN_linear().to(device)
        incompatible = model.load_state_dict(clean_state, strict=False)
        allowed_unexpected = {
            "esm2.supervised_linear.weight",
            "esm2.supervised_linear.bias",
        }
        unexpected = set(incompatible.unexpected_keys)
        if incompatible.missing_keys or unexpected != allowed_unexpected:
            raise ValueError(
                f"checkpoint/model mismatch fold {fold}: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        _, dataloader = predictor.generate_dataset_dataloader(labels.tolist(), sequences)
        result = predictor.predict_step(dataloader, model, fold)["df"]
        # FastaBatchedDataset groups examples by token count. Therefore prediction
        # rows follow batch-sampler order, not the input DataFrame order.
        result_labels = result["category"].to_numpy(dtype=int)
        result_sequences = result["sequence"].astype(str).tolist()
        probabilities = result[f"Prob_U{fold}"].to_numpy(dtype=float)
        predictions = result[f"Pred_U{fold}"].to_numpy(dtype=int)
        if sorted(zip(sequences, labels.tolist())) != sorted(
            zip(result_sequences, result_labels.tolist())
        ):
            raise ValueError(f"prediction rows do not match input examples for fold {fold}")
        metric_rows.append(
            {
                "model": "IRES-UTRLM",
                "status": "released_checkpoint_rerun",
                "protocol": "released_native_test_fold",
                "fold": fold,
                "n": len(result_labels),
                "n_positive": int(result_labels.sum()),
                "auc": float(roc_auc_score(result_labels, probabilities)),
                "aupr": float(average_precision_score(result_labels, probabilities)),
                "f1": float(f1_score(result_labels, predictions, zero_division=0)),
                "accuracy": float(accuracy_score(result_labels, predictions)),
            }
        )
        row_ids: defaultdict[tuple[str, int], deque[int]] = defaultdict(deque)
        for idx, sequence, label in zip(test["idx"], sequences, labels):
            row_ids[(sequence, int(label))].append(int(idx))
        for sequence, label, probability, prediction in zip(
            result_sequences, result_labels, probabilities, predictions
        ):
            idx = row_ids[(sequence, int(label))].popleft()
            prediction_rows.append(
                {
                    "fold": fold,
                    "idx": int(idx),
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    "label": int(label),
                    "probability": float(probability),
                    "prediction": int(prediction),
                }
            )
        checkpoint_audits.append(
            {
                "fold": fold,
                "path": str(checkpoint),
                "sha256": sha256(checkpoint),
                "ignored_checkpoint_keys": sorted(unexpected),
                "compatibility_reason": (
                    "The released checkpoint was trained with esm2_supervised. Its auxiliary "
                    "supervised_linear head is absent from the released predictor and is not "
                    "called by CNN_linear.forward; all prediction-path parameters load exactly."
                ),
            }
        )
        print(json.dumps(metric_rows[-1], sort_keys=True), flush=True)

    write_csv(args.output_dir / "metrics.csv", metric_rows)
    with gzip.open(args.output_dir / "predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "released_ires_utrlm_native_checkpoint_rerun",
        "scope": "classification-only reproduction",
        "dataset_zip": str(args.dataset_zip),
        "dataset_zip_sha256": sha256(args.dataset_zip),
        "upstream_predictor": str(args.upstream_script_dir / "UTRLM_Predictor.py"),
        "upstream_predictor_sha256": sha256(args.upstream_script_dir / "UTRLM_Predictor.py"),
        "folds": folds,
        "batch_toks": args.batch_toks,
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoints": checkpoint_audits,
        "test_data_used_for_training_or_threshold_selection_in_this_rerun": False,
        "upstream_training_protocol_warning": "Released checkpoints were originally selected on test AUPR; this script only reruns their frozen predictions.",
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
