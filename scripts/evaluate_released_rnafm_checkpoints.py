#!/usr/bin/env python3
"""Rerun released IRES-RNAFM checkpoints without fetching the base model again."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import re
import sys
import types
import zipfile
from argparse import Namespace
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--upstream-script-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--batch-toks", type=int, default=1024)
    parser.add_argument("--truncate-num", type=int, default=1024)
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


def load_fm(script_dir: Path):
    package_dir = script_dir / "fm"
    package = types.ModuleType("fm")
    package.__package__ = "fm"
    package.__path__ = [str(package_dir)]
    sys.modules["fm"] = package
    data_module = importlib.import_module("fm.data")
    model_module = importlib.import_module("fm.model")
    # The upstream fm/__init__.py imports optional downstream tooling and fails
    # when the unused ptflops package is absent. Frozen classification needs
    # only these three public classes, so load their modules directly.
    return types.SimpleNamespace(
        Alphabet=data_module.Alphabet,
        FastaBatchedDataset=data_module.FastaBatchedDataset,
        RNABertModel=model_module.RNABertModel,
    )


def clean_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {key.removeprefix("module."): value for key, value in state.items()}


def infer_backbone_args(
    state: dict[str, torch.Tensor], alphabet: object
) -> tuple[Namespace, dict[str, int | bool | str]]:
    prefix = "rnafm."
    embed_dim = int(state[prefix + "embed_tokens.weight"].shape[1])
    ffn_embed_dim = int(state[prefix + "layers.0.fc1.weight"].shape[0])
    layer_ids = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"rnafm\.layers\.(\d+)\.", key)) is not None
    }
    layers = max(layer_ids) + 1
    contact_width = int(state[prefix + "contact_head.regression.weight"].shape[1])
    if contact_width % layers:
        raise ValueError("cannot infer RNA-FM attention-head count from checkpoint")
    attention_heads = contact_width // layers
    position_rows = int(state[prefix + "embed_positions.weight"].shape[0])
    max_positions = position_rows - int(alphabet.padding_idx) - 1
    values: dict[str, int | bool | str] = {
        "arch": "roberta_large",
        "layers": layers,
        "embed_dim": embed_dim,
        "ffn_embed_dim": ffn_embed_dim,
        "attention_heads": attention_heads,
        "max_positions": max_positions,
        "token_dropout": True,
    }
    return Namespace(**values), values


class RNAFMClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.rnafm = backbone
        self.dropout3 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(embed_dim, 40)
        self.output = nn.Linear(40, 2)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.rnafm(tokens, [12])["representations"][12][:, 0]
        logits = self.output(self.dropout3(self.relu(self.fc(representation))))
        probabilities = torch.softmax(logits, dim=1)
        return probabilities[:, 1], torch.argmax(probabilities, dim=1)


def predict(
    model: nn.Module,
    dataloader: object,
    device: torch.device,
    truncate_num: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    sequences: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch_labels, batch_sequences, _, tokens, _, _ in dataloader:
            tokens = tokens[:, :truncate_num].to(device)
            batch_probabilities, batch_predictions = model(tokens)
            sequences.extend(batch_sequences)
            labels.extend(int(label) for label in batch_labels)
            probabilities.extend(batch_probabilities.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
    return (
        sequences,
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
        np.asarray(predictions, dtype=int),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Return fixed-width expected calibration error on the positive class."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if selected.any():
            value += float(selected.mean()) * abs(
                float(labels[selected].mean()) - float(probabilities[selected].mean())
            )
    return value


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    negatives = labels == 0
    specificity = float((predictions[negatives] == 0).mean()) if negatives.any() else float("nan")
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "ece10": expected_calibration_error(labels, probabilities),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    data = load_dataset(args.dataset_zip)
    fm = load_fm(args.upstream_script_dir)
    alphabet = fm.Alphabet.from_architecture("roberta_large", theme="rna")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    checkpoint_audits: list[dict[str, object]] = []
    architecture: dict[str, int | bool | str] | None = None
    for fold in folds:
        test = data[(data["fold"] == fold) & (data["type"] == "test")].copy()
        test = test.sort_values("idx", kind="stable")
        labels = test["IRES_class_600"].astype(int).to_numpy()
        sequences = [sequence.replace("T", "U") for sequence in test["Sequence"].astype(str)]
        checkpoint = args.checkpoint_dir / f"IRES_RNAFM_best_model_fold{fold}.pt"
        state = clean_checkpoint(checkpoint)
        backbone_args, inferred = infer_backbone_args(state, alphabet)
        if architecture is not None and architecture != inferred:
            raise ValueError(f"RNA-FM architecture differs at fold {fold}")
        architecture = inferred
        model = RNAFMClassifier(
            fm.RNABertModel(backbone_args, alphabet), int(inferred["embed_dim"])
        ).to(device)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                f"checkpoint/model mismatch fold {fold}: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        dataset = fm.FastaBatchedDataset(labels.tolist(), sequences, mask_prob=0)
        batches = dataset.get_batch_indices(
            toks_per_batch=args.batch_toks, extra_toks_per_seq=1
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(),
            batch_sampler=batches,
            shuffle=False,
        )
        result_sequences, result_labels, probabilities, predictions = predict(
            model, dataloader, device, args.truncate_num
        )
        if sorted(zip(sequences, labels.tolist())) != sorted(
            zip(result_sequences, result_labels.tolist())
        ):
            raise ValueError(f"prediction rows do not match input examples for fold {fold}")

        metric_rows.append(
            {
                "model": "IRES-RNAFM",
                "status": "released_checkpoint_rerun",
                "protocol": "released_native_test_fold",
                "fold": fold,
                "n": len(result_labels),
                "n_positive": int(result_labels.sum()),
                **classification_metrics(result_labels, probabilities, predictions),
            }
        )
        row_ids: defaultdict[tuple[str, int], deque[int]] = defaultdict(deque)
        for idx, sequence, label in zip(test["idx"], sequences, labels):
            row_ids[(sequence, int(label))].append(int(idx))
        for sequence, label, probability, prediction in zip(
            result_sequences, result_labels, probabilities, predictions
        ):
            prediction_rows.append(
                {
                    "fold": fold,
                    "idx": row_ids[(sequence, int(label))].popleft(),
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    "label": int(label),
                    "probability": float(probability),
                    "prediction": int(prediction),
                }
            )
        checkpoint_audits.append(
            {"fold": fold, "path": str(checkpoint), "sha256": sha256(checkpoint)}
        )
        print(json.dumps(metric_rows[-1], sort_keys=True), flush=True)
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "metrics.csv", metric_rows)
    with gzip.open(
        args.output_dir / "predictions.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "released_ires_rnafm_native_checkpoint_rerun",
        "scope": "classification-only reproduction",
        "dataset_zip": str(args.dataset_zip),
        "dataset_zip_sha256": sha256(args.dataset_zip),
        "upstream_fm_dir": str(args.upstream_script_dir / "fm"),
        "upstream_fm_init_bypassed": (
            "Optional downstream imports were bypassed; fm.data and fm.model were loaded directly."
        ),
        "folds": folds,
        "batch_toks": args.batch_toks,
        "truncate_num": args.truncate_num,
        "device": str(device),
        "torch_version": torch.__version__,
        "architecture_inferred_from_checkpoint": architecture,
        "base_checkpoint_needed_for_frozen_inference": False,
        "checkpoints": checkpoint_audits,
        "test_data_used_for_training_or_threshold_selection_in_this_rerun": False,
        "upstream_training_protocol_warning": (
            "Released checkpoints were originally selected on test AUPR; this script only reruns "
            "their frozen predictions."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
