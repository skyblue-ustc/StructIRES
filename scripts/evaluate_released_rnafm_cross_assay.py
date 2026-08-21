#!/usr/bin/env python3
"""Evaluate frozen released IRES-RNAFM checkpoints on direct-RNA labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from evaluate_released_rnafm_checkpoints import (
    RNAFMClassifier,
    clean_checkpoint,
    infer_backbone_args,
    load_fm,
    predict,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-data", type=Path, required=True)
    parser.add_argument("--overlap-data", type=Path, required=True)
    parser.add_argument("--upstream-script-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--batch-toks", type=int, default=4096)
    parser.add_argument("--truncate-num", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def canonicalize(sequence: str) -> str:
    return "".join(sequence.split()).upper().replace("T", "U")


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    error = 0.0
    for index in range(10):
        upper = probabilities <= edges[index + 1] if index == 9 else probabilities < edges[index + 1]
        mask = (probabilities >= edges[index]) & upper
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(error)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predicted = probabilities >= 0.5
    negatives = labels == 0
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float((~predicted[negatives]).mean()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece10": expected_calibration_error(labels, probabilities),
        "decision_threshold": 0.5,
    }


def bootstrap(
    labels: np.ndarray, probabilities: np.ndarray, replicates: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    aucs = np.empty(replicates)
    auprs = np.empty(replicates)
    for index in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        aucs[index] = roc_auc_score(labels[sampled], probabilities[sampled])
        auprs[index] = average_precision_score(labels[sampled], probabilities[sampled])
    return {
        "auc_ci_low": float(np.quantile(aucs, 0.025)),
        "auc_ci_high": float(np.quantile(aucs, 0.975)),
        "aupr_ci_low": float(np.quantile(auprs, 0.025)),
        "aupr_ci_high": float(np.quantile(auprs, 0.975)),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]

    direct = pd.read_csv(args.direct_data)
    direct = direct[direct["consensus_status"].isin(["active", "inactive"])].copy()
    direct["label"] = direct["consensus_label"].astype(int)
    direct["sequence_rna"] = direct["sequence"].map(canonicalize)
    if direct["sequence_rna"].duplicated().any():
        raise ValueError("duplicate direct-RNA sequence among labelled records")
    overlap = pd.read_csv(args.overlap_data).rename(
        columns={"direct_sequence_id": "sequence_id"}
    )
    direct = direct.merge(
        overlap[["sequence_id", "legacy_contained_in_direct", "high_similarity_90_80"]],
        on="sequence_id",
        how="left",
        validate="one_to_one",
    )

    fm = load_fm(args.upstream_script_dir)
    alphabet = fm.Alphabet.from_architecture("roberta_large", theme="rna")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_probabilities: list[np.ndarray] = []
    checkpoint_rows: list[dict[str, object]] = []
    architecture: dict[str, int | bool | str] | None = None

    for fold in folds:
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

        dataset = fm.FastaBatchedDataset(
            direct["label"].tolist(), direct["sequence_rna"].tolist(), mask_prob=0
        )
        batches = dataset.get_batch_indices(
            toks_per_batch=args.batch_toks, extra_toks_per_seq=1
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(),
            batch_sampler=batches,
            shuffle=False,
        )
        result_sequences, result_labels, probabilities, _ = predict(
            model, loader, device, args.truncate_num
        )
        scored = pd.DataFrame(
            {
                "sequence_rna": [canonicalize(value) for value in result_sequences],
                "result_label": result_labels,
                "probability": probabilities,
            }
        )
        aligned = direct[["sequence_rna", "label"]].merge(
            scored, on="sequence_rna", validate="one_to_one"
        )
        if not (aligned["label"] == aligned["result_label"]).all():
            raise ValueError(f"prediction labels do not align at fold {fold}")
        fold_probabilities.append(aligned["probability"].to_numpy(dtype=float))
        direct[f"probability_fold{fold}"] = fold_probabilities[-1]
        checkpoint_rows.append(
            {"fold": fold, "path": str(checkpoint), "sha256": sha256(checkpoint)}
        )
        print(json.dumps({"fold": fold, "n": len(aligned)}, sort_keys=True), flush=True)
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    direct["probability"] = np.mean(fold_probabilities, axis=0)
    subset_masks = {
        "all_labelled": np.ones(len(direct), dtype=bool),
        "non_control": ~direct["is_control"].astype(bool).to_numpy(),
        "legacy_contained": direct["legacy_contained_in_direct"].fillna(False).astype(bool).to_numpy(),
        "high_similarity_90_80": direct["high_similarity_90_80"].fillna(False).astype(bool).to_numpy(),
    }
    rows: list[dict[str, object]] = []
    for offset, (subset, mask) in enumerate(subset_masks.items()):
        labels = direct.loc[mask, "label"].to_numpy(dtype=np.int8)
        probabilities = direct.loc[mask, "probability"].to_numpy(dtype=float)
        if len(np.unique(labels)) < 2:
            continue
        rows.append(
            {
                "model": "IRES-RNAFM",
                "checkpoint_ensemble_size": len(folds),
                "training_assay": "released IRES-LM mixed-source legacy benchmark",
                "evaluation_assay": "direct RNA A-cap/G-cap nascent-translation MPRA",
                "subset": subset,
                **metrics(labels, probabilities),
                **bootstrap(labels, probabilities, args.bootstrap_replicates, args.seed + offset),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    prediction_columns = [
        "sequence_id",
        "label",
        "is_control",
        "legacy_contained_in_direct",
        "high_similarity_90_80",
        *[f"probability_fold{fold}" for fold in folds],
        "probability",
    ]
    direct[prediction_columns].to_csv(
        args.output_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "released_ires_rnafm_legacy_to_direct_rna_transfer",
        "scope": "classification-only reproduction",
        "direct_data": str(args.direct_data.resolve()),
        "direct_data_sha256": sha256(args.direct_data),
        "overlap_data": str(args.overlap_data.resolve()),
        "overlap_data_sha256": sha256(args.overlap_data),
        "folds": folds,
        "checkpoints": checkpoint_rows,
        "architecture_inferred_from_checkpoint": architecture,
        "device": str(device),
        "torch_version": torch.__version__,
        "batch_toks": args.batch_toks,
        "truncate_num": args.truncate_num,
        "direct_labels_used_for_training_or_thresholding": False,
        "decision_threshold": 0.5,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
