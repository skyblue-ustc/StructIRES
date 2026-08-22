#!/usr/bin/env python3
"""Train validation-clean RNA-FM and StructIRES classification heads.

The script deliberately uses the public *pretrained* RNA-FM checkpoint, not a
released IRES-RNAFM fold checkpoint.  The latter was selected with its test
fold and is therefore unsuitable as an initialization for a new held-out run.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import random
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ires_design.prediction import load_ires_ai_records


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("sequence_only", "structure_profile_only", "concat_profile", "gated_profile"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--rnafm-base", type=Path, required=True)
    parser.add_argument("--upstream-fm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--structure-profile-cache", type=Path)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--tokens-per-batch", type=int, default=8192)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last-layers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=.20)
    parser.add_argument("--pooling", choices=("bos", "mean"), default="bos",
                        help="RNA-FM readout; BOS matches the released upstream training command")
    parser.add_argument("--mask-prob", type=float, default=.15,
                        help="training-only RNA masked-LM probability, matching upstream default")
    parser.add_argument("--mlm-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_fm(script_dir: Path):
    package_dir = script_dir / "fm"
    package = types.ModuleType("fm")
    package.__package__ = "fm"; package.__path__ = [str(package_dir)]
    sys.modules["fm"] = package
    data = importlib.import_module("fm.data")
    model = importlib.import_module("fm.model")
    # The upstream pretrained factory references these through ``fm`` rather
    # than importing them locally.  The original package __init__ exports
    # them; reproduce only these safe, core exports while avoiding optional
    # downstream dependencies.
    package.Alphabet = data.Alphabet
    package.RNABertModel = model.RNABertModel
    pretrained = importlib.import_module("fm.pretrained")
    return data, pretrained


def locked_records(dataset: Path, assignments: Path):
    with gzip.open(assignments, "rt", encoding="utf-8", newline="") as handle:
        assigned = {row["sequence_id"]: row for row in csv.DictReader(handle)}
    records = [row for row in load_ires_ai_records(dataset) if row.sequence_id in assigned]
    records.sort(key=lambda row: row.sequence_id)
    if len(records) != len(assigned):
        raise ValueError("locked split and canonical records differ")
    for record in records:
        if hashlib.sha256(record.sequence.encode("ascii")).hexdigest() != assigned[record.sequence_id]["sequence_sha256"]:
            raise ValueError(f"sequence hash mismatch: {record.sequence_id}")
    return records, assigned


def best_f1_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, probability)
    if not len(thresholds): return .5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metric_row(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
    prediction = probability >= threshold
    negatives = labels == 0
    ece = 0.0
    for low, high in zip(np.linspace(0.0, .9, 10), np.linspace(.1, 1.0, 10)):
        in_bin = (probability >= low) & ((probability <= high) if high == 1.0 else (probability < high))
        if in_bin.any(): ece += float(in_bin.mean() * abs(probability[in_bin].mean() - labels[in_bin].mean()))
    return {"auc": float(roc_auc_score(labels, probability)), "aupr": float(average_precision_score(labels, probability)), "f1": float(f1_score(labels, prediction, zero_division=0)), "accuracy": float(accuracy_score(labels, prediction)), "mcc": float(matthews_corrcoef(labels, prediction)), "sensitivity": float((prediction[labels == 1]).mean()), "specificity": float((~prediction[negatives]).mean()), "ece10": float(ece), "threshold": float(threshold)}


def stratified_bootstrap(labels: np.ndarray, probability: np.ndarray, *, seed: int, replicates: int = 1000) -> dict[str, float]:
    """Post-hoc test uncertainty; it never affects model or threshold selection."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    random_state = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    auc, aupr = np.empty(replicates), np.empty(replicates)
    for index in range(replicates):
        sample = np.concatenate((random_state.choice(positive, len(positive), replace=True), random_state.choice(negative, len(negative), replace=True)))
        auc[index] = roc_auc_score(labels[sample], probability[sample])
        aupr[index] = average_precision_score(labels[sample], probability[sample])
    return {"auc_ci_low": float(np.quantile(auc, .025)), "auc_ci_high": float(np.quantile(auc, .975)), "aupr_ci_low": float(np.quantile(aupr, .025)), "aupr_ci_high": float(np.quantile(aupr, .975)), "bootstrap_replicates": replicates}


class PositionProfileEncoder(nn.Module):
    """Small local-secondary-structure encoder; inputs are fixed ViennaRNA profiles."""
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        return self.network(profile.transpose(1, 2))


class StructIRES(nn.Module):
    def __init__(self, backbone: nn.Module, *, variant: str, profile_channels: int, dropout: float, pooling: str, pad_token_id: int):
        super().__init__(); self.backbone = backbone; self.variant = variant; self.pooling = pooling; self.pad_token_id = pad_token_id
        self.sequence = nn.Sequential(nn.LayerNorm(640), nn.Linear(640, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU())
        if variant != "sequence_only":
            self.structure = PositionProfileEncoder(profile_channels, dropout)
        if variant == "concat_profile":
            self.combine = nn.Sequential(nn.Linear(256, 128), nn.GELU())
        if variant == "gated_profile":
            self.gate = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, tokens: torch.Tensor, structure: torch.Tensor | None = None, *, return_mlm: bool = False):
        if self.variant == "structure_profile_only":
            assert structure is not None
            return self.classifier(self.structure(structure)), None
        output = self.backbone(tokens, [12])
        representations = output["representations"][12]
        if self.pooling == "bos":
            encoded = representations[:, 0]
        else:
            # Faithful to upstream IRES_RNAFM.py: pool biological positions,
            # exclude BOS/EOS, and ignore padding (alphabet padding index=1).
            biological = representations[:, 1:-1]
            mask = (tokens[:, 1:-1] != self.pad_token_id).to(biological.dtype)
            encoded = (biological * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        sequence = self.sequence(encoded)
        if self.variant != "sequence_only":
            assert structure is not None
            structural = self.structure(structure)
            if self.variant == "concat_profile":
                sequence = self.combine(torch.cat((sequence, structural), dim=1))
            elif self.variant == "gated_profile":
                gate = self.gate(torch.cat((sequence, structural), dim=1))
                sequence = gate * sequence + (1.0 - gate) * structural
        return self.classifier(sequence), (output.get("logits") if return_mlm else None)


def configure_backbone(model: StructIRES, last_layers: int) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    for parameter in model.backbone.parameters(): parameter.requires_grad = False
    if last_layers >= 12:
        for parameter in model.backbone.parameters(): parameter.requires_grad = True
    elif last_layers:
        for index in range(12 - last_layers, 12):
            for parameter in model.backbone.layers[index].parameters(): parameter.requires_grad = True
        for parameter in model.backbone.emb_layer_norm_after.parameters(): parameter.requires_grad = True
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    return backbone, head


def batches(data_module, alphabet, indices: np.ndarray, sequences: list[str], labels: np.ndarray, tokens_per_batch: int, mask_prob: float):
    subset_sequences = [sequences[int(index)] for index in indices]
    subset_labels = [int(index) for index in indices]
    dataset = data_module.FastaBatchedDataset(subset_labels, subset_sequences, mask_prob=mask_prob)
    batch_indices = dataset.get_batch_indices(toks_per_batch=tokens_per_batch, extra_toks_per_seq=1)
    return torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batch_indices)


def run_epoch(model, loader, labels, structure, device, criterion, optimizer=None, *, mask_token_id: int, mlm_loss_weight: float, cls_loss_weight: float):
    train = optimizer is not None; model.train(train)
    # Frozen pretrained features must remain deterministic while the small
    # classifier head is learned.  Otherwise backbone dropout injects noise
    # although no backbone parameters can adapt to it.
    if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
        model.backbone.eval()
    probabilities, actual = [], []
    for indices, _, _, clean_tokens, masked_tokens, _ in loader:
        idx = torch.as_tensor(indices, dtype=torch.long)
        target = torch.as_tensor(labels[idx.numpy()], dtype=torch.long, device=device)
        clean_tokens, masked_tokens = clean_tokens.to(device), masked_tokens.to(device)
        tokens = masked_tokens if train and mlm_loss_weight > 0.0 else clean_tokens
        structural = None if structure is None else structure[idx].to(device)
        with torch.set_grad_enabled(train):
            logits, mlm_logits = model(tokens, structural, return_mlm=train and mlm_loss_weight > 0.0)
            loss = cls_loss_weight * criterion(logits, target)
            if mlm_logits is not None:
                masked_targets = clean_tokens.clone()
                masked_targets[masked_tokens != mask_token_id] = -100
                loss = loss + mlm_loss_weight * nn.functional.cross_entropy(mlm_logits.transpose(1, 2), masked_targets, ignore_index=-100)
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()); actual.extend(target.detach().cpu().tolist())
    return np.asarray(actual), np.asarray(probabilities)


def main() -> int:
    args = arguments()
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    profile_variant = args.variant != "sequence_only"
    if profile_variant and args.structure_profile_cache is None: raise ValueError(f"{args.variant} requires --structure-profile-cache")
    records, assignment = locked_records(args.dataset, args.assignments)
    sequences, labels = [row.sequence for row in records], np.asarray([row.label for row in records], dtype=np.int64)
    split = np.asarray([assignment[row.sequence_id]["split"] for row in records])
    train, validation, test = split == "train", split == "validation", split == "test"
    structure = None
    if args.structure_profile_cache:
        cache = np.load(args.structure_profile_cache, allow_pickle=False)
        if list(cache["sequence_ids"].astype(str)) != [row.sequence_id for row in records]: raise ValueError("structure cache IDs differ")
        values = np.asarray(cache["profiles"], dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != len(records): raise ValueError("invalid structure-profile cache")
        mean, std = values[train].mean(axis=(0, 1)), values[train].std(axis=(0, 1))
        structure = torch.from_numpy((values - mean[None, None, :]) / np.maximum(std[None, None, :], 1e-6))
    data_module, pretrained = load_fm(args.upstream_fm_dir)
    device = torch.device(args.device)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    all_rows, all_predictions = [], []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        backbone, alphabet = pretrained.rna_fm_t12(str(args.rnafm_base))
        model = StructIRES(backbone, variant=args.variant, profile_channels=0 if structure is None else structure.shape[2], dropout=args.dropout, pooling=args.pooling, pad_token_id=alphabet.tok_to_idx["<pad>"]).to(device)
        trainable_backbone, head = configure_backbone(model, args.unfreeze_last_layers)
        groups = [{"params": head, "lr": args.head_lr}]
        if trainable_backbone: groups.append({"params": trainable_backbone, "lr": args.backbone_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        weight = torch.tensor([1.0, float(train.sum() / labels[train].sum() - 1.0)], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        train_loader = batches(data_module, alphabet, np.flatnonzero(train), sequences, labels, args.tokens_per_batch, args.mask_prob if args.variant != "structure_profile_only" else 0.0)
        val_loader = batches(data_module, alphabet, np.flatnonzero(validation), sequences, labels, args.tokens_per_batch, 0.0)
        test_loader = batches(data_module, alphabet, np.flatnonzero(test), sequences, labels, args.tokens_per_batch, 0.0)
        best, best_state, best_threshold = -np.inf, None, .5
        for epoch in range(1, args.epochs + 1):
            run_epoch(model, train_loader, labels, structure, device, criterion, optimizer, mask_token_id=alphabet.tok_to_idx["<mask>"], mlm_loss_weight=args.mlm_loss_weight, cls_loss_weight=args.cls_loss_weight)
            y_val, p_val = run_epoch(model, val_loader, labels, structure, device, criterion, mask_token_id=alphabet.tok_to_idx["<mask>"], mlm_loss_weight=0.0, cls_loss_weight=args.cls_loss_weight)
            threshold = best_f1_threshold(y_val, p_val); values = metric_row(y_val, p_val, threshold)
            print(json.dumps({"seed": seed, "epoch": epoch, "validation": values}), flush=True)
            if values["aupr"] > best:
                best, best_threshold = values["aupr"], threshold
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        assert best_state is not None; model.load_state_dict(best_state)
        y_test, p_test = run_epoch(model, test_loader, labels, structure, device, criterion, mask_token_id=alphabet.tok_to_idx["<mask>"], mlm_loss_weight=0.0, cls_loss_weight=args.cls_loss_weight)
        result = {"seed": seed, "best_validation_aupr": best, **metric_row(y_test, p_test, best_threshold), **stratified_bootstrap(y_test, p_test, seed=seed)}; all_rows.append(result)
        for record, probability in zip(np.asarray(records, dtype=object)[test], p_test): all_predictions.append({"seed": seed, "sequence_id": record.sequence_id, "label": record.label, "probability": float(probability)})
        args.output_dir.mkdir(parents=True, exist_ok=True); torch.save({"model": best_state, "seed": seed, "variant": args.variant, "best_validation_aupr": best}, args.output_dir / f"best_seed{seed}.pt")
        del model, backbone; torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "test_metrics.csv").open("w", encoding="utf-8", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    with gzip.open(args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0])); writer.writeheader(); writer.writerows(all_predictions)
    manifest = {"schema_version": 1, "experiment": "structires_rnafm_validation_clean", "variant": args.variant, "pooling": args.pooling, "mask_prob_training_only": args.mask_prob, "mlm_loss_weight": args.mlm_loss_weight, "cls_loss_weight": args.cls_loss_weight, "dataset_sha256": sha256(args.dataset), "assignments_sha256": sha256(args.assignments), "rnafm_base_sha256": sha256(args.rnafm_base), "structure_profile_cache_sha256": sha256(args.structure_profile_cache) if args.structure_profile_cache else None, "seeds": seeds, "epochs": args.epochs, "unfreeze_last_layers": args.unfreeze_last_layers, "model_selection": "validation AUPR", "test_labels_used_for_training_threshold_selection_or_epoch_selection": False}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
