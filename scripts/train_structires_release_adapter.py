#!/usr/bin/env python3
"""Attach a zero-initialized structure adapter to a released IRES-RNAFM fold.

This is intentionally a *baseline modification*, not a new RNA foundation
model.  Before adapter training, every trainable structural contribution is
zero; predictions exactly equal the released IRES-RNAFM checkpoint.  The
upstream fold's official test split is never used for epoch or threshold
selection: a deterministic validation split is carved from its train portion.
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rnafm-base", type=Path, required=True)
    parser.add_argument("--upstream-fm-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(10))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    # The released IRES-RNAFM command used --truncate --truncate_num 1024.
    # Matching that preprocessing is required for long input records and for
    # a faithful checkpoint-native evaluation.
    parser.add_argument("--truncate-num", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=.10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fm(script_dir: Path):
    package_dir = script_dir / "fm"
    package = types.ModuleType("fm")
    package.__package__ = "fm"; package.__path__ = [str(package_dir)]
    sys.modules["fm"] = package
    data = importlib.import_module("fm.data")
    model = importlib.import_module("fm.model")
    package.Alphabet = data.Alphabet; package.RNABertModel = model.RNABertModel
    return data, importlib.import_module("fm.pretrained")


def load_fold_rows(dataset: Path, fold: int):
    import pandas as pd
    with zipfile.ZipFile(dataset) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv") and "__MACOSX" not in name]
        if len(names) != 1: raise ValueError("expected exactly one canonical CSV inside dataset zip")
        with archive.open(names[0]) as handle: frame = pd.read_csv(handle)
    frame = frame.loc[frame["fold"] == fold, ["ID", "Sequence", "IRES_class_600", "type"]].copy()
    if frame.ID.duplicated().any() or len(frame) != 46774: raise ValueError("unexpected per-fold unique record set")
    frame.Sequence = frame.Sequence.str.upper().str.replace("T", "U", regex=False)
    frame.rename(columns={"IRES_class_600": "label"}, inplace=True)
    return frame


def best_f1_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = precision_recall_curve(labels, probability)
    if not len(threshold): return .5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(threshold[int(np.nanargmax(f1))])


def metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
    prediction = probability >= threshold; negative = labels == 0
    ece = 0.0
    for low, high in zip(np.linspace(0., .9, 10), np.linspace(.1, 1., 10)):
        mask = (probability >= low) & ((probability <= high) if high == 1. else (probability < high))
        if mask.any(): ece += float(mask.mean() * abs(probability[mask].mean() - labels[mask].mean()))
    return {"auc": float(roc_auc_score(labels, probability)), "aupr": float(average_precision_score(labels, probability)), "f1": float(f1_score(labels, prediction, zero_division=0)), "accuracy": float(accuracy_score(labels, prediction)), "sensitivity": float(prediction[labels == 1].mean()), "specificity": float((~prediction[negative]).mean()), "mcc": float(matthews_corrcoef(labels, prediction)), "ece10": float(ece), "threshold": threshold}


class ReleasedRNAFM(nn.Module):
    """Exact released IRES-RNAFM classifier topology (BOS -> 40 -> 2)."""
    def __init__(self, rnafm: nn.Module, dropout: float = .5):
        super().__init__(); self.rnafm = rnafm
        self.fc = nn.Linear(640, 40); self.dropout3 = nn.Dropout(dropout); self.relu = nn.ReLU(); self.output = nn.Linear(40, 2)

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc(self.rnafm(tokens, [12])["representations"][12][:, 0]))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout3(self.hidden(tokens)))


class StructuralResidualAdapter(nn.Module):
    """Frozen release model plus a zero-initialized gated structural residual."""
    def __init__(self, base: ReleasedRNAFM, feature_dim: int, dropout: float):
        super().__init__(); self.base = base
        for parameter in self.base.parameters(): parameter.requires_grad = False
        self.structure = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 40))
        self.gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        # h_struct=0 at initialization, hence adapter logits exactly equal the
        # released baseline logits in eval mode.
        nn.init.zeros_(self.structure[-1].weight); nn.init.zeros_(self.structure[-1].bias)
        nn.init.zeros_(self.gate[0].weight); nn.init.zeros_(self.gate[0].bias)

    def forward(self, tokens: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        sequence = self.base.hidden(tokens)
        structural = self.structure(features)
        gate = self.gate(torch.cat((sequence, structural), dim=1))
        return self.base.output(sequence + gate * structural)


def make_loader(data_module, alphabet, indices: np.ndarray, sequences: list[str], labels: np.ndarray, tokens_per_batch: int):
    dataset = data_module.FastaBatchedDataset(indices.tolist(), [sequences[int(index)] for index in indices], mask_prob=0)
    batches = dataset.get_batch_indices(toks_per_batch=tokens_per_batch, extra_toks_per_seq=2)
    return torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)


def probabilities(model, loader, features, device, *, adapter: bool, truncate_num: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); labels, scores = [], []
    with torch.no_grad():
        for indices, _, _, tokens, _, _ in loader:
            index = torch.as_tensor(indices, dtype=torch.long)
            tokens = tokens[:, :truncate_num].to(device)
            logits = model(tokens, features[index].to(device)) if adapter else model(tokens)
            labels.extend(index.tolist()); scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def main() -> int:
    args = arguments()
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    frame = load_fold_rows(args.dataset, args.fold)
    cache = np.load(args.feature_cache, allow_pickle=False)
    cache_ids = list(cache["sequence_ids"].astype(str)); feature_by_id = {identifier: index for index, identifier in enumerate(cache_ids)}
    if set(frame.ID) != set(cache_ids): raise ValueError("dataset IDs and structural cache IDs differ")
    frame["feature_index"] = [feature_by_id[identifier] for identifier in frame.ID]
    order = np.argsort(frame.ID.to_numpy()); frame = frame.iloc[order].reset_index(drop=True)
    features_raw = np.asarray(cache["features"], dtype=np.float32)
    features = features_raw[np.asarray(frame.feature_index, dtype=np.int64)]
    labels = frame.label.to_numpy(dtype=np.int64); sequences = frame.Sequence.tolist()
    source_train = frame.type.to_numpy() == "train"; source_test = frame.type.to_numpy() == "test"
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    train_local, validation_local = next(splitter.split(np.flatnonzero(source_train), labels[source_train]))
    upstream_train_indices = np.flatnonzero(source_train)
    train, validation = upstream_train_indices[train_local], upstream_train_indices[validation_local]
    test = np.flatnonzero(source_test)
    mean, std = features[train].mean(0), features[train].std(0)
    feature_tensor = torch.from_numpy((features - mean) / np.maximum(std, 1e-6))
    data_module, pretrained = load_fm(args.upstream_fm_dir)
    backbone, alphabet = pretrained.rna_fm_t12(str(args.rnafm_base))
    base = ReleasedRNAFM(backbone)
    release_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    release_state = {key.removeprefix("module."): value for key, value in release_state.items()}
    incompatible = base.load_state_dict(release_state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    device = torch.device(args.device); base = base.to(device).eval()
    train_loader = make_loader(data_module, alphabet, train, sequences, labels, args.tokens_per_batch)
    validation_loader = make_loader(data_module, alphabet, validation, sequences, labels, args.tokens_per_batch)
    test_loader = make_loader(data_module, alphabet, test, sequences, labels, args.tokens_per_batch)
    # Checkpoint-native baseline: threshold selected only on validation.
    val_index, val_probability = probabilities(base, validation_loader, feature_tensor, device, adapter=False, truncate_num=args.truncate_num)
    test_index, test_probability = probabilities(base, test_loader, feature_tensor, device, adapter=False, truncate_num=args.truncate_num)
    baseline_threshold = best_f1_threshold(labels[val_index], val_probability)
    baseline = metrics(labels[test_index], test_probability, baseline_threshold)
    model = StructuralResidualAdapter(base, features.shape[1], args.dropout).to(device)
    # This is an invariant of the proposed residual adapter: before learning,
    # it must be numerically identical to the released sequence-only model.
    initial_index, initial_probability = probabilities(model, test_loader, feature_tensor, device, adapter=True, truncate_num=args.truncate_num)
    if not np.array_equal(test_index, initial_index):
        raise RuntimeError("baseline and adapter test ordering differ")
    if not np.allclose(test_probability, initial_probability, rtol=0., atol=1e-7):
        raise RuntimeError("zero-initialized adapter does not reproduce the released baseline")
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=1e-4)
    positive_weight = float(train.size / labels[train].sum() - 1.)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., positive_weight], device=device))
    best, best_state, best_threshold, curve = -np.inf, None, .5, []
    for epoch in range(1, args.epochs + 1):
        model.train(); model.base.eval()
        for indices, _, _, tokens, _, _ in train_loader:
            index = torch.as_tensor(indices, dtype=torch.long)
            logits = model(tokens[:, :args.truncate_num].to(device), feature_tensor[index].to(device))
            target = torch.as_tensor(labels[index.numpy()], dtype=torch.long, device=device)
            loss = criterion(logits, target); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        val_index, val_probability = probabilities(model, validation_loader, feature_tensor, device, adapter=True, truncate_num=args.truncate_num)
        threshold = best_f1_threshold(labels[val_index], val_probability); report = metrics(labels[val_index], val_probability, threshold)
        curve.append({"epoch": epoch, **report}); print(json.dumps({"epoch": epoch, "validation": report}), flush=True)
        if report["aupr"] > best:
            best, best_threshold = report["aupr"], threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None; model.load_state_dict(best_state)
    adapter_test_index, adapter_test_probability = probabilities(model, test_loader, feature_tensor, device, adapter=True, truncate_num=args.truncate_num)
    if not np.array_equal(test_index, adapter_test_index):
        raise RuntimeError("baseline and final adapter test ordering differ")
    adapter_metrics = metrics(labels[test_index], adapter_test_probability, best_threshold)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"fold": args.fold, "baseline_released_checkpoint": baseline, "adapter_validation_selected": adapter_metrics, "best_validation_aupr": best, "validation_curve": curve}, handle, indent=2, sort_keys=True); handle.write("\n")
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *baseline.keys()])
        writer.writeheader()
        writer.writerow({"model": "released_checkpoint", **baseline})
        writer.writerow({"model": "structires_adapter", **adapter_metrics})
    with gzip.open(args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "baseline_probability", "adapter_probability"]); writer.writeheader()
        for index, base_value, adapter_value in zip(test_index, test_probability, adapter_test_probability):
            writer.writerow({"id": frame.ID.iloc[int(index)], "label": int(labels[index]), "baseline_probability": float(base_value), "adapter_probability": float(adapter_value)})
    torch.save({"adapter_state": best_state, "fold": args.fold, "best_validation_aupr": best}, args.output_dir / "best_adapter.pt")
    manifest = {"schema_version": 1, "experiment": "structires_release_checkpoint_zero_initialized_structure_adapter", "fold": args.fold, "dataset_sha256": sha256(args.dataset), "feature_cache_sha256": sha256(args.feature_cache), "checkpoint_sha256": sha256(args.checkpoint), "rnafm_base_sha256": sha256(args.rnafm_base), "validation_fraction": args.validation_fraction, "seed": args.seed, "epochs": args.epochs, "truncate_num": args.truncate_num, "baseline_parameters_frozen": True, "structure_residual_zero_initialized": True, "selection": "validation AUPR", "upstream_test_used_for_selection": False}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
