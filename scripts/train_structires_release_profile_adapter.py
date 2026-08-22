#!/usr/bin/env python3
"""Train a checkpoint-native, position-aware StructIRES residual adapter.

This is the full-benchmark successor to the global-feature adapter.  It keeps
the released IRES-RNAFM classifier frozen, encodes label-free ViennaRNA
ensemble profiles with a small masked CNN, and injects the resulting residual
through a learned gate.  At initialization the residual path is exactly zero,
so predictions numerically equal the released sequence-only checkpoint.

The upstream fold's official test records are never used for selection: a
deterministic validation partition is made solely within its upstream train
records.  Output directories are immutable run artifacts.
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
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("profile_fusion_frozen", "sequence_head", "profile_fusion_head"),
                        default="profile_fusion_frozen",
                        help="Frozen residual reference, or a matched trainable released-head control/fusion pair.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rnafm-base", type=Path, required=True)
    parser.add_argument("--upstream-fm-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(10))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    parser.add_argument("--truncate-num", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=.10)
    parser.add_argument("--rnafm-adaptation", choices=("frozen", "last_layer", "full"), default="frozen",
                        help="Keep RNA-FM frozen, adapt only layer 12, or adapt the full RNA-FM backbone.")
    parser.add_argument("--train-mask-prob", type=float, default=0.,
                        help="Mask probability for training inputs; validation and test always remain unmasked.")
    parser.add_argument("--mlm-loss-weight", type=float, default=0.,
                        help="Weight on author-style masked-nucleotide reconstruction loss. Requires RNA-FM adaptation.")
    parser.add_argument("--classification-loss-weight", type=float, default=1.,
                        help="Multiplier on the class-weighted IRES cross-entropy loss.")
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
    def __init__(self, rnafm: nn.Module, dropout: float = .5):
        super().__init__(); self.rnafm = rnafm
        self.fc = nn.Linear(640, 40); self.dropout3 = nn.Dropout(dropout); self.relu = nn.ReLU(); self.output = nn.Linear(40, 2)

    def hidden_with_lm(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.rnafm(tokens, [12])
        return self.relu(self.fc(outputs["representations"][12][:, 0])), outputs["logits"]

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.hidden_with_lm(tokens)
        return hidden

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout3(self.hidden(tokens)))

    def forward_with_lm(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, lm_logits = self.hidden_with_lm(tokens)
        return self.output(self.dropout3(hidden)), lm_logits


class PositionProfileResidualAdapter(nn.Module):
    """Masked local structure encoder plus a zero-initialized residual gate."""
    def __init__(self, base: ReleasedRNAFM, dropout: float, *, train_release_head: bool = False):
        super().__init__(); self.base = base
        for parameter in self.base.rnafm.parameters(): parameter.requires_grad = False
        if not train_release_head:
            for parameter in self.base.fc.parameters(): parameter.requires_grad = False
            for parameter in self.base.output.parameters(): parameter.requires_grad = False
        self.profile = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(32, 40, kernel_size=5, padding=2), nn.GELU(),
        )
        self.profile_dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(40, 40)
        self.gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate[0].weight); nn.init.zeros_(self.gate[0].bias)

    def structure_hidden(self, profile: torch.Tensor) -> torch.Tensor:
        # First four channels are train-split normalized structural values;
        # channel five is an unnormalized sequence-validity mask.
        values, valid = profile[:, :, :4], profile[:, :, 4]
        local = self.profile(values.transpose(1, 2)).transpose(1, 2)
        pooled = (local * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1.)
        return self.residual(self.profile_dropout(pooled))

    def forward(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        sequence = self.base.hidden(tokens)
        structural = self.structure_hidden(profile)
        gate = self.gate(torch.cat((sequence, structural), dim=1))
        # Reuse the released classifier's dropout before its output layer.
        # This keeps head-only and fusion-head training matched; in eval mode
        # dropout is identity, preserving the zero-initialization invariant.
        return self.base.output(self.base.dropout3(sequence + gate * structural))

    def forward_with_lm(self, tokens: torch.Tensor, profile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, lm_logits = self.base.hidden_with_lm(tokens)
        structural = self.structure_hidden(profile)
        gate = self.gate(torch.cat((sequence, structural), dim=1))
        return self.base.output(self.base.dropout3(sequence + gate * structural)), lm_logits


def make_loader(data_module, alphabet, indices: np.ndarray, sequences: list[str], tokens_per_batch: int, *, mask_prob: float = 0.):
    dataset = data_module.FastaBatchedDataset(indices.tolist(), [sequences[int(index)] for index in indices], mask_prob=mask_prob)
    batches = dataset.get_batch_indices(toks_per_batch=tokens_per_batch, extra_toks_per_seq=2)
    return torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)


def profile_train_statistics(profiles, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute masked channel moments without materializing the full cache."""
    total = np.zeros(4, dtype=np.float64); squared = np.zeros(4, dtype=np.float64); count = 0.0
    for start in range(0, len(train), 256):
        value = np.asarray(profiles[train[start:start + 256]], dtype=np.float64)
        mask = value[:, :, 4:5]
        total += (value[:, :, :4] * mask).sum(axis=(0, 1))
        squared += ((value[:, :, :4] ** 2) * mask).sum(axis=(0, 1))
        count += float(mask.sum())
    if count <= 0.0: raise RuntimeError("profile cache has no valid structural positions in train split")
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - mean ** 2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_profile_view(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize structural channels for either scalar or batched cache reads."""
    value = np.asarray(raw, dtype=np.float32).copy()
    if value.shape[-1] != 5:
        raise ValueError(f"expected five profile channels, received {value.shape}")
    value[..., :4] = (value[..., :4] - mean) / std
    value[..., :4] *= value[..., 4:5]
    return value


def probabilities(model, loader, profiles, device, *, adapter: bool, truncate_num: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); labels, scores = [], []
    with torch.no_grad():
        for indices, _, _, tokens, _, _ in loader:
            index = np.asarray(indices, dtype=np.int64)
            tokens = tokens[:, :truncate_num].to(device)
            logits = model(tokens, torch.from_numpy(np.asarray(profiles[index])).to(device)) if adapter else model(tokens)
            labels.extend(index.tolist()); scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def freeze_backbone_keep_release_head(base: ReleasedRNAFM) -> None:
    """Train the published classifier head while keeping RNA-FM fixed."""
    for parameter in base.rnafm.parameters(): parameter.requires_grad = False
    for parameter in base.fc.parameters(): parameter.requires_grad = True
    for parameter in base.output.parameters(): parameter.requires_grad = True


def configure_rnafm_adaptation(base: ReleasedRNAFM, mode: str) -> None:
    """Set the RNA-FM trainability for a paired sequence/fusion experiment."""
    for name, parameter in base.rnafm.named_parameters():
        parameter.requires_grad = mode == "full" or (mode == "last_layer" and name.startswith("layers.11."))


def main() -> int:
    args = arguments()
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    frame = load_fold_rows(args.dataset, args.fold)
    ids = np.load(args.profile_dir / "sequence_ids.npy", allow_pickle=False).astype(str)
    profiles_raw = np.load(args.profile_dir / "profiles.npy", mmap_mode="r")
    if profiles_raw.shape[:2] != (46774, args.truncate_num) or profiles_raw.shape[2] != 5:
        raise ValueError(f"unexpected profile array shape {profiles_raw.shape}")
    profile_by_id = {identifier: index for index, identifier in enumerate(ids)}
    if len(profile_by_id) != len(ids) or set(frame.ID.astype(str)) != set(profile_by_id): raise ValueError("profile IDs and benchmark IDs differ")
    frame["profile_index"] = [profile_by_id[str(identifier)] for identifier in frame.ID]
    order = np.argsort(frame.ID.astype(str).to_numpy()); frame = frame.iloc[order].reset_index(drop=True)
    # The cache and the frame are both lexicographically ordered by string ID.
    # Keep the cache memory-mapped instead of creating a ~0.9-GB fancy-indexed
    # copy per fold; this is also a direct integrity assertion on the ordering.
    if not np.array_equal(np.asarray(frame.profile_index, dtype=np.int64), np.arange(len(frame))):
        raise ValueError("profile cache order does not match sorted benchmark IDs")
    profiles = profiles_raw
    labels = frame.label.to_numpy(dtype=np.int64); sequences = frame.Sequence.tolist()
    source_train = frame.type.to_numpy() == "train"; source_test = frame.type.to_numpy() == "test"
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    train_local, validation_local = next(splitter.split(np.flatnonzero(source_train), labels[source_train]))
    upstream_train_indices = np.flatnonzero(source_train)
    train, validation, test = upstream_train_indices[train_local], upstream_train_indices[validation_local], np.flatnonzero(source_test)
    # Standardize structural channels using train records and valid positions only;
    # preserve the fifth mask channel exactly.
    print(json.dumps({"stage": "profile_train_statistics", "fold": args.fold, "n_train": int(train.size)}), flush=True)
    mean, std = profile_train_statistics(profiles, train)
    print(json.dumps({"stage": "profile_train_statistics_complete", "fold": args.fold}), flush=True)

    class NormalizedProfiles:
        def __getitem__(self, index):
            return normalize_profile_view(profiles[index], mean, std)
    normalized_profiles = NormalizedProfiles()
    data_module, pretrained = load_fm(args.upstream_fm_dir)
    print(json.dumps({"stage": "loading_released_checkpoint", "fold": args.fold}), flush=True)
    backbone, alphabet = pretrained.rna_fm_t12(str(args.rnafm_base))
    base = ReleasedRNAFM(backbone)
    release_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    release_state = {key.removeprefix("module."): value for key, value in release_state.items()}
    incompatible = base.load_state_dict(release_state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    device = torch.device(args.device); base = base.to(device).eval()
    if args.mlm_loss_weight and args.rnafm_adaptation == "frozen":
        raise ValueError("--mlm-loss-weight requires --rnafm-adaptation last_layer or full")
    if not 0. <= args.train_mask_prob < 1.:
        raise ValueError("--train-mask-prob must lie in [0, 1)")
    train_loader = make_loader(data_module, alphabet, train, sequences, args.tokens_per_batch,
                               mask_prob=args.train_mask_prob)
    validation_loader = make_loader(data_module, alphabet, validation, sequences, args.tokens_per_batch)
    test_loader = make_loader(data_module, alphabet, test, sequences, args.tokens_per_batch)
    val_index, val_probability = probabilities(base, validation_loader, normalized_profiles, device, adapter=False, truncate_num=args.truncate_num)
    test_index, baseline_test_probability = probabilities(base, test_loader, normalized_profiles, device, adapter=False, truncate_num=args.truncate_num)
    baseline_threshold = best_f1_threshold(labels[val_index], val_probability)
    baseline = metrics(labels[test_index], baseline_test_probability, baseline_threshold)
    print(json.dumps({"stage": "baseline_evaluated", "fold": args.fold, "baseline": baseline}), flush=True)
    uses_profile = args.variant != "sequence_head"
    if args.variant == "sequence_head":
        freeze_backbone_keep_release_head(base)
        model = base
        trained_model_name = "sequence_head_validation_selected"
    else:
        model = PositionProfileResidualAdapter(
            base, args.dropout, train_release_head=args.variant == "profile_fusion_head"
        ).to(device)
        trained_model_name = "adapter_validation_selected" if args.variant == "profile_fusion_frozen" else "profile_fusion_head_validation_selected"
    if args.variant == "profile_fusion_frozen" and args.rnafm_adaptation != "frozen":
        raise ValueError("profile_fusion_frozen is incompatible with RNA-FM adaptation")
    configure_rnafm_adaptation(base, args.rnafm_adaptation)
    initial_index, initial_probability = probabilities(model, test_loader, normalized_profiles, device, adapter=uses_profile, truncate_num=args.truncate_num)
    if not np.array_equal(test_index, initial_index) or not np.allclose(baseline_test_probability, initial_probability, rtol=0., atol=1e-7):
        raise RuntimeError("initial matched model does not reproduce the released checkpoint")
    print(json.dumps({"stage": "zero_initialization_equivalence_passed", "fold": args.fold}), flush=True)
    optimizer = torch.optim.AdamW([item for item in model.parameters() if item.requires_grad], lr=args.lr, weight_decay=1e-4)
    positive_weight = float(train.size / labels[train].sum() - 1.)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., positive_weight], device=device))
    mask_token_id = int(alphabet.tok_to_idx["<mask>"])
    best, best_state, best_threshold, curve, stale = -np.inf, None, .5, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep a frozen RNA-FM deterministic, while allowing the author-style
        # last-layer/full adaptation settings to train normally.
        rnafm = model.base.rnafm if uses_profile else model.rnafm
        if args.rnafm_adaptation == "frozen": rnafm.eval()
        for indices, _, _, tokens, masked_tokens, _ in train_loader:
            index = np.asarray(indices, dtype=np.int64)
            token_batch = tokens[:, :args.truncate_num].to(device)
            masked_token_batch = masked_tokens[:, :args.truncate_num].to(device)
            classification_input = masked_token_batch if args.train_mask_prob else token_batch
            if uses_profile:
                profile_batch = torch.from_numpy(normalized_profiles[index]).to(device)
                logits, lm_logits = model.forward_with_lm(classification_input, profile_batch)
            else:
                logits, lm_logits = model.forward_with_lm(classification_input)
            target = torch.as_tensor(labels[index], dtype=torch.long, device=device)
            class_loss = args.classification_loss_weight * criterion(logits, target)
            if args.mlm_loss_weight:
                mlm_target = torch.full_like(token_batch, -1)
                masked_position = masked_token_batch == mask_token_id
                mlm_target[masked_position] = token_batch[masked_position]
                mlm_loss = torch.nn.functional.cross_entropy(
                    lm_logits.transpose(1, 2), mlm_target, ignore_index=-1, reduction="mean"
                )
            else:
                mlm_loss = torch.zeros((), device=device)
            loss = class_loss + args.mlm_loss_weight * mlm_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        val_index, val_probability = probabilities(model, validation_loader, normalized_profiles, device, adapter=uses_profile, truncate_num=args.truncate_num)
        threshold = best_f1_threshold(labels[val_index], val_probability); report = metrics(labels[val_index], val_probability, threshold)
        curve.append({"epoch": epoch, **report}); print(json.dumps({"epoch": epoch, "validation": report}), flush=True)
        if report["aupr"] > best:
            best, best_threshold, stale = report["aupr"], threshold, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                print(json.dumps({"early_stopping": True, "epoch": epoch, "best_validation_aupr": best}), flush=True)
                break
    assert best_state is not None; model.load_state_dict(best_state)
    test_index, test_probability = probabilities(model, test_loader, normalized_profiles, device, adapter=uses_profile, truncate_num=args.truncate_num)
    adapter = metrics(labels[test_index], test_probability, best_threshold)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({trained_model_name: adapter, "baseline_released_checkpoint": baseline, "best_validation_aupr": best, "epochs_completed": len(curve), "fold": args.fold, "variant": args.variant, "validation_curve": curve}, handle, indent=2, sort_keys=True); handle.write("\n")
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", *adapter.keys()]; writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerow({"model": "baseline_released_checkpoint", **baseline}); writer.writerow({"model": trained_model_name, **adapter})
    with gzip.open(args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "label", "baseline_probability", "adapter_probability"]); writer.writeheader()
        for index, label, base_probability, adapter_probability in zip(test_index, labels[test_index], baseline_test_probability, test_probability):
            writer.writerow({"sample_index": int(index), "label": int(label), "baseline_probability": float(base_probability), "adapter_probability": float(adapter_probability)})
    torch.save({"model": best_state, "fold": args.fold, "best_validation_aupr": best}, args.output_dir / "best_adapter.pt")
    architecture = {"profile_fusion_frozen": "frozen released RNA-FM checkpoint plus masked-CNN position profile residual gate", "sequence_head": "released classifier head with configurable RNA-FM adaptation", "profile_fusion_head": "released classifier head plus masked-CNN position-profile residual gate with configurable RNA-FM adaptation"}[args.variant]
    manifest = {"schema_version": 1, "experiment": "structires_release_position_profile_adapter", "variant": args.variant, "fold": args.fold, "dataset_sha256": sha256(args.dataset), "profile_manifest_sha256": sha256(args.profile_dir / "manifest.json"), "checkpoint_sha256": sha256(args.checkpoint), "rnafm_base_sha256": sha256(args.rnafm_base), "validation_fraction": args.validation_fraction, "model_selection": "validation AUPR", "test_labels_used_for_selection": False, "architecture": architecture, "rnafm_adaptation": args.rnafm_adaptation, "train_mask_prob": args.train_mask_prob, "mlm_loss_weight": args.mlm_loss_weight, "classification_loss_weight": args.classification_loss_weight}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
