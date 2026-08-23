#!/usr/bin/env python3
"""Train a checkpoint-native StructIRES fusion with sparse MFE contact edges.

The released RNA-FM IRES classifier is frozen.  A small contact encoder pools
RNA-FM residue-pair representations over explicit ViennaRNA MFE base-pair
edges, then contributes a zero-initialized gated residual to the released
classifier hidden state.  Pair contacts are label-free and the official fold
test is never used for selection.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rnafm-base", type=Path, required=True)
    parser.add_argument("--upstream-fm-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    parser.add_argument("--truncate-num", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=.10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_utility():
    source = Path(__file__).with_name("train_structires_release_profile_adapter.py")
    spec = importlib.util.spec_from_file_location("structires_profile_utility", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load profile-adapter utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MFEContactResidualAdapter(nn.Module):
    """Frozen RNA-FM with learned pooling over actual MFE base-pair edges."""
    def __init__(self, base: nn.Module, dropout: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * 640, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 40)
        )
        self.gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        nn.init.zeros_(self.pair_encoder[-1].weight)
        nn.init.zeros_(self.pair_encoder[-1].bias)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def contact_hidden(self, residue: torch.Tensor, indices: np.ndarray, offsets: np.ndarray,
                       pairs: np.ndarray, tokens: torch.Tensor) -> torch.Tensor:
        parts, owners = [], []
        for batch, record in enumerate(indices.tolist()):
            left = int(offsets[record]); right = int(offsets[record + 1])
            pair = pairs[left:right]
            # Cache positions are biological 0-based coordinates; tokens have
            # a leading BOS and use 1 as right-padding after the terminal EOS.
            length = int((tokens[batch, 1:-1] != 1).sum().item())
            pair = pair[(pair[:, 0] < length) & (pair[:, 1] < length)]
            if len(pair):
                value = torch.as_tensor(pair, dtype=torch.long, device=residue.device) + 1
                h_left, h_right = residue[batch, value[:, 0]], residue[batch, value[:, 1]]
                parts.append(torch.cat((h_left, h_right, h_left * h_right, (h_left - h_right).abs()), dim=1))
                owners.append(torch.full((len(pair),), batch, dtype=torch.long, device=residue.device))
        pooled = torch.zeros((len(indices), 40), dtype=residue.dtype, device=residue.device)
        if not parts:
            return pooled
        encoded = self.pair_encoder(torch.cat(parts, dim=0))
        owner = torch.cat(owners, dim=0)
        pooled.index_add_(0, owner, encoded)
        counts = torch.bincount(owner, minlength=len(indices)).to(residue.dtype).clamp_min(1.).unsqueeze(1)
        return pooled / counts

    def forward(self, tokens: torch.Tensor, indices: np.ndarray, offsets: np.ndarray, pairs: np.ndarray) -> torch.Tensor:
        # The RNA-FM and released head are frozen; autograd therefore only
        # retains the compact contact branch, keeping this experiment tractable.
        with torch.no_grad():
            outputs = self.base.rnafm(tokens, [12])
            residue = outputs["representations"][12]
            sequence = self.base.relu(self.base.fc(residue[:, 0]))
        structure = self.contact_hidden(residue, indices, offsets, pairs, tokens)
        gate = self.gate(torch.cat((sequence, structure), dim=1))
        return self.base.output(self.base.dropout3(sequence + gate * structure))


def probabilities(model, loader, offsets, pairs, device, truncate_num: int):
    model.eval(); indices, score = [], []
    with torch.no_grad():
        for batch_index, _, _, tokens, _, _ in loader:
            record = np.asarray(batch_index, dtype=np.int64)
            logits = model(tokens[:, :truncate_num].to(device), record, offsets, pairs)
            indices.extend(record.tolist())
            score.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(indices, dtype=np.int64), np.asarray(score, dtype=np.float64)


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    utility = load_utility()
    frame = utility.load_fold_rows(args.dataset, args.fold)
    cache_ids = np.load(args.contact_dir / "sequence_ids.npy", allow_pickle=False).astype(str)
    offsets = np.load(args.contact_dir / "pair_offsets.npy", allow_pickle=False, mmap_mode="r")
    pairs = np.load(args.contact_dir / "pairs.npy", allow_pickle=False, mmap_mode="r")
    by_id = {identifier: index for index, identifier in enumerate(cache_ids)}
    if len(by_id) != len(cache_ids) or set(frame.ID.astype(str)) != set(by_id):
        raise ValueError("contact cache IDs and fold IDs differ")
    frame["contact_index"] = [by_id[identifier] for identifier in frame.ID.astype(str)]
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    if not np.array_equal(frame.contact_index.to_numpy(dtype=np.int64), np.arange(len(frame))):
        raise ValueError("contact cache ordering is not canonical")
    labels = frame.label.to_numpy(dtype=np.int64); sequence = frame.Sequence.tolist()
    source_train = frame.type.to_numpy() == "train"; source_test = frame.type.to_numpy() == "test"
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    local_train, local_validation = next(split.split(np.flatnonzero(source_train), labels[source_train]))
    upstream_train = np.flatnonzero(source_train)
    train, validation, test = upstream_train[local_train], upstream_train[local_validation], np.flatnonzero(source_test)
    data_module, pretrained = utility.load_fm(args.upstream_fm_dir)
    backbone, alphabet = pretrained.rna_fm_t12(str(args.rnafm_base))
    base = utility.ReleasedRNAFM(backbone)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = base.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    device = torch.device(args.device); base = base.to(device).eval()
    train_loader = utility.make_loader(data_module, alphabet, train, sequence, args.tokens_per_batch)
    validation_loader = utility.make_loader(data_module, alphabet, validation, sequence, args.tokens_per_batch)
    test_loader = utility.make_loader(data_module, alphabet, test, sequence, args.tokens_per_batch)
    # Released reference and threshold use validation records only.
    base_index, base_validation = utility.probabilities(base, validation_loader, None, device, adapter=False, truncate_num=args.truncate_num)
    test_index, base_test = utility.probabilities(base, test_loader, None, device, adapter=False, truncate_num=args.truncate_num)
    baseline = utility.metrics(labels[test_index], base_test, utility.best_f1_threshold(labels[base_index], base_validation))
    model = MFEContactResidualAdapter(base, args.dropout).to(device)
    initial_index, initial_score = probabilities(model, test_loader, offsets, pairs, device, args.truncate_num)
    if not np.array_equal(test_index, initial_index) or not np.allclose(base_test, initial_score, rtol=0., atol=1e-7):
        raise RuntimeError("zero-initialized contact adapter does not reproduce released checkpoint")
    print(json.dumps({"stage": "zero_initialization_equivalence_passed", "fold": args.fold}), flush=True)
    optimizer = torch.optim.AdamW(model.pair_encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer.add_param_group({"params": model.gate.parameters()})
    positive_weight = float(train.size / labels[train].sum() - 1.)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., positive_weight], device=device))
    best, best_state, best_threshold, curve, stale = -np.inf, None, .5, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train(); model.base.rnafm.eval()
        for batch_index, _, _, tokens, _, _ in train_loader:
            record = np.asarray(batch_index, dtype=np.int64)
            logits = model(tokens[:, :args.truncate_num].to(device), record, offsets, pairs)
            target = torch.as_tensor(labels[record], dtype=torch.long, device=device)
            loss = criterion(logits, target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        validation_index, validation_score = probabilities(model, validation_loader, offsets, pairs, device, args.truncate_num)
        threshold = utility.best_f1_threshold(labels[validation_index], validation_score)
        report = utility.metrics(labels[validation_index], validation_score, threshold)
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
    adapter_index, adapter_score = probabilities(model, test_loader, offsets, pairs, device, args.truncate_num)
    if not np.array_equal(test_index, adapter_index):
        raise RuntimeError("test ordering changed during contact-adapter evaluation")
    adapter = utility.metrics(labels[test_index], adapter_score, best_threshold)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"fold": args.fold, "baseline_released_checkpoint": baseline,
                   "mfe_contact_fusion_validation_selected": adapter,
                   "best_validation_aupr": best, "epochs_completed": len(curve),
                   "validation_curve": curve}, handle, indent=2, sort_keys=True); handle.write("\n")
    with gzip.open(args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "label", "baseline_probability", "adapter_probability"))
        writer.writeheader()
        for index, raw, adapted in zip(test_index, base_test, adapter_score):
            writer.writerow({"sample_index": int(index), "label": int(labels[index]),
                             "baseline_probability": float(raw), "adapter_probability": float(adapted)})
    torch.save({"model": best_state, "fold": args.fold, "best_validation_aupr": best}, args.output_dir / "best_adapter.pt")
    manifest = {"schema_version": 1, "experiment": "released_rnafm_sparse_mfe_contact_fusion",
                "fold": args.fold, "seed": args.seed, "validation_fraction": args.validation_fraction,
                "dataset_sha256": sha256(args.dataset), "contact_manifest_sha256": sha256(args.contact_dir / "manifest.json"),
                "checkpoint_sha256": sha256(args.checkpoint), "rnafm_base_sha256": sha256(args.rnafm_base),
                "selection": "validation AUPR", "test_labels_used_for_selection": False,
                "architecture": "frozen released RNA-FM plus residue-pair MFE-contact encoder and zero-initialized gated residual"}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
