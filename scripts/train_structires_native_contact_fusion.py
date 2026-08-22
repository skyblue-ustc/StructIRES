#!/usr/bin/env python3
"""Validation-clean, author-style IRES-RNAFM retraining with MFE contacts.

This is the primary matched training entry point for StructIRES classification.
It starts from the public RNA-FM t12 *pretraining* checkpoint, never from an
IRES classifier checkpoint selected on an upstream test fold.  ``sequence``
reproduces the published IRES-RNAFM topology and training objective; ``contact``
adds a sparse ViennaRNA MFE base-pair encoder through a gated residual.  Both
variants use an identical source fold, validation split, RNA-FM MLM auxiliary
loss, optimisation budget, and untouched official-fold test records.
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
    parser.add_argument("--variant", choices=("sequence", "contact"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path)
    parser.add_argument("--rnafm-base", type=Path, required=True)
    parser.add_argument("--upstream-fm-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--tokens-per-batch", type=int, default=4096)
    parser.add_argument("--truncate-num", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=.5)
    parser.add_argument("--mask-prob", type=float, default=.15)
    parser.add_argument("--classification-loss-weight", type=float, default=2.)
    parser.add_argument("--mlm-loss-weight", type=float, default=1.)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def utility_module():
    source = Path(__file__).with_name("train_structires_release_profile_adapter.py")
    spec = importlib.util.spec_from_file_location("structires_profile_utility", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load shared IRES-RNAFM utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuthorStyleRNAFM(nn.Module):
    """Published topology: t12 BOS (640) -> 40 -> ReLU/dropout -> 2."""
    def __init__(self, rnafm: nn.Module, dropout: float):
        super().__init__()
        self.rnafm = rnafm
        self.fc = nn.Linear(640, 40)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(40, 2)

    def sequence_hidden_with_lm(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.rnafm(tokens, [12])
        residue = output["representations"][12]
        return self.relu(self.fc(residue[:, 0])), residue, output["logits"]

    def forward_with_lm(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, lm_logits = self.sequence_hidden_with_lm(tokens)
        return self.output(self.dropout(sequence)), lm_logits

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_lm(tokens)[0]


class ContactStructIRES(AuthorStyleRNAFM):
    """Author-style RNA-FM with trainable pooling over actual MFE pair edges."""
    def __init__(self, rnafm: nn.Module, dropout: float):
        super().__init__(rnafm, dropout)
        self.contact_encoder = nn.Sequential(
            nn.Linear(4 * 640, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 40)
        )
        self.gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        # Sequence-only behavior at initialization; gradients reach the final
        # contact projection immediately and subsequently its full encoder.
        nn.init.zeros_(self.contact_encoder[-1].weight)
        nn.init.zeros_(self.contact_encoder[-1].bias)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def contact_hidden(self, residue: torch.Tensor, indices: np.ndarray, offsets: np.ndarray,
                       pairs: np.ndarray, tokens: torch.Tensor) -> torch.Tensor:
        values, owners = [], []
        for batch, record in enumerate(indices.tolist()):
            left, right = int(offsets[record]), int(offsets[record + 1])
            edge = pairs[left:right]
            # Cached coordinates are biological 0-based; RNA-FM includes BOS.
            length = int((tokens[batch, 1:-1] != 1).sum().item())
            edge = edge[(edge[:, 0] < length) & (edge[:, 1] < length)]
            if len(edge):
                edge = torch.as_tensor(edge, dtype=torch.long, device=residue.device) + 1
                h_left, h_right = residue[batch, edge[:, 0]], residue[batch, edge[:, 1]]
                values.append(torch.cat((h_left, h_right, h_left * h_right, (h_left - h_right).abs()), dim=1))
                owners.append(torch.full((len(edge),), batch, dtype=torch.long, device=residue.device))
        pooled = torch.zeros((len(indices), 40), dtype=residue.dtype, device=residue.device)
        if not values:
            return pooled
        encoded = self.contact_encoder(torch.cat(values, dim=0))
        owner = torch.cat(owners, dim=0)
        pooled.index_add_(0, owner, encoded)
        count = torch.bincount(owner, minlength=len(indices)).to(residue.dtype).clamp_min(1.).unsqueeze(1)
        return pooled / count

    def forward_with_lm(self, tokens: torch.Tensor, indices: np.ndarray, offsets: np.ndarray,
                        pairs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, residue, lm_logits = self.sequence_hidden_with_lm(tokens)
        structure = self.contact_hidden(residue, indices, offsets, pairs, tokens)
        gate = self.gate(torch.cat((sequence, structure), dim=1))
        return self.output(self.dropout(sequence + gate * structure)), lm_logits

    def forward(self, tokens: torch.Tensor, indices: np.ndarray, offsets: np.ndarray, pairs: np.ndarray) -> torch.Tensor:
        return self.forward_with_lm(tokens, indices, offsets, pairs)[0]


def probability(model: nn.Module, loader, labels: np.ndarray, *, variant: str, offsets, pairs,
                device: torch.device, truncate_num: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); index_all, score = [], []
    with torch.no_grad():
        for index, _, _, tokens, _, _ in loader:
            index = np.asarray(index, dtype=np.int64)
            token_batch = tokens[:, :truncate_num].to(device)
            logits = (model(token_batch) if variant == "sequence" else
                      model(token_batch, index, offsets, pairs))
            index_all.extend(index.tolist())
            score.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(index_all, dtype=np.int64), np.asarray(score, dtype=np.float64)


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if args.variant == "contact" and args.contact_dir is None:
        raise ValueError("--contact-dir is required for --variant contact")
    if not 0. <= args.mask_prob < 1.:
        raise ValueError("--mask-prob must lie in [0, 1)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    utility = utility_module()
    frame = utility.load_fold_rows(args.dataset, args.fold)
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    labels, sequences = frame.label.to_numpy(dtype=np.int64), frame.Sequence.tolist()
    source_train, source_test = frame.type.to_numpy() == "train", frame.type.to_numpy() == "test"
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    train_local, val_local = next(split.split(np.flatnonzero(source_train), labels[source_train]))
    upstream_train = np.flatnonzero(source_train)
    train, validation, test = upstream_train[train_local], upstream_train[val_local], np.flatnonzero(source_test)
    offsets = pairs = None
    contact_manifest_hash = None
    if args.variant == "contact":
        ids = np.load(args.contact_dir / "sequence_ids.npy", allow_pickle=False).astype(str)
        offsets = np.load(args.contact_dir / "pair_offsets.npy", allow_pickle=False, mmap_mode="r")
        pairs = np.load(args.contact_dir / "pairs.npy", allow_pickle=False, mmap_mode="r")
        if list(ids) != frame.ID.astype(str).tolist():
            raise ValueError("canonical contact IDs and source fold differ")
        contact_manifest_hash = sha256(args.contact_dir / "manifest.json")
    data_module, pretrained = utility.load_fm(args.upstream_fm_dir)
    backbone, alphabet = pretrained.rna_fm_t12(str(args.rnafm_base))
    model = (AuthorStyleRNAFM(backbone, args.dropout) if args.variant == "sequence" else
             ContactStructIRES(backbone, args.dropout))
    device = torch.device(args.device); model = model.to(device)
    train_loader = utility.make_loader(data_module, alphabet, train, sequences, args.tokens_per_batch, mask_prob=args.mask_prob)
    validation_loader = utility.make_loader(data_module, alphabet, validation, sequences, args.tokens_per_batch)
    test_loader = utility.make_loader(data_module, alphabet, test, sequences, args.tokens_per_batch)
    positive_weight = float(train.size / labels[train].sum() - 1.)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., positive_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mask_token_id = int(alphabet.tok_to_idx["<mask>"])
    best, best_state, best_threshold, curve, stale = -np.inf, None, .5, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for index, _, _, clean_tokens, masked_tokens, _ in train_loader:
            index = np.asarray(index, dtype=np.int64)
            clean = clean_tokens[:, :args.truncate_num].to(device)
            masked = masked_tokens[:, :args.truncate_num].to(device)
            logits, lm_logits = (model.forward_with_lm(masked) if args.variant == "sequence" else
                                 model.forward_with_lm(masked, index, offsets, pairs))
            target = torch.as_tensor(labels[index], dtype=torch.long, device=device)
            masked_target = torch.full_like(clean, -1)
            mask = masked == mask_token_id
            masked_target[mask] = clean[mask]
            loss = (args.classification_loss_weight * criterion(logits, target) +
                    args.mlm_loss_weight * nn.functional.cross_entropy(lm_logits.transpose(1, 2), masked_target, ignore_index=-1))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        val_index, val_score = probability(model, validation_loader, labels, variant=args.variant, offsets=offsets, pairs=pairs, device=device, truncate_num=args.truncate_num)
        threshold = utility.best_f1_threshold(labels[val_index], val_score)
        report = utility.metrics(labels[val_index], val_score, threshold)
        curve.append({"epoch": epoch, **report}); print(json.dumps({"epoch": epoch, "validation": report}), flush=True)
        if report["aupr"] > best:
            best, best_threshold, stale = report["aupr"], threshold, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                print(json.dumps({"early_stopping": True, "epoch": epoch, "best_validation_aupr": best}), flush=True)
                break
    assert best_state is not None; model.load_state_dict(best_state)
    test_index, test_score = probability(model, test_loader, labels, variant=args.variant, offsets=offsets, pairs=pairs, device=device, truncate_num=args.truncate_num)
    result = utility.metrics(labels[test_index], test_score, best_threshold)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"native_validation_selected": result, "best_validation_aupr": best,
                   "epochs_completed": len(curve), "validation_curve": curve}, handle, indent=2, sort_keys=True); handle.write("\n")
    with gzip.open(args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "label", "probability")); writer.writeheader()
        for index, score in zip(test_index, test_score):
            writer.writerow({"sample_index": int(index), "label": int(labels[index]), "probability": float(score)})
    torch.save({"model": best_state, "fold": args.fold, "variant": args.variant,
                "best_validation_aupr": best}, args.output_dir / "best_model.pt")
    manifest = {"schema_version": 1, "experiment": "native_rnafm_structires_retraining",
                "variant": args.variant, "fold": args.fold, "seed": args.seed,
                "dataset_sha256": sha256(args.dataset), "rnafm_base_sha256": sha256(args.rnafm_base),
                "contact_manifest_sha256": contact_manifest_hash, "validation_fraction": args.validation_fraction,
                "test_labels_used_for_selection": False, "selection": "validation AUPR",
                "architecture": "author-style RNA-FM t12 BOS 640->40->2" if args.variant == "sequence" else
                "author-style RNA-FM t12 BOS 640->40->2 plus sparse MFE contact encoder and gated residual",
                "training_objective": "classification CE*2 plus masked-LM CE*1", "mask_probability": args.mask_prob,
                "epochs_requested": args.epochs, "tokens_per_batch": args.tokens_per_batch}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
