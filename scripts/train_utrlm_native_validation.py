#!/usr/bin/env python3
"""Validation-clean IRES-UTRLM training on an IRES-LM native source fold.

The architecture and full-backbone MLM/classification objective follow the
released NMI IRES-UTRLM implementation.  The official outer test records are
never loaded into a DataLoader or used for selection.
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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--upstream-script-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-toks", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=.5)
    parser.add_argument("--mask-prob", type=float, default=.15)
    parser.add_argument("--classification-loss-weight", type=float, default=20.)
    parser.add_argument("--mlm-loss-weight", type=float, default=1.)
    parser.add_argument("--selection-metric", choices=("auc", "aupr"), default="auc")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utility_module():
    source = Path(__file__).with_name("train_structires_release_profile_adapter.py")
    spec = importlib.util.spec_from_file_location("utrlm_native_utility", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_esm(script_dir: Path):
    sys.path.insert(0, str(script_dir))
    data = importlib.import_module("esm.data")
    model = importlib.import_module("esm.model.esm2")
    return data, model


class UTRLMClassifier(nn.Module):
    def __init__(self, esm2_class, alphabet, dropout: float):
        super().__init__()
        self.esm2 = esm2_class(
            num_layers=6, embed_dim=128, attention_heads=16, alphabet=alphabet
        )
        self.fc = nn.Linear(128, 40)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(40, 2)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.esm2(
            tokens,
            [6],
            need_head_weights=False,
            return_contacts=False,
            return_representation=True,
        )
        hidden = result["representations"][6][:, 0]
        logits = self.output(self.dropout(self.relu(self.fc(hidden))))
        return logits, result["logits"]


def make_loader(data_module, alphabet, indices, sequences, labels, mask_prob, batch_toks, *, shuffle_batches, seed):
    subset_sequences = [sequences[int(index)] for index in indices]
    dataset = data_module.FastaBatchedDataset(indices.tolist(), subset_sequences, mask_prob=mask_prob)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    if shuffle_batches:
        random.Random(seed).shuffle(batches)
    return torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches, shuffle=False
    )


def predict(model, loader, labels, device):
    model.eval()
    indices, scores = [], []
    with torch.no_grad():
        for batch_indices, _, _, clean_tokens, _, _ in loader:
            logits, _ = model(clean_tokens.to(device))
            indices.extend(int(index) for index in batch_indices)
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(indices, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    utility = utility_module()
    frame = utility.load_fold_rows(args.dataset, args.fold)
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    sequences = frame.Sequence.str.replace("U", "T", regex=False).tolist()
    labels = frame.label.to_numpy(dtype=np.int64)
    upstream_train = np.flatnonzero(frame.type.to_numpy() == "train")
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=args.validation_fraction, random_state=args.seed
    )
    train_local, validation_local = next(
        split.split(upstream_train, labels[upstream_train])
    )
    train = upstream_train[train_local]
    validation = upstream_train[validation_local]

    data_module, esm2_module = load_esm(args.upstream_script_dir)
    alphabet = data_module.Alphabet(mask_prob=args.mask_prob, standard_toks="AGCT")
    model = UTRLMClassifier(esm2_module.ESM2, alphabet, args.dropout)
    pretrained = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    model.esm2.load_state_dict(pretrained, strict=True)
    device = torch.device(args.device)
    model = model.to(device)
    counts = np.bincount(labels[train], minlength=2).astype(np.float32)
    class_weight = torch.tensor(train.size / (2. * counts), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mask_token = int(alphabet.tok_to_idx["<mask>"])
    validation_loader = make_loader(
        data_module, alphabet, validation, sequences, labels, 0., args.batch_toks,
        shuffle_batches=False, seed=args.seed,
    )
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f"utrlm-clean-f{args.fold}-s{args.seed}",
            group="native-utrlm-structure-fusion-development",
            dir=str(args.output_dir.parent),
            config={
                "fold": args.fold,
                "seed": args.seed,
                "validation_fraction": args.validation_fraction,
                "architecture": "released IRES-UTRLM 6-layer Transformer",
                "test_labels_used": False,
            },
            tags=["classifier", "UTR-LM", "validation-clean", f"fold-{args.fold}"],
        )

    best_value = -np.inf
    best_state = None
    best_metrics = None
    best_threshold = .5
    best_epoch = 0
    stale = 0
    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = make_loader(
            data_module, alphabet, train, sequences, labels, args.mask_prob,
            args.batch_toks, shuffle_batches=True, seed=args.seed + epoch,
        )
        loss_sum = 0.
        examples = 0
        for batch_indices, _, _, clean_tokens, masked_tokens, _ in loader:
            target = torch.as_tensor(
                [labels[int(index)] for index in batch_indices], dtype=torch.long, device=device
            )
            clean_tokens = clean_tokens.to(device)
            masked_tokens = masked_tokens.to(device)
            logits, lm_logits = model(masked_tokens)
            classification_loss = nn.functional.cross_entropy(
                logits, target, weight=class_weight
            )
            masked_target = clean_tokens.clone()
            masked_target[masked_tokens != mask_token] = -1
            mlm_loss = nn.functional.cross_entropy(
                lm_logits.transpose(1, 2), masked_target, ignore_index=-1
            )
            loss = (
                args.classification_loss_weight * classification_loss
                + args.mlm_loss_weight * mlm_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            examples += len(target)
        val_index, val_score = predict(model, validation_loader, labels, device)
        threshold = utility.best_f1_threshold(labels[val_index], val_score)
        report = utility.metrics(labels[val_index], val_score, threshold)
        row = {"epoch": epoch, "train_loss": loss_sum / examples, **report}
        curve.append(row)
        print(json.dumps({"epoch": epoch, "validation": row}), flush=True)
        if wandb_run is not None:
            wandb_run.log({f"validation/{key}": value for key, value in row.items()}, step=epoch)
        selected = report[args.selection_metric]
        if selected > best_value:
            best_value = selected
            best_metrics = dict(report)
            best_threshold = threshold
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    val_index, val_score = predict(model, validation_loader, labels, device)
    args.output_dir.mkdir(parents=True)
    torch.save(
        {"model": best_state, "fold": args.fold, "best_epoch": best_epoch},
        args.output_dir / "best_model.pt",
    )
    with gzip.open(
        args.output_dir / "validation_predictions.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "label", "probability"))
        writer.writeheader()
        for index, score in zip(val_index, val_score):
            writer.writerow(
                {"sample_index": int(index), "label": int(labels[index]), "probability": float(score)}
            )
    metrics = {
        "best_validation": best_metrics,
        "best_validation_threshold": best_threshold,
        "best_epoch": best_epoch,
        "epochs_completed": len(curve),
        "validation_curve": curve,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "validation_clean_ires_utrlm_training",
        "fold": args.fold,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "architecture": "NMI IRES-UTRLM: 6-layer, 16-head, 128-dim Transformer BOS classifier",
        "training_objective": "class-weighted CE*20 plus masked-LM CE*1",
        "selection_metric": args.selection_metric,
        "outer_test_records_scored": False,
        "dataset_sha256": sha256(args.dataset),
        "pretrained_sha256": sha256(args.pretrained),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        for key, value in best_metrics.items():
            wandb_run.summary[f"best_validation/{key}"] = value
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
