#!/usr/bin/env python3
"""Train a DeepIRES-topology baseline on an IRES-LM native holdout.

The public DeepIRES implementation fixes inputs to 174 nt and uses two
multi-kernel dilated residual blocks, a bidirectional GRU, learned attention,
and a small dense classifier.  This independent PyTorch implementation keeps
that topology while adapting evaluation to one of the ten released IRES-LM
90/10 holdouts.  A validation subset is cut only from the released training
records; the test holdout is never used for checkpoint or threshold selection.
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset


DEEPIRES_SOURCE = "https://github.com/SongLab-at-NUAA/DeepIRES"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=174)
    parser.add_argument("--crop", choices=("left", "center", "right"), default="left")
    parser.add_argument("--padding", choices=("post", "pre"), default="post")
    parser.add_argument(
        "--unmasked-fixed-length",
        action="store_true",
        help="Process all 174 positions, including padding, as in the public Keras model",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss", choices=("weighted_bce", "bce", "focal"), default="weighted_bce")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", help="Enable W&B logging under this project")
    parser.add_argument("--wandb-entity", help="Optional W&B entity; default uses logged-in account")
    parser.add_argument("--wandb-group", default="native-baseline-reproduction")
    parser.add_argument("--wandb-name", help="Optional run name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def utility_module():
    source = Path(__file__).with_name("train_structires_release_profile_adapter.py")
    spec = importlib.util.spec_from_file_location("deepires_native_utility", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load shared native-fold utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crop_sequence(sequence: str, max_length: int, crop: str) -> str:
    sequence = sequence.upper().replace("T", "U")
    if len(sequence) <= max_length:
        return sequence
    if crop == "left":
        return sequence[:max_length]
    if crop == "right":
        return sequence[-max_length:]
    start = (len(sequence) - max_length) // 2
    return sequence[start : start + max_length]


def encode_sequences(
    sequences: list[str], max_length: int = 174, crop: str = "left", padding: str = "post"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode A/C/G/U as the fixed public DeepIRES one-hot vocabulary."""
    mapping = {"A": 1, "C": 2, "G": 3, "U": 4}
    encoded = np.zeros((len(sequences), max_length), dtype=np.int64)
    lengths = np.empty(len(sequences), dtype=np.int64)
    for row, raw in enumerate(sequences):
        sequence = crop_sequence(raw, max_length, crop)
        lengths[row] = max(1, len(sequence))
        values = [mapping.get(base, 0) for base in sequence]
        if padding == "post":
            encoded[row, : len(sequence)] = values
        else:
            encoded[row, max_length - len(sequence) :] = values
    return torch.from_numpy(encoded), torch.from_numpy(lengths)


class MultiKernelResidualBlock(nn.Module):
    """PyTorch equivalent of the public DeepIRES ``ResBlock1``."""

    def __init__(self, input_channels: int, filters: int, dilation: int, dropout: float):
        super().__init__()

        def branch(kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    filters,
                    kernel_size,
                    padding="same",
                    dilation=dilation,
                ),
                nn.BatchNorm1d(filters),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        self.first_2 = branch(2)
        self.first_3 = branch(3)

        def second_branch(kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(
                    filters * 2,
                    filters,
                    kernel_size,
                    padding="same",
                    dilation=dilation,
                ),
                nn.BatchNorm1d(filters),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        self.second_2 = second_branch(2)
        self.second_3 = second_branch(3)
        output_channels = filters * 2
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first_2(values), self.first_3(values)), dim=1)
        second = torch.cat((self.second_2(first), self.second_3(first)), dim=1)
        return self.activation(second + self.shortcut(values))


class AttentionPool(nn.Module):
    """Additive attention matching the public DeepIRES ``AttLayer``."""

    def __init__(self, input_size: int, attention_size: int):
        super().__init__()
        self.projection = nn.Linear(input_size, attention_size)
        self.context = nn.Linear(attention_size, 1, bias=False)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.context(torch.tanh(self.projection(values))).squeeze(-1)
        score = score.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(score, dim=1)
        return (values * weights.unsqueeze(-1)).sum(dim=1)


class DeepIRES(nn.Module):
    """Published DeepIRES topology with padding-aware recurrent pooling."""

    def __init__(self, dropout: float = 0.2, unmasked_fixed_length: bool = False):
        super().__init__()
        self.unmasked_fixed_length = unmasked_fixed_length
        embedding = torch.zeros((5, 4), dtype=torch.float32)
        embedding[1:] = torch.eye(4)
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True, padding_idx=0)
        self.block1 = MultiKernelResidualBlock(4, filters=16, dilation=1, dropout=dropout)
        self.block2 = MultiKernelResidualBlock(32, filters=8, dilation=2, dropout=dropout)
        self.gru = nn.GRU(16, 8, batch_first=True, bidirectional=True)
        self.attention = AttentionPool(16, 8)
        self.classifier = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return the 16-dimensional attended sequence representation."""
        values = self.embedding(tokens).transpose(1, 2)
        values = self.block2(self.block1(values)).transpose(1, 2)
        if self.unmasked_fixed_length:
            values, _ = self.gru(values)
            mask = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
        else:
            packed = pack_padded_sequence(
                values, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed, _ = self.gru(packed)
            values, _ = pad_packed_sequence(
                packed, batch_first=True, total_length=tokens.shape[1]
            )
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
            mask = positions < lengths.unsqueeze(1)
        return self.attention(values, mask)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(tokens, lengths)).squeeze(1)


def binary_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    target_probability = targets * probability + (1.0 - targets) * (1.0 - probability)
    alpha_weight = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return (
        -alpha_weight
        * (1.0 - target_probability).pow(gamma)
        * target_probability.clamp_min(1e-7).log()
    ).mean()


def make_loader(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    subset = TensorDataset(tokens[indices], lengths[indices], labels[indices])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def probability(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for tokens, lengths, _ in loader:
            logits = model(tokens.to(device), lengths.to(device))
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def start_wandb(args: argparse.Namespace, dataset_hash: str):
    """Start optional metric-only W&B tracking without uploading sequence data."""
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("--wandb-project requires the wandb package") from exc
    name = args.wandb_name or f"deepires-native-f{args.fold}-{args.loss}-s{args.seed}"
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=name,
        mode=args.wandb_mode,
        dir=str(args.output_dir.parent),
        config={
            "method": "DeepIRES-topology",
            "fold": args.fold,
            "seed": args.seed,
            "dataset_sha256": dataset_hash,
            "split": "IRES-LM native repeated holdout",
            "validation_fraction": args.validation_fraction,
            "max_length": args.max_length,
            "crop": args.crop,
            "padding": args.padding,
            "unmasked_fixed_length": args.unmasked_fixed_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "loss": args.loss,
            "test_labels_used_for_selection": False,
        },
        tags=["classifier", "baseline", "DeepIRES", f"fold-{args.fold}"],
    )


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must lie in (0, 1)")
    if args.max_length < 3:
        raise ValueError("--max-length must be at least three")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_hash = sha256(args.dataset)
    wandb_run = start_wandb(args, dataset_hash)
    utility = utility_module()
    frame = utility.load_fold_rows(args.dataset, args.fold)
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    sequences = frame.Sequence.astype(str).tolist()
    labels_np = frame.label.to_numpy(dtype=np.int64)
    source_train = frame.type.to_numpy() == "train"
    source_test = frame.type.to_numpy() == "test"

    from sklearn.model_selection import StratifiedShuffleSplit

    upstream_train = np.flatnonzero(source_train)
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=args.validation_fraction, random_state=args.seed
    )
    train_local, validation_local = next(
        split.split(upstream_train, labels_np[upstream_train])
    )
    train = upstream_train[train_local]
    validation = upstream_train[validation_local]
    test = np.flatnonzero(source_test)

    if args.padding == "pre" and not args.unmasked_fixed_length:
        raise ValueError("pre-padding requires --unmasked-fixed-length")
    tokens, lengths = encode_sequences(
        sequences, args.max_length, args.crop, args.padding
    )
    labels = torch.from_numpy(labels_np.astype(np.float32))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = DeepIRES(args.dropout, args.unmasked_fixed_length).to(device)

    validation_loader = make_loader(
        tokens,
        lengths,
        labels,
        validation,
        args.batch_size,
        False,
        args.seed,
        args.num_workers,
    )
    test_loader = None
    if not args.skip_test:
        test_loader = make_loader(
            tokens,
            lengths,
            labels,
            test,
            args.batch_size,
            False,
            args.seed,
            args.num_workers,
        )

    counts = np.bincount(labels_np[train], minlength=2).astype(np.float64)
    positive_weight = float(counts[0] / counts[1])
    weighted_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    plain_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=args.plateau_patience
    )

    best_auc = -np.inf
    best_state = None
    best_threshold = 0.5
    best_validation = None
    stale = 0
    curve: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(
            tokens,
            lengths,
            labels,
            train,
            args.batch_size,
            True,
            args.seed + epoch,
            args.num_workers,
        )
        model.train()
        loss_sum = 0.0
        examples = 0
        for batch_tokens, batch_lengths, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_lengths = batch_lengths.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_tokens, batch_lengths)
            if args.loss == "weighted_bce":
                loss = weighted_criterion(logits, batch_labels)
            elif args.loss == "bce":
                loss = plain_criterion(logits, batch_labels)
            else:
                loss = binary_focal_loss(
                    logits, batch_labels, args.focal_alpha, args.focal_gamma
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            examples += len(batch_labels)

        validation_score = probability(model, validation_loader, device)
        threshold = utility.best_f1_threshold(labels_np[validation], validation_score)
        report = utility.metrics(labels_np[validation], validation_score, threshold)
        scheduler.step(report["auc"])
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(examples, 1),
            "lr": optimizer.param_groups[0]["lr"],
            **report,
        }
        curve.append(row)
        print(json.dumps({"epoch": epoch, "validation": row}), flush=True)
        if wandb_run is not None:
            wandb_run.log({f"validation/{key}": value for key, value in row.items()})
        if report["auc"] > best_auc:
            best_auc = report["auc"]
            best_validation = dict(report)
            best_threshold = threshold
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                print(
                    json.dumps(
                        {
                            "early_stopping": True,
                            "epoch": epoch,
                            "best_validation_auc": best_auc,
                        }
                    ),
                    flush=True,
                )
                break

    if best_state is None or best_validation is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    test_report = None
    test_score = None
    if test_loader is not None:
        test_score = probability(model, test_loader, device)
        test_report = utility.metrics(labels_np[test], test_score, best_threshold)

    args.output_dir.mkdir(parents=True)
    metrics = {
        "best_validation": best_validation,
        "best_validation_auc": best_auc,
        "best_validation_threshold": best_threshold,
        "epochs_completed": len(curve),
        "validation_curve": curve,
    }
    if test_report is not None:
        metrics["native_validation_selected"] = test_report
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with gzip.open(
        args.output_dir / "validation_predictions.csv.gz",
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "label", "probability"))
        writer.writeheader()
        validation_score = probability(model, validation_loader, device)
        for index, score in zip(validation, validation_score):
            writer.writerow(
                {
                    "sample_index": int(index),
                    "label": int(labels_np[index]),
                    "probability": float(score),
                }
            )
    torch.save(
        {"model": best_state, "fold": args.fold, "best_validation_auc": best_auc},
        args.output_dir / "best_model.pt",
    )
    if test_score is not None:
        with gzip.open(
            args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("sample_index", "label", "probability")
            )
            writer.writeheader()
            for index, score in zip(test, test_score):
                writer.writerow(
                    {
                        "sample_index": int(index),
                        "label": int(labels_np[index]),
                        "probability": float(score),
                    }
                )

    manifest = {
        "schema_version": 1,
        "experiment": "deepires_topology_on_ireslm_native_holdout",
        "external_method": "DeepIRES",
        "external_source": DEEPIRES_SOURCE,
        "implementation": "independent PyTorch port of the published topology",
        "fold": args.fold,
        "seed": args.seed,
        "dataset_sha256": dataset_hash,
        "native_split_semantics": "released repeated 90/10 holdout",
        "validation_fraction_within_native_train": args.validation_fraction,
        "test_labels_used_for_selection": False,
        "selection": "maximum validation AUROC; validation-only F1 threshold",
        "max_length": args.max_length,
        "crop": args.crop,
        "padding": args.padding,
        "unmasked_fixed_length": args.unmasked_fixed_length,
        "n_exact_length": int(sum(len(sequence) == args.max_length for sequence in sequences)),
        "n_cropped": int(sum(len(sequence) > args.max_length for sequence in sequences)),
        "n_padded": int(sum(len(sequence) < args.max_length for sequence in sequences)),
        "architecture": (
            "fixed one-hot; multi-kernel residual CNN 16/d1 then 8/d2; "
            "BiGRU 8x2; additive attention; dense 32"
        ),
        "loss": args.loss,
        "positive_weight": positive_weight if args.loss == "weighted_bce" else None,
        "focal_alpha": args.focal_alpha if args.loss == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "test_evaluation_skipped": args.skip_test,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        for key, value in best_validation.items():
            wandb_run.summary[f"best_validation/{key}"] = value
        if test_report is not None:
            for key, value in test_report.items():
                wandb_run.summary[f"test/{key}"] = value
        wandb_run.summary["best_epoch"] = max(curve, key=lambda row: row["auc"])["epoch"]
        wandb_run.finish()
    print(json.dumps({"best_validation": best_validation, "test": test_report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
