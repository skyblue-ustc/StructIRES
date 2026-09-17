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
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=(
            "sequence", "contact", "ensemble", "conditioned", "deepires",
            "graph_sequence", "mfe_graph", "bpp_graph", "deep_structure", "hybrid_structure",
            "token_structure_aux",
            "multilayer_structure_aux",
            "pretrained_deep_adapter",
            "pretrained_refined_deep_adapter",
            "pretrained_separated_deep_adapter",
            "pretrained_gated_deep_adapter",
            "pretrained_sequence_deep_adapter",
            "pretrained_multipool_adapter",
            "pretrained_tunable_structure_adapter",
            "pretrained_penultimate_adapter",
            "pretrained_multiscale_adapter",
            "pretrained_sequence_penultimate_adapter",
            "pretrained_penultimate_warmstart",
            "pretrained_raw_penultimate_warmstart",
            "dual_pretrained_penultimate_warmstart",
            "cross_gated_penultimate_adapter",
        ),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path,
                        help="ViennaRNA ensemble position-profile cache for --variant ensemble")
    parser.add_argument(
        "--profile-permutation-seed",
        type=int,
        help=(
            "Negative control: replace each record's profile with a deterministic "
            "same-length profile from the same fit/validation/test partition"
        ),
    )
    parser.add_argument(
        "--drop-profile-channel",
        choices=("bpp", "mfe", "centroid", "position"),
        help="Set one normalized profile channel to zero for a second-stage input ablation",
    )
    parser.add_argument("--rnafm-base", type=Path, required=True)
    parser.add_argument("--upstream-fm-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Keep the inner train/validation membership fixed while varying training seeds",
    )
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--tokens-per-batch", type=int, default=4096)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Accumulate this many physical batches per optimizer update")
    parser.add_argument("--truncate-num", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float,
                        help="Learning rate for the classifier/structure head; defaults to --lr")
    parser.add_argument("--backbone-lr", type=float,
                        help="Learning rate for trainable RNA-FM parameters; defaults to --lr")
    parser.add_argument(
        "--layerwise-lr-decay",
        type=float,
        default=1.0,
        help="Multiply the RNA-FM learning rate by this factor for each lower layer",
    )
    parser.add_argument("--adapter-lr", type=float,
                        help="Learning rate for trainable pretrained-adapter parameters; defaults to head LR")
    parser.add_argument("--unfreeze-last-layers", type=int, choices=range(13), default=12,
                        help="Number of RNA-FM transformer layers to fine-tune")
    parser.add_argument("--head-only-epochs", type=int, default=0,
                        help="Train only the new head for this many initial epochs")
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("linear", "warmup_cosine"), default="linear")
    parser.add_argument("--warmup-fraction", type=float, default=.05)
    parser.add_argument("--gradient-clip", type=float, default=0.,
                        help="Clip global gradient norm when positive")
    parser.add_argument("--skip-test", action="store_true",
                        help="Development mode: select and report on validation without reading the test fold")
    parser.add_argument("--dropout", type=float, default=.5)
    parser.add_argument(
        "--adapter-init",
        choices=("default", "zero"),
        default="zero",
        help="Initialization for the conditioned structure residual; default uses PyTorch initialization",
    )
    parser.add_argument("--mask-prob", type=float, default=.15)
    parser.add_argument(
        "--classification-input",
        choices=("masked", "clean"),
        default="masked",
        help="Use masked or clean RNA tokens for the classification forward pass",
    )
    parser.add_argument("--classification-loss-weight", type=float, default=2.)
    parser.add_argument("--mlm-loss-weight", type=float, default=1.)
    parser.add_argument(
        "--structure-aux-weight",
        type=float,
        default=0.0,
        help="Weight of per-nucleotide structure-profile reconstruction loss",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="Optional same-fold DeepIRES+structure checkpoint used only for training distillation",
    )
    parser.add_argument(
        "--rnafm-classifier-checkpoint",
        type=Path,
        help="Optional same-fold RNA-FM classifier checkpoint used to initialize hidden features",
    )
    parser.add_argument(
        "--warmstart-checkpoint",
        type=Path,
        help="Optional same-fold joint adapter checkpoint used to initialize a refinement model",
    )
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--distillation-temperature", type=float, default=1.0)
    parser.add_argument("--pairwise-auc-weight", type=float, default=0.0,
                        help="Mix this fraction of pairwise logistic ranking loss into classification CE")
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        choices=("aupr", "auc"),
        default="aupr",
        help="Validation metric used for checkpoint selection and early stopping",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", help="Enable metric-only W&B tracking")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="native-rnafm-development")
    parser.add_argument("--wandb-name")
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
    spec = importlib.util.spec_from_file_location("structires_profile_utility", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load shared IRES-RNAFM utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deepires_teacher_module():
    source = Path(__file__).with_name("train_deepires_structure_fusion.py")
    spec = importlib.util.spec_from_file_location("structires_deepires_teacher", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DeepIRES structure teacher implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE_CHANNEL_INDEX = {"bpp": 0, "mfe": 1, "centroid": 2, "position": 3}


def split_local_profile_permutation(
    sequences: list[str],
    partitions: tuple[np.ndarray, ...],
    *,
    max_length: int,
    seed: int,
) -> np.ndarray:
    """Build a deterministic, split-local, same-length profile derangement.

    Profiles never cross fit/validation/test boundaries. Within every group of
    at least two records having the same truncated length, a randomized cyclic
    shift guarantees that no record retains its own profile. Singleton length
    groups necessarily remain fixed and are reported through the mapping hash
    and changed fraction in the run manifest.
    """
    mapping = np.arange(len(sequences), dtype=np.int64)
    for partition_number, partition in enumerate(partitions):
        groups: dict[int, list[int]] = {}
        for index in np.asarray(partition, dtype=np.int64).tolist():
            groups.setdefault(min(len(sequences[index]), max_length), []).append(index)
        rng = np.random.default_rng(seed + partition_number)
        for members in groups.values():
            if len(members) < 2:
                continue
            randomized = np.asarray(members, dtype=np.int64)[rng.permutation(len(members))]
            mapping[randomized] = np.roll(randomized, 1)
    return mapping


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


class EnsembleStructIRES(AuthorStyleRNAFM):
    """Author-style RNA-FM fused with label-free ViennaRNA ensemble profiles.

    The profile contains per-position ensemble pairing probability, MFE and
    centroid paired states, and relative position.  A validity mask prevents
    padded positions from contributing.  The residual projection is zeroed so
    this model starts exactly at the matched sequence-only topology.
    """
    def __init__(self, rnafm: nn.Module, dropout: float):
        super().__init__(rnafm, dropout)
        self.profile_encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(32, 40, kernel_size=5, padding=2), nn.GELU(),
        )
        self.profile_dropout = nn.Dropout(dropout)
        self.profile_residual = nn.Linear(40, 40)
        self.gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        nn.init.zeros_(self.profile_residual.weight)
        nn.init.zeros_(self.profile_residual.bias)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def ensemble_hidden(self, profile: torch.Tensor) -> torch.Tensor:
        values, valid = profile[:, :, :4], profile[:, :, 4]
        local = self.profile_encoder(values.transpose(1, 2)).transpose(1, 2)
        pooled = (local * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1.)
        return self.profile_residual(self.profile_dropout(pooled))

    def forward_with_lm(self, tokens: torch.Tensor, profile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, lm_logits = self.sequence_hidden_with_lm(tokens)
        structure = self.ensemble_hidden(profile)
        gate = self.gate(torch.cat((sequence, structure), dim=1))
        return self.output(self.dropout(sequence + gate * structure)), lm_logits

    def forward(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        return self.forward_with_lm(tokens, profile)[0]


class ConditionedStructIRES(AuthorStyleRNAFM):
    """Pool contextual RNA-FM residues with structure-conditioned attention."""

    def __init__(self, rnafm: nn.Module, dropout: float, *, zero_start: bool):
        super().__init__(rnafm, dropout)
        self.profile_encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.GELU(),
        )
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(640), nn.Linear(640, 128), nn.GELU(),
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(192, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.attention = nn.Linear(128, 1)
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 40), nn.GELU(), nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        if zero_start:
            # Preserve the fitted RNA-FM path at initialization.  Keeping this
            # selectable lets the validation-only screen compare it fairly to
            # the original PyTorch initialization without touching test data.
            nn.init.zeros_(self.pool_projection[1].weight)
            nn.init.zeros_(self.pool_projection[1].bias)
            nn.init.zeros_(self.gate[0].weight)
            nn.init.zeros_(self.gate[0].bias)

    def conditioned_hidden(self, residue: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        # RNA-FM token 0 is BOS.  The cache contains biological positions only;
        # its fifth channel masks sequence padding and any truncated tail.
        length = min(residue.shape[1] - 1, profile.shape[1])
        residue = residue[:, 1:1 + length]
        profile = profile[:, :length]
        values, mask = profile[:, :, :4], profile[:, :, 4]
        structural = self.profile_encoder(values.transpose(1, 2)).transpose(1, 2)
        local = self.local_fusion(torch.cat((self.residue_projection(residue), structural), dim=2))
        valid = mask.bool()
        score = self.attention(local).squeeze(-1).masked_fill(~valid, -torch.inf)
        attended = (local * torch.softmax(score, dim=1).unsqueeze(-1)).sum(dim=1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (local * mask.unsqueeze(-1)).sum(dim=1) / denominator
        maximum = local.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(dim=1)
        return self.pool_projection(torch.cat((attended, mean, maximum), dim=1))

    def forward_with_lm(self, tokens: torch.Tensor, profile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, residue, lm_logits = self.sequence_hidden_with_lm(tokens)
        structure = self.conditioned_hidden(residue, profile)
        gate = self.gate(torch.cat((sequence, structure), dim=1))
        return self.output(self.dropout(sequence + gate * structure)), lm_logits

    def forward(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        return self.forward_with_lm(tokens, profile)[0]


class ResidueGraphStructIRES(AuthorStyleRNAFM):
    """Zero-start residue adapter with optional probability-weighted BPP edges.

    ``use_bpp=False`` retains exactly the same trainable architecture but
    suppresses every structural edge.  It is therefore the matched control
    for testing whether BPP connectivity, rather than the local pooling head
    or its parameter count, contributes to recognition.
    """

    def __init__(self, rnafm: nn.Module, dropout: float, *, use_bpp: bool):
        super().__init__(rnafm, dropout)
        self.use_bpp = use_bpp
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(640), nn.Linear(640, 64), nn.GELU(),
        )
        self.pair_message = nn.Sequential(
            nn.Linear(4 * 64, 96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, 64),
        )
        self.message_gate = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())
        self.block1 = ContextualMultiKernelBlock(64, 64, dilation=1, dropout=dropout)
        self.block2 = ContextualMultiKernelBlock(128, 32, dilation=2, dropout=dropout)
        self.gru = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.attention_projection = nn.Linear(64, 32)
        self.attention_context = nn.Linear(32, 1, bias=False)
        self.graph_projection = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, 40), nn.GELU(), nn.Dropout(dropout),
        )
        self.graph_residual = nn.Linear(40, 40)
        self.output_gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        # Both graph variants begin numerically identical to the published
        # BOS-only RNA-FM classifier.  The adapter is introduced gradually.
        nn.init.zeros_(self.graph_residual.weight)
        nn.init.zeros_(self.graph_residual.bias)
        nn.init.zeros_(self.output_gate[0].weight)
        nn.init.zeros_(self.output_gate[0].bias)

    @staticmethod
    def biological_mask(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        non_padding = (tokens != 1).sum(dim=1)
        has_eos = (tokens == 2).any(dim=1).to(non_padding.dtype)
        lengths = (non_padding - 1 - has_eos).clamp(min=1, max=tokens.shape[1] - 1)
        position = torch.arange(tokens.shape[1] - 1, device=tokens.device).unsqueeze(0)
        return lengths, position < lengths.unsqueeze(1)

    def add_pair_messages(
        self,
        node: torch.Tensor,
        indices: np.ndarray,
        offsets: np.ndarray,
        pairs: np.ndarray,
        pair_probabilities: np.ndarray,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        message = torch.zeros_like(node)
        normalizer = torch.zeros(node.shape[:2], dtype=node.dtype, device=node.device)
        for batch, record in enumerate(indices.tolist()):
            left, right = int(offsets[record]), int(offsets[record + 1])
            edge = np.asarray(pairs[left:right], dtype=np.int64)
            probability = np.asarray(pair_probabilities[left:right], dtype=np.float32)
            length = int(lengths[batch].item())
            keep = ((edge[:, 0] < length) & (edge[:, 1] < length)) if len(edge) else np.zeros(0, dtype=bool)
            edge, probability = edge[keep], probability[keep]
            if not len(edge):
                continue
            edge_tensor = torch.as_tensor(edge, dtype=torch.long, device=node.device)
            weight = torch.as_tensor(probability, dtype=node.dtype, device=node.device).unsqueeze(1)
            left_node = node[batch, edge_tensor[:, 0]]
            right_node = node[batch, edge_tensor[:, 1]]
            encoded = self.pair_message(torch.cat(
                (left_node, right_node, left_node * right_node, (left_node - right_node).abs()), dim=1
            )) * weight
            message[batch].index_add_(0, edge_tensor[:, 0], encoded)
            message[batch].index_add_(0, edge_tensor[:, 1], encoded)
            normalizer[batch].index_add_(0, edge_tensor[:, 0], weight.squeeze(1))
            normalizer[batch].index_add_(0, edge_tensor[:, 1], weight.squeeze(1))
        message = message / normalizer.clamp_min(1e-6).unsqueeze(-1)
        gate = self.message_gate(torch.cat((node, message), dim=2))
        return node + gate * message

    def graph_hidden(
        self,
        residue: torch.Tensor,
        tokens: torch.Tensor,
        indices: np.ndarray,
        offsets: np.ndarray | None,
        pairs: np.ndarray | None,
        pair_probabilities: np.ndarray | None,
    ) -> torch.Tensor:
        lengths, valid = self.biological_mask(tokens)
        node = self.residue_projection(residue[:, 1:]) * valid.unsqueeze(-1)
        if self.use_bpp:
            assert offsets is not None and pairs is not None and pair_probabilities is not None
            node = self.add_pair_messages(
                node, indices, offsets, pairs, pair_probabilities, lengths
            ) * valid.unsqueeze(-1)
        values = self.block2(self.block1(node.transpose(1, 2))).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            values, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.gru(packed)
        values, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=node.shape[1]
        )
        attention = self.attention_context(
            torch.tanh(self.attention_projection(values))
        ).squeeze(-1).masked_fill(~valid, -torch.inf)
        pooled = (values * torch.softmax(attention, dim=1).unsqueeze(-1)).sum(dim=1)
        return self.graph_projection(pooled)

    def forward_with_lm(
        self,
        tokens: torch.Tensor,
        indices: np.ndarray,
        offsets: np.ndarray | None = None,
        pairs: np.ndarray | None = None,
        pair_probabilities: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, residue, lm_logits = self.sequence_hidden_with_lm(tokens)
        graph = self.graph_hidden(
            residue, tokens, indices, offsets, pairs, pair_probabilities
        )
        residual = self.graph_residual(graph)
        gate = self.output_gate(torch.cat((sequence, residual), dim=1))
        return self.output(self.dropout(sequence + gate * residual)), lm_logits

    def forward(
        self,
        tokens: torch.Tensor,
        indices: np.ndarray,
        offsets: np.ndarray | None = None,
        pairs: np.ndarray | None = None,
        pair_probabilities: np.ndarray | None = None,
    ) -> torch.Tensor:
        return self.forward_with_lm(
            tokens, indices, offsets, pairs, pair_probabilities
        )[0]


class ContextualMultiKernelBlock(nn.Module):
    """DeepIRES-style two-stage, two-kernel residual block."""

    def __init__(self, input_channels: int, filters: int, dilation: int, dropout: float):
        super().__init__()

        def branch(channels: int, kernel: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(channels, filters, kernel, padding="same", dilation=dilation),
                nn.BatchNorm1d(filters),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        self.first2 = branch(input_channels, 2)
        self.first3 = branch(input_channels, 3)
        self.second2 = branch(filters * 2, 2)
        self.second3 = branch(filters * 2, 3)
        output_channels = filters * 2
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, 1)
        )
        self.relu = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first2(values), self.first3(values)), dim=1)
        second = torch.cat((self.second2(first), self.second3(first)), dim=1)
        return self.relu(second + self.shortcut(values))


class DeepStructureAdapter(AuthorStyleRNAFM):
    """DeepIRES-style local sequence/structure branch added to RNA-FM residually."""

    def __init__(self, rnafm: nn.Module, dropout: float, *, zero_start: bool):
        super().__init__(rnafm, dropout)
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(640), nn.Linear(640, 64), nn.GELU()
        )
        self.structure_projection = nn.Sequential(nn.Linear(4, 16), nn.GELU())
        self.block1 = ContextualMultiKernelBlock(80, 64, dilation=1, dropout=dropout)
        self.block2 = ContextualMultiKernelBlock(128, 32, dilation=2, dropout=dropout)
        self.gru = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.attention_projection = nn.Linear(64, 32)
        self.attention_context = nn.Linear(32, 1, bias=False)
        self.adapter_projection = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, 40), nn.GELU(), nn.Dropout(dropout)
        )
        self.adapter_residual = nn.Linear(40, 40)
        self.output_gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        if zero_start:
            nn.init.zeros_(self.adapter_residual.weight)
            nn.init.zeros_(self.adapter_residual.bias)
            nn.init.zeros_(self.output_gate[0].weight)
            nn.init.zeros_(self.output_gate[0].bias)

    def deep_structure_hidden(
        self, residue: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        length = min(residue.shape[1] - 1, profile.shape[1])
        biological = residue[:, 1:1 + length]
        profile = profile[:, :length]
        valid = profile[:, :, 4].bool()
        lengths = valid.sum(dim=1).clamp_min(1).to(torch.long)
        sequence = self.residue_projection(biological)
        structure = self.structure_projection(profile[:, :, :4])
        values = torch.cat((sequence, structure), dim=2) * valid.unsqueeze(-1)
        values = self.block2(self.block1(values.transpose(1, 2))).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            values, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.gru(packed)
        values, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=length
        )
        attention = self.attention_context(
            torch.tanh(self.attention_projection(values))
        ).squeeze(-1).masked_fill(~valid, -torch.inf)
        pooled = (values * torch.softmax(attention, dim=1).unsqueeze(-1)).sum(dim=1)
        return self.adapter_projection(pooled)

    def forward_with_lm(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, residue, lm_logits = self.sequence_hidden_with_lm(tokens)
        adapter = self.adapter_residual(self.deep_structure_hidden(residue, profile))
        gate = self.output_gate(torch.cat((sequence, adapter), dim=1))
        return self.output(self.dropout(sequence + gate * adapter)), lm_logits

    def forward(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        return self.forward_with_lm(tokens, profile)[0]


class HybridDeepIRESStructureAdapter(AuthorStyleRNAFM):
    """Raw nucleotide/structure DeepIRES branch injected into RNA-FM residually."""

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        nucleotide_ids: list[int],
        zero_start: bool,
    ):
        super().__init__(rnafm, dropout)
        self.register_buffer(
            "nucleotide_ids", torch.tensor(nucleotide_ids, dtype=torch.long)
        )
        self.block1 = ContextualMultiKernelBlock(8, 16, dilation=1, dropout=dropout)
        self.block2 = ContextualMultiKernelBlock(32, 8, dilation=2, dropout=dropout)
        self.gru = nn.GRU(16, 8, batch_first=True, bidirectional=True)
        self.attention_projection = nn.Linear(16, 8)
        self.attention_context = nn.Linear(8, 1, bias=False)
        self.adapter_projection = nn.Sequential(
            nn.Linear(16, 40), nn.GELU(), nn.Dropout(dropout)
        )
        self.adapter_residual = nn.Linear(40, 40)
        self.output_gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        if zero_start:
            nn.init.zeros_(self.adapter_residual.weight)
            nn.init.zeros_(self.adapter_residual.bias)
            nn.init.zeros_(self.output_gate[0].weight)
            nn.init.zeros_(self.output_gate[0].bias)

    def hybrid_hidden(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        length = min(tokens.shape[1] - 1, profile.shape[1])
        biological_tokens = tokens[:, 1:1 + length]
        profile = profile[:, :length]
        valid = profile[:, :, 4].bool()
        lengths = valid.sum(dim=1).clamp_min(1).to(torch.long)
        one_hot = (
            biological_tokens.unsqueeze(-1) == self.nucleotide_ids.view(1, 1, -1)
        ).to(profile.dtype)
        values = torch.cat((one_hot, profile[:, :, :4]), dim=2)
        values = values * valid.unsqueeze(-1)
        values = self.block2(self.block1(values.transpose(1, 2))).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            values, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.gru(packed)
        values, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=length
        )
        attention = self.attention_context(
            torch.tanh(self.attention_projection(values))
        ).squeeze(-1).masked_fill(~valid, -torch.inf)
        pooled = (values * torch.softmax(attention, dim=1).unsqueeze(-1)).sum(dim=1)
        return self.adapter_projection(pooled)

    def forward_with_lm(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, lm_logits = self.sequence_hidden_with_lm(tokens)
        adapter = self.adapter_residual(self.hybrid_hidden(tokens, profile))
        gate = self.output_gate(torch.cat((sequence, adapter), dim=1))
        return self.output(self.dropout(sequence + gate * adapter)), lm_logits

    def forward(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        return self.forward_with_lm(tokens, profile)[0]


class TokenStructureAuxAdapter(AuthorStyleRNAFM):
    """Inject per-base structure into RNA-FM residues with auxiliary supervision.

    Unlike a score-level ensemble, this module conditions every contextual
    nucleotide representation on the aligned ViennaRNA profile before pooling.
    A decoder predicts the first three structure channels from RNA-FM residues
    alone, encouraging the fine-tuned backbone to retain structure information.
    """

    def __init__(self, rnafm: nn.Module, dropout: float, *, zero_start: bool):
        super().__init__(rnafm, dropout)
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(640), nn.Linear(640, 128), nn.GELU(),
        )
        self.profile_encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attention = nn.Linear(128, 1)
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 40), nn.GELU(), nn.Dropout(dropout),
        )
        self.adapter_residual = nn.Linear(40, 40)
        self.output_gate = nn.Sequential(nn.Linear(80, 40), nn.Sigmoid())
        self.structure_decoder = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3),
        )
        if zero_start:
            nn.init.zeros_(self.adapter_residual.weight)
            nn.init.zeros_(self.adapter_residual.bias)
            nn.init.zeros_(self.output_gate[0].weight)
            nn.init.zeros_(self.output_gate[0].bias)

    def token_structure_hidden(
        self, residue: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = min(residue.shape[1] - 1, profile.shape[1])
        biological = residue[:, 1:1 + length]
        profile = profile[:, :length]
        valid = profile[:, :, 4].bool()
        sequence = self.residue_projection(biological)
        structure = self.profile_encoder(profile[:, :, :4].transpose(1, 2)).transpose(1, 2)
        gate = self.fusion_gate(torch.cat((sequence, structure), dim=2))
        fused = sequence + gate * structure
        fused = self.context(fused, src_key_padding_mask=~valid)
        fused = fused * valid.unsqueeze(-1)
        score = self.attention(fused).squeeze(-1).masked_fill(~valid, -torch.inf)
        attended = (fused * torch.softmax(score, dim=1).unsqueeze(-1)).sum(dim=1)
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1).to(fused.dtype)
        mean = fused.sum(dim=1) / denominator
        maximum = fused.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(dim=1)
        pooled = self.pool_projection(torch.cat((attended, mean, maximum), dim=1))
        structure_prediction = self.structure_decoder(sequence)
        return pooled, structure_prediction, valid

    def forward_with_aux(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, residue, lm_logits = self.sequence_hidden_with_lm(tokens)
        adapter, structure_prediction, valid = self.token_structure_hidden(residue, profile)
        adapter = self.adapter_residual(adapter)
        gate = self.output_gate(torch.cat((sequence, adapter), dim=1))
        logits = self.output(self.dropout(sequence + gate * adapter))
        return logits, lm_logits, structure_prediction, valid

    def forward_with_lm(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, lm_logits, _, _ = self.forward_with_aux(tokens, profile)
        return logits, lm_logits

    def forward(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        return self.forward_with_lm(tokens, profile)[0]


class MultiLayerTokenStructureAuxAdapter(TokenStructureAuxAdapter):
    """Token structure adapter over a learned mixture of RNA-FM depths."""

    def __init__(self, rnafm: nn.Module, dropout: float, *, zero_start: bool):
        super().__init__(rnafm, dropout, zero_start=zero_start)
        self.representation_layers = (6, 8, 10, 12)
        self.layer_logits = nn.Parameter(torch.zeros(len(self.representation_layers)))

    def sequence_hidden_with_lm(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.rnafm(tokens, list(self.representation_layers))
        weights = torch.softmax(self.layer_logits, dim=0)
        residue = sum(
            weight * output["representations"][layer]
            for weight, layer in zip(weights, self.representation_layers)
        )
        return self.relu(self.fc(residue[:, 0])), residue, output["logits"]


class PretrainedDeepStructureRNAFM(AuthorStyleRNAFM):
    """RNA-FM with a pretrained local sequence/structure representation adapter.

    The adapter's original classifier is never called.  Its 16-dimensional
    sequence/structure representation is joined with the RNA-FM hidden state
    and learned by one new classification head, so inference is not logit or
    probability fusion.
    """

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(rnafm, dropout)
        self.adapter = adapter
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "deepires_token_lookup",
            torch.tensor(deepires_token_lookup, dtype=torch.long),
        )
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(56), nn.Linear(56, 80), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(80, 40), nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.adapter.eval()
        return self

    def adapter_components(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        biological = tokens[:, 1:]
        non_padding = (tokens != 1).sum(dim=1)
        has_eos = (tokens == 2).any(dim=1).to(non_padding.dtype)
        lengths = (non_padding - 1 - has_eos).clamp(min=1, max=174)
        local_tokens = torch.zeros((len(tokens), 174), dtype=torch.long, device=tokens.device)
        copy_length = min(174, biological.shape[1])
        local_tokens[:, :copy_length] = self.deepires_token_lookup[
            biological[:, :copy_length]
        ]
        with torch.no_grad():
            sequence = self.adapter.sequence.encode(local_tokens, lengths)
            structure = self.adapter.structure(profile[:, :174, :4], lengths)
            gate = torch.sigmoid(self.adapter.gate(torch.cat((sequence, structure), dim=1)))
            return sequence, gate * self.adapter.structure_delta(structure)

    def adapter_hidden(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        sequence, structure = self.adapter_components(tokens, profile)
        return sequence + structure

    def forward_with_lm(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, lm_logits = self.sequence_hidden_with_lm(tokens)
        adapter = self.adapter_hidden(tokens, profile)
        joint = self.joint_projection(torch.cat((sequence, adapter), dim=1))
        return self.output(self.dropout(joint)), lm_logits

    def forward(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        return self.forward_with_lm(tokens, profile)[0]


class PretrainedSeparatedDeepStructureRNAFM(PretrainedDeepStructureRNAFM):
    """Expose pretrained local sequence and structural residuals separately.

    A single joint head receives RNA-FM, local sequence, and gated structural
    hidden states.  Keeping the two 16-D local components separate prevents
    their forced addition from suppressing structure-specific evidence while
    never evaluating the pretrained adapter's scalar classifier.
    """

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(72), nn.Linear(72, 96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, 40), nn.GELU(),
        )

    def forward_with_lm(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, lm_logits = self.sequence_hidden_with_lm(tokens)
        local_sequence, structure = self.adapter_components(tokens, profile)
        joint = self.joint_projection(
            torch.cat((sequence, local_sequence, structure), dim=1)
        )
        return self.output(self.dropout(joint)), lm_logits


class WarmstartedStructureRefinementRNAFM(PretrainedDeepStructureRNAFM):
    """Refine a strong joint model with an explicit structural correction.

    The full RNA-FM/local-adapter classifier is loaded from a same-fold
    development checkpoint. A zero-initialized residual maps only the gated
    structural component into the RNA-FM hidden state, while a cross-branch
    gate controls its magnitude. Before training, the refinement path is
    exactly zero and predictions therefore reproduce the source joint model.
    The adapter's original scalar classifier is never evaluated.
    """

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        self.refinement_structure_norm = nn.LayerNorm(16)
        self.refinement_residual = nn.Sequential(
            nn.Linear(16, 40),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(40, 40),
        )
        self.refinement_gate = nn.Sequential(nn.Linear(72, 40), nn.Sigmoid())
        nn.init.zeros_(self.refinement_residual[-1].weight)
        nn.init.zeros_(self.refinement_residual[-1].bias)
        nn.init.zeros_(self.refinement_gate[0].weight)
        nn.init.constant_(self.refinement_gate[0].bias, -2.0)

    def forward_with_lm(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, lm_logits = self.sequence_hidden_with_lm(tokens)
        local_sequence, structure = self.adapter_components(tokens, profile)
        normalized_structure = self.refinement_structure_norm(structure)
        gate = self.refinement_gate(
            torch.cat((sequence, local_sequence, normalized_structure), dim=1)
        )
        conditioned = sequence + gate * self.refinement_residual(normalized_structure)
        adapter = local_sequence + structure
        joint = self.joint_projection(torch.cat((conditioned, adapter), dim=1))
        return self.output(self.dropout(joint)), lm_logits


class PretrainedGatedDeepStructureRNAFM(PretrainedDeepStructureRNAFM):
    """Use the pretrained 16-D structure state to modulate RNA-FM features.

    This is feature-level conditioning inside one predictor.  The frozen
    structure module's scalar classifier is never called: its 16-D hidden
    state produces a bounded gate and residual over the 40-D RNA-FM state,
    after which one joint head consumes the conditioned state and structure
    state.  Branch-specific normalization prevents scale alone from deciding
    which representation dominates.
    """

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        self.sequence_norm = nn.LayerNorm(40)
        self.structure_norm = nn.LayerNorm(16)
        self.structure_residual = nn.Sequential(
            nn.Linear(16, 40), nn.GELU(), nn.Dropout(dropout), nn.Linear(40, 40)
        )
        self.structure_gate = nn.Sequential(nn.Linear(56, 40), nn.Sigmoid())
        self.conditioned_norm = nn.LayerNorm(40)
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(56), nn.Linear(56, 80), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(80, 40), nn.GELU(),
        )

    def forward_with_lm(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, lm_logits = self.sequence_hidden_with_lm(tokens)
        structure = self.adapter_hidden(tokens, profile)
        sequence = self.sequence_norm(sequence)
        structure = self.structure_norm(structure)
        gate = self.structure_gate(torch.cat((sequence, structure), dim=1))
        conditioned = self.conditioned_norm(
            sequence + gate * self.structure_residual(structure)
        )
        joint = self.joint_projection(torch.cat((conditioned, structure), dim=1))
        return self.output(self.dropout(joint)), lm_logits


class PretrainedPenultimateStructureRNAFM(PretrainedDeepStructureRNAFM):
    """RNA-FM joined to the teacher's structure-aware penultimate features.

    The frozen adapter's scalar classifier output is deliberately excluded.
    Instead, the fused sequence/structure encoding is passed through the
    pretrained classifier up to (but not including) its final output layer,
    yielding a 32-dimensional task-informed representation for one new joint
    RNA-FM classifier. Inference therefore remains feature-level integration,
    not logit or probability fusion.
    """

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(72), nn.Linear(72, 96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, 40), nn.GELU(),
        )

    def adapter_hidden(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        fused = super().adapter_hidden(tokens, profile)
        with torch.no_grad():
            # DeepIRES classifier is Linear->BatchNorm->ReLU->Dropout->Linear.
            # Exclude its final scalar output layer and retain the 32-D hidden.
            return self.adapter.sequence.classifier[:-1](fused)


class TunablePretrainedStructureRNAFM(PretrainedDeepStructureRNAFM):
    """Joint adapter with only the pretrained structural gate trainable."""

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        for module in (self.adapter.gate, self.adapter.structure_delta):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.adapter.gate.train(mode)
        self.adapter.structure_delta.train(mode)
        return self

    def adapter_hidden(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        biological = tokens[:, 1:]
        non_padding = (tokens != 1).sum(dim=1)
        has_eos = (tokens == 2).any(dim=1).to(non_padding.dtype)
        lengths = (non_padding - 1 - has_eos).clamp(min=1, max=174)
        local_tokens = torch.zeros((len(tokens), 174), dtype=torch.long, device=tokens.device)
        copy_length = min(174, biological.shape[1])
        local_tokens[:, :copy_length] = self.deepires_token_lookup[
            biological[:, :copy_length]
        ]
        with torch.no_grad():
            sequence = self.adapter.sequence.encode(local_tokens, lengths)
            structure = self.adapter.structure(profile[:, :174, :4], lengths)
        gate = torch.sigmoid(self.adapter.gate(torch.cat((sequence, structure), dim=1)))
        return sequence + gate * self.adapter.structure_delta(structure)


class PretrainedMultiPoolStructureRNAFM(PretrainedDeepStructureRNAFM):
    """Structure adapter over complementary RNA-FM BOS and mean-pooled states."""

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        self.mean_fc = nn.Linear(640, 40)
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(96), nn.Linear(96, 112), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(112, 40), nn.GELU(),
        )

    def sequence_hidden_with_lm(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.rnafm(tokens, [12])
        residue = output["representations"][12]
        valid = (tokens != 0) & (tokens != 1) & (tokens != 2)
        mean = (residue * valid.unsqueeze(-1)).sum(dim=1)
        mean = mean / valid.sum(dim=1, keepdim=True).clamp_min(1)
        bos = self.relu(self.fc(residue[:, 0]))
        pooled = self.relu(self.mean_fc(mean))
        return torch.cat((bos, pooled), dim=1), residue, output["logits"]


class PretrainedSequenceDeepRNAFM(PretrainedDeepStructureRNAFM):
    """Matched 16-D adapter ablation that excludes structure information."""

    def adapter_hidden(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        del profile
        biological = tokens[:, 1:]
        non_padding = (tokens != 1).sum(dim=1)
        has_eos = (tokens == 2).any(dim=1).to(non_padding.dtype)
        lengths = (non_padding - 1 - has_eos).clamp(min=1, max=174)
        local_tokens = torch.zeros((len(tokens), 174), dtype=torch.long, device=tokens.device)
        copy_length = min(174, biological.shape[1])
        local_tokens[:, :copy_length] = self.deepires_token_lookup[
            biological[:, :copy_length]
        ]
        with torch.no_grad():
            return self.adapter.sequence.encode(local_tokens, lengths)


class PretrainedMultiscaleStructureRNAFM(PretrainedDeepStructureRNAFM):
    """RNA-FM joined to complementary 16-D and 32-D structure states.

    The 16-D state is the pretrained adapter's gated sequence/structure
    representation, while the 32-D state is its task-informed penultimate
    projection.  Both are hidden features from one frozen structure module;
    its scalar classifier output is never evaluated.  A single joint head
    therefore learns the final prediction from RNA-FM and multiscale
    structure representations without output-score fusion.
    """

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(88), nn.Linear(88, 112), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(112, 40), nn.GELU(),
        )

    def adapter_hidden(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        fused = super().adapter_hidden(tokens, profile)
        with torch.no_grad():
            penultimate = self.adapter.sequence.classifier[:-1](fused)
            return torch.cat((fused, penultimate), dim=1)


class PretrainedSequencePenultimateRNAFM(PretrainedPenultimateStructureRNAFM):
    """Matched penultimate adapter ablation without the structure branch."""

    def adapter_hidden(self, tokens: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        del profile
        biological = tokens[:, 1:]
        non_padding = (tokens != 1).sum(dim=1)
        has_eos = (tokens == 2).any(dim=1).to(non_padding.dtype)
        lengths = (non_padding - 1 - has_eos).clamp(min=1, max=174)
        local_tokens = torch.zeros((len(tokens), 174), dtype=torch.long, device=tokens.device)
        copy_length = min(174, biological.shape[1])
        local_tokens[:, :copy_length] = self.deepires_token_lookup[
            biological[:, :copy_length]
        ]
        with torch.no_grad():
            sequence = self.adapter.sequence.encode(local_tokens, lengths)
            return self.adapter.sequence.classifier[:-1](sequence)


class PretrainedPenultimateWarmstartRNAFM(PretrainedPenultimateStructureRNAFM):
    """One joint RNA-FM/structure head initialized to preserve the teacher.

    The classifier consumes one concatenated 72-D representation. Its
    structure columns are initialized from the pretrained adapter's final
    linear classifier, while the RNA-FM columns start at zero and are learned
    jointly. No separate teacher score is computed or combined at inference.
    """

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        final = self.adapter.sequence.classifier[-1]
        if not isinstance(final, nn.Linear) or final.in_features != 32 or final.out_features != 1:
            raise ValueError("expected a scalar DeepIRES classifier over 32-D hidden features")
        self.joint_projection = nn.Identity()
        self.output = nn.Linear(72, 2)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()
            # softmax([-z/2, z/2])[:, 1] is exactly sigmoid(z).
            self.output.weight[0, 40:].copy_(-0.5 * final.weight[0])
            self.output.weight[1, 40:].copy_(0.5 * final.weight[0])
            self.output.bias[0].copy_(-0.5 * final.bias[0])
            self.output.bias[1].copy_(0.5 * final.bias[0])


class PretrainedRawPenultimateWarmstartRNAFM(PretrainedPenultimateStructureRNAFM):
    """Direct linear probe over frozen raw RNA-FM and structure features."""

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        final = self.adapter.sequence.classifier[-1]
        if not isinstance(final, nn.Linear) or final.in_features != 32 or final.out_features != 1:
            raise ValueError("expected a scalar DeepIRES classifier over 32-D hidden features")
        self.joint_projection = nn.Identity()
        self.output = nn.Linear(672, 2)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()
            self.output.weight[0, 640:].copy_(-0.5 * final.weight[0])
            self.output.weight[1, 640:].copy_(0.5 * final.weight[0])
            self.output.bias[0].copy_(-0.5 * final.bias[0])
            self.output.bias[1].copy_(0.5 * final.bias[0])

    def forward_with_lm(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.rnafm(tokens, [12])
        structure = self.adapter_hidden(tokens, profile)
        joint = torch.cat((output["representations"][12][:, 0], structure), dim=1)
        return self.output(self.dropout(joint)), output["logits"]


class CrossGatedPenultimateRNAFM(PretrainedPenultimateStructureRNAFM):
    """Inject task-trained RNA-FM features into a structure representation.

    The frozen structure-aware 32-D penultimate representation remains the
    main state.  A task-trained RNA-FM 40-D state proposes a bounded residual,
    whose gate is conditioned jointly on both branches.  The residual output
    projection is zero initialized, and the frozen final classifier is copied
    from the structure teacher.  Consequently epoch 0 exactly reproduces the
    teacher, while subsequent predictions are produced by one internally
    structure-conditioned path rather than by combining model logits.
    """

    def __init__(
        self,
        rnafm: nn.Module,
        dropout: float,
        *,
        adapter: nn.Module,
        deepires_token_lookup: list[int],
    ):
        super().__init__(
            rnafm,
            dropout,
            adapter=adapter,
            deepires_token_lookup=deepires_token_lookup,
        )
        self.sequence_residual = nn.Sequential(
            nn.LayerNorm(40),
            nn.Linear(40, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
        )
        self.cross_gate = nn.Sequential(nn.Linear(72, 32), nn.Sigmoid())
        nn.init.zeros_(self.sequence_residual[-1].weight)
        nn.init.zeros_(self.sequence_residual[-1].bias)
        nn.init.zeros_(self.cross_gate[0].weight)
        nn.init.zeros_(self.cross_gate[0].bias)

        final = self.adapter.sequence.classifier[-1]
        if not isinstance(final, nn.Linear) or final.in_features != 32 or final.out_features != 1:
            raise ValueError("expected a scalar DeepIRES classifier over 32-D hidden features")
        self.joint_projection = nn.Identity()
        self.output = nn.Linear(32, 2)
        with torch.no_grad():
            self.output.weight[0].copy_(-0.5 * final.weight[0])
            self.output.weight[1].copy_(0.5 * final.weight[0])
            self.output.bias[0].copy_(-0.5 * final.bias[0])
            self.output.bias[1].copy_(0.5 * final.bias[0])
        for parameter in self.output.parameters():
            parameter.requires_grad_(False)

    def forward_with_lm(
        self, tokens: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _, lm_logits = self.sequence_hidden_with_lm(tokens)
        structure = self.adapter_hidden(tokens, profile)
        gate = self.cross_gate(torch.cat((sequence, structure), dim=1))
        fused = structure + gate * self.sequence_residual(sequence)
        return self.output(fused), lm_logits


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use stored statistics for a singleton training batch.

    Token-budget batching can legitimately produce a final batch containing
    one long sequence. Standard BatchNorm1d rejects a two-dimensional [1, C]
    tensor in training mode even though its running statistics are available.
    """

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values_per_channel = values.numel() // values.shape[1]
        if self.training and values_per_channel == 1:
            return torch.nn.functional.batch_norm(
                values,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(values)


class ContextualDeepIRES(AuthorStyleRNAFM):
    """RNA-FM residue embeddings followed by a DeepIRES-style local head."""

    def __init__(self, rnafm: nn.Module, dropout: float):
        super().__init__(rnafm, dropout)
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(640), nn.Linear(640, 64), nn.GELU()
        )
        self.block1 = ContextualMultiKernelBlock(64, 64, dilation=1, dropout=dropout)
        self.block2 = ContextualMultiKernelBlock(128, 32, dilation=2, dropout=dropout)
        self.gru = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.attention_projection = nn.Linear(64, 32)
        self.attention_context = nn.Linear(32, 1, bias=False)
        self.deepires_fc = nn.Sequential(
            nn.Linear(64, 40), SafeBatchNorm1d(40), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward_with_lm(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.rnafm(tokens, [12])
        residue = output["representations"][12]
        # Count biological tokens after BOS.  Subtract EOS only when it is
        # still present after truncation; padding uses RNA-FM token id 1.
        non_padding = (tokens != 1).sum(dim=1)
        has_eos = (tokens == 2).any(dim=1).to(non_padding.dtype)
        lengths = (non_padding - 1 - has_eos).clamp(min=1, max=tokens.shape[1] - 1)
        biological = residue[:, 1:]
        position = torch.arange(biological.shape[1], device=tokens.device).unsqueeze(0)
        valid = position < lengths.unsqueeze(1)
        values = self.residue_projection(biological) * valid.unsqueeze(-1)
        values = self.block2(self.block1(values.transpose(1, 2))).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            values, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.gru(packed)
        values, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=biological.shape[1]
        )
        attention = self.attention_context(
            torch.tanh(self.attention_projection(values))
        ).squeeze(-1)
        attention = attention.masked_fill(~valid, -torch.inf)
        pooled = (values * torch.softmax(attention, dim=1).unsqueeze(-1)).sum(dim=1)
        hidden = self.deepires_fc(pooled)
        return self.output(hidden), output["logits"]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_lm(tokens)[0]


def probability(model: nn.Module, loader, labels: np.ndarray, *, variant: str, offsets, pairs,
                pair_probabilities, profiles,
                device: torch.device, truncate_num: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); index_all, score = [], []
    with torch.no_grad():
        for index, _, _, tokens, _, _ in loader:
            index = np.asarray(index, dtype=np.int64)
            token_batch = tokens[:, :truncate_num].to(device)
            if variant in {"sequence", "deepires"}:
                logits = model(token_batch)
            elif variant == "contact":
                logits = model(token_batch, index, offsets, pairs)
            elif variant in {"graph_sequence", "mfe_graph", "bpp_graph"}:
                logits = model(
                    token_batch, index, offsets, pairs, pair_probabilities
                )
            else:
                logits = model(token_batch, torch.from_numpy(np.asarray(profiles[index])).to(device))
            index_all.extend(index.tolist())
            score.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(index_all, dtype=np.int64), np.asarray(score, dtype=np.float64)


def shuffled_train_loader(utility, data_module, alphabet, indices: np.ndarray, sequences: list[str],
                          tokens_per_batch: int, mask_prob: float, seed: int):
    """Match upstream DistributedSampler behaviour over length-aware batches."""
    source = utility.make_loader(data_module, alphabet, indices, sequences, tokens_per_batch, mask_prob=mask_prob)
    batch_order = list(source.batch_sampler)
    random.Random(seed).shuffle(batch_order)
    return torch.utils.data.DataLoader(source.dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batch_order)


def configure_trainable_parameters(model: AuthorStyleRNAFM, last_layers: int):
    """Separate RNA-FM, trainable adapter, and newly initialized parameters."""
    for parameter in model.rnafm.parameters():
        parameter.requires_grad = False
    if last_layers >= 12:
        for parameter in model.rnafm.parameters():
            parameter.requires_grad = True
    elif last_layers:
        for index in range(12 - last_layers, 12):
            for parameter in model.rnafm.layers[index].parameters():
                parameter.requires_grad = True
        if hasattr(model.rnafm, "emb_layer_norm_after"):
            for parameter in model.rnafm.emb_layer_norm_after.parameters():
                parameter.requires_grad = True
    backbone = [parameter for parameter in model.rnafm.parameters() if parameter.requires_grad]
    adapter = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("adapter.") and parameter.requires_grad
    ]
    head = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("rnafm.")
        and not name.startswith("adapter.")
        and parameter.requires_grad
    ]
    return backbone, adapter, head


def layerwise_backbone_groups(
    model: AuthorStyleRNAFM,
    parameters: list[torch.nn.Parameter],
    base_lr: float,
    decay: float,
) -> list[dict]:
    """Build discriminative RNA-FM learning-rate groups from bottom to top."""
    if not parameters:
        return []
    if decay == 1.0:
        return [{"params": parameters, "lr": base_lr, "base_lr": base_lr,
                 "role": "backbone"}]

    layers = list(model.rnafm.layers)
    top_depth = len(layers) + 1
    depth_by_id = {id(parameter): 0 for parameter in model.rnafm.parameters()}
    for depth, layer in enumerate(layers, start=1):
        for parameter in layer.parameters():
            depth_by_id[id(parameter)] = depth
    if hasattr(model.rnafm, "emb_layer_norm_after"):
        for parameter in model.rnafm.emb_layer_norm_after.parameters():
            depth_by_id[id(parameter)] = top_depth

    grouped: dict[int, list[torch.nn.Parameter]] = {}
    for parameter in parameters:
        grouped.setdefault(depth_by_id[id(parameter)], []).append(parameter)
    output = []
    for depth in sorted(grouped):
        learning_rate = base_lr * decay ** (top_depth - depth)
        output.append({
            "params": grouped[depth],
            "lr": learning_rate,
            "base_lr": learning_rate,
            "role": f"backbone_depth_{depth}",
        })
    return output


def warmup_cosine_factor(step: int, total_steps: int, warmup_fraction: float) -> float:
    """Return a per-update learning-rate multiplier in (0, 1]."""
    progress = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    if warmup_fraction > 0.0 and progress < warmup_fraction:
        return max(1e-3, progress / warmup_fraction)
    decay = (progress - warmup_fraction) / max(1e-12, 1.0 - warmup_fraction)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, decay))))


def pairwise_auc_loss(logits: torch.Tensor, target: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Pairwise logistic surrogate encouraging positives to outrank negatives."""
    score = logits[:, 1] - logits[:, 0]
    positive = score[target == 1]
    negative = score[target == 0]
    if not len(positive) or not len(negative):
        return logits.sum() * 0.0
    difference = positive.unsqueeze(1) - negative.unsqueeze(0)
    return nn.functional.softplus(margin - difference).mean()


def start_wandb(args: argparse.Namespace, dataset_hash: str):
    """Start optional W&B tracking without uploading sequences or checkpoints."""
    if not args.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_name or f"rnafm-{args.variant}-f{args.fold}-s{args.seed}",
        mode=args.wandb_mode,
        dir=str(args.output_dir.parent),
        config={
            "variant": args.variant,
            "fold": args.fold,
            "seed": args.seed,
            "split_seed": args.seed if args.split_seed is None else args.split_seed,
            "dataset_sha256": dataset_hash,
            "validation_fraction": args.validation_fraction,
            "epochs": args.epochs,
            "tokens_per_batch": args.tokens_per_batch,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "head_lr": args.head_lr or args.lr,
            "backbone_lr": args.backbone_lr or args.lr,
            "layerwise_lr_decay": args.layerwise_lr_decay,
            "adapter_lr": args.adapter_lr or args.head_lr or args.lr,
            "unfreeze_last_layers": args.unfreeze_last_layers,
            "head_only_epochs": args.head_only_epochs,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "adapter_init": args.adapter_init,
            "mask_probability": args.mask_prob,
            "classification_input": args.classification_input,
            "classification_loss_weight": args.classification_loss_weight,
            "mlm_loss_weight": args.mlm_loss_weight,
            "structure_aux_weight": args.structure_aux_weight,
            "distillation_weight": args.distillation_weight,
            "distillation_temperature": args.distillation_temperature,
            "pairwise_auc_weight": args.pairwise_auc_weight,
            "pairwise_margin": args.pairwise_margin,
            "profile_permutation_seed": args.profile_permutation_seed,
            "drop_profile_channel": args.drop_profile_channel,
            "selection_metric": args.selection_metric,
            "test_labels_used_for_selection": False,
        },
        tags=["classifier", "RNA-FM", args.variant, f"fold-{args.fold}"],
    )


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if args.variant in {"contact", "mfe_graph", "bpp_graph"} and args.contact_dir is None:
        raise ValueError(f"--contact-dir is required for --variant {args.variant}")
    structure_aux_variants = {"token_structure_aux", "multilayer_structure_aux"}
    profile_variants = {
        "ensemble", "conditioned", "deep_structure", "hybrid_structure",
        "pretrained_deep_adapter", "pretrained_sequence_deep_adapter",
        "pretrained_refined_deep_adapter",
        "pretrained_separated_deep_adapter",
        "pretrained_gated_deep_adapter",
        "pretrained_multipool_adapter",
        "pretrained_tunable_structure_adapter",
        "pretrained_penultimate_adapter",
        "pretrained_multiscale_adapter",
        "pretrained_sequence_penultimate_adapter",
        "pretrained_penultimate_warmstart",
        "pretrained_raw_penultimate_warmstart",
        "dual_pretrained_penultimate_warmstart",
        "cross_gated_penultimate_adapter",
        *structure_aux_variants,
    }
    if args.variant in profile_variants and args.profile_dir is None:
        raise ValueError(f"--profile-dir is required for --variant {args.variant}")
    pretrained_adapter_variants = {
        "pretrained_deep_adapter", "pretrained_sequence_deep_adapter",
        "pretrained_refined_deep_adapter",
        "pretrained_separated_deep_adapter",
        "pretrained_gated_deep_adapter",
        "pretrained_multipool_adapter",
        "pretrained_tunable_structure_adapter",
        "pretrained_penultimate_adapter",
        "pretrained_multiscale_adapter",
        "pretrained_sequence_penultimate_adapter",
        "pretrained_penultimate_warmstart",
        "pretrained_raw_penultimate_warmstart",
        "dual_pretrained_penultimate_warmstart",
        "cross_gated_penultimate_adapter",
    }
    if args.variant in pretrained_adapter_variants and args.teacher_checkpoint is None:
        raise ValueError(f"{args.variant} requires --teacher-checkpoint")
    if (
        args.variant in {
            "dual_pretrained_penultimate_warmstart",
            "cross_gated_penultimate_adapter",
        }
        and args.rnafm_classifier_checkpoint is None
    ):
        raise ValueError(
            f"{args.variant} requires "
            "--rnafm-classifier-checkpoint"
        )
    if not 0. <= args.mask_prob < 1.:
        raise ValueError("--mask-prob must lie in [0, 1)")
    if not 0. <= args.warmup_fraction < 1.:
        raise ValueError("--warmup-fraction must lie in [0, 1)")
    if args.head_only_epochs < 0 or args.head_only_epochs >= args.epochs:
        raise ValueError("--head-only-epochs must lie in [0, epochs)")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if not 0.0 <= args.pairwise_auc_weight <= 1.0:
        raise ValueError("--pairwise-auc-weight must lie in [0, 1]")
    if args.structure_aux_weight < 0.0:
        raise ValueError("--structure-aux-weight must be non-negative")
    if args.structure_aux_weight and args.variant not in structure_aux_variants:
        raise ValueError("--structure-aux-weight requires a structure-aux adapter")
    if args.distillation_weight < 0.0:
        raise ValueError("--distillation-weight must be non-negative")
    if args.distillation_temperature <= 0.0:
        raise ValueError("--distillation-temperature must be positive")
    if not 0.0 < args.layerwise_lr_decay <= 1.0:
        raise ValueError("--layerwise-lr-decay must lie in (0, 1]")
    if args.distillation_weight and args.variant not in profile_variants:
        raise ValueError("distillation requires a profile-based structure variant")
    if args.distillation_weight and args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required when distillation is enabled")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dataset_hash = sha256(args.dataset)
    wandb_run = start_wandb(args, dataset_hash)
    utility = utility_module()
    frame = utility.load_fold_rows(args.dataset, args.fold)
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    labels, sequences = frame.label.to_numpy(dtype=np.int64), frame.Sequence.tolist()
    source_train, source_test = frame.type.to_numpy() == "train", frame.type.to_numpy() == "test"
    from sklearn.model_selection import StratifiedShuffleSplit
    split_seed = args.seed if args.split_seed is None else args.split_seed
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=args.validation_fraction, random_state=split_seed
    )
    train_local, val_local = next(split.split(np.flatnonzero(source_train), labels[source_train]))
    upstream_train = np.flatnonzero(source_train)
    train, validation, test = upstream_train[train_local], upstream_train[val_local], np.flatnonzero(source_test)
    offsets = pairs = pair_probabilities = profiles = None
    contact_manifest_hash = None
    if args.variant in {"contact", "mfe_graph"}:
        ids = np.load(args.contact_dir / "sequence_ids.npy", allow_pickle=False).astype(str)
        offsets = np.load(args.contact_dir / "pair_offsets.npy", allow_pickle=False, mmap_mode="r")
        pairs = np.load(args.contact_dir / "pairs.npy", allow_pickle=False, mmap_mode="r")
        if list(ids) != frame.ID.astype(str).tolist():
            raise ValueError("canonical contact IDs and source fold differ")
        if args.variant == "mfe_graph":
            pair_probabilities = np.ones(len(pairs), dtype=np.float32)
        contact_manifest_hash = sha256(args.contact_dir / "manifest.json")
    elif args.variant == "bpp_graph":
        ids = np.load(args.contact_dir / "sequence_ids.npy", allow_pickle=False).astype(str)
        offsets = np.load(args.contact_dir / "pair_offsets.npy", allow_pickle=False, mmap_mode="r")
        pairs = np.load(args.contact_dir / "pairs.npy", allow_pickle=False, mmap_mode="r")
        pair_probabilities = np.load(
            args.contact_dir / "pair_probabilities.npy", allow_pickle=False, mmap_mode="r"
        )
        if list(ids) != frame.ID.astype(str).tolist():
            raise ValueError("canonical BPP contact IDs and source fold differ")
        if len(pairs) != len(pair_probabilities) or len(offsets) != len(frame) + 1:
            raise ValueError("invalid BPP contact cache shapes")
        contact_manifest_hash = sha256(args.contact_dir / "manifest.json")
    profile_manifest_hash = None
    profile_mapping_hash = None
    profile_mapping_fraction_changed = 0.0
    if args.variant in profile_variants:
        ids = np.load(args.profile_dir / "sequence_ids.npy", allow_pickle=False).astype(str)
        raw_profiles = np.load(args.profile_dir / "profiles.npy", mmap_mode="r")
        if list(ids) != frame.ID.astype(str).tolist() or raw_profiles.shape != (len(frame), args.truncate_num, 5):
            raise ValueError("ensemble profile IDs or shape differ from canonical source fold")
        profile_index = np.arange(len(frame), dtype=np.int64)
        if args.profile_permutation_seed is not None:
            profile_index = split_local_profile_permutation(
                sequences,
                (train, validation, test),
                max_length=args.truncate_num,
                seed=args.profile_permutation_seed,
            )
            profile_mapping_hash = hashlib.sha256(
                profile_index.astype("<i8", copy=False).tobytes()
            ).hexdigest()
            profile_mapping_fraction_changed = float(
                np.mean(profile_index != np.arange(len(frame), dtype=np.int64))
            )
        # Normalize only the structural value channels on the training records;
        # preserve the label-free valid-position mask.
        value = np.asarray(raw_profiles[profile_index[train]], dtype=np.float64)
        mask = value[:, :, 4:5]
        count = float(mask.sum())
        mean = (value[:, :, :4] * mask).sum(axis=(0, 1)) / count
        std = np.sqrt(np.maximum(((value[:, :, :4] ** 2) * mask).sum(axis=(0, 1)) / count - mean ** 2, 1e-6))
        class NormalizedProfiles:
            def __getitem__(self, index):
                source_index = profile_index[np.asarray(index, dtype=np.int64)]
                output = np.asarray(raw_profiles[source_index], dtype=np.float32).copy()
                output[..., :4] = (output[..., :4] - mean.astype(np.float32)) / std.astype(np.float32)
                output[..., :4] *= output[..., 4:5]
                if args.drop_profile_channel is not None:
                    output[..., PROFILE_CHANNEL_INDEX[args.drop_profile_channel]] = 0.0
                return output
        profiles = NormalizedProfiles()
        profile_manifest_hash = sha256(args.profile_dir / "manifest.json")
    data_module, pretrained = utility.load_fm(args.upstream_fm_dir)
    backbone, alphabet = pretrained.rna_fm_t12(str(args.rnafm_base))
    pretrained_structure_adapter = None
    pretrained_adapter_checkpoint_hash = None
    if args.variant in pretrained_adapter_variants:
        adapter_module = deepires_teacher_module()
        deepires = adapter_module.load_deepires_module()
        pretrained_structure_adapter = adapter_module.GatedDeepIRESStructure(
            deepires, dropout=.2
        )
        adapter_payload = torch.load(
            args.teacher_checkpoint, map_location="cpu", weights_only=True
        )
        pretrained_structure_adapter.load_state_dict(adapter_payload["model"], strict=True)
        pretrained_adapter_checkpoint_hash = sha256(args.teacher_checkpoint)
    deepires_lookup = [0] * (max(alphabet.tok_to_idx.values()) + 1)
    for encoded, base in enumerate("ACGU", start=1):
        deepires_lookup[alphabet.tok_to_idx[base]] = encoded
    model = (AuthorStyleRNAFM(backbone, args.dropout) if args.variant == "sequence" else
             ContactStructIRES(backbone, args.dropout) if args.variant == "contact" else
             EnsembleStructIRES(backbone, args.dropout) if args.variant == "ensemble" else
             ConditionedStructIRES(
                 backbone, args.dropout, zero_start=args.adapter_init == "zero"
             ) if args.variant == "conditioned" else
             DeepStructureAdapter(
                 backbone, args.dropout, zero_start=args.adapter_init == "zero"
             ) if args.variant == "deep_structure" else
             HybridDeepIRESStructureAdapter(
                 backbone,
                 args.dropout,
                 nucleotide_ids=[alphabet.tok_to_idx[base] for base in "AGCU"],
                 zero_start=args.adapter_init == "zero",
             ) if args.variant == "hybrid_structure" else
             TokenStructureAuxAdapter(
                 backbone, args.dropout, zero_start=args.adapter_init == "zero"
             ) if args.variant == "token_structure_aux" else
             MultiLayerTokenStructureAuxAdapter(
                 backbone, args.dropout, zero_start=args.adapter_init == "zero"
             ) if args.variant == "multilayer_structure_aux" else
             PretrainedDeepStructureRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_deep_adapter" else
             WarmstartedStructureRefinementRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_refined_deep_adapter" else
             PretrainedSeparatedDeepStructureRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_separated_deep_adapter" else
             PretrainedGatedDeepStructureRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_gated_deep_adapter" else
             PretrainedSequenceDeepRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_sequence_deep_adapter" else
             PretrainedMultiPoolStructureRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_multipool_adapter" else
             TunablePretrainedStructureRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_tunable_structure_adapter" else
             PretrainedPenultimateStructureRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_penultimate_adapter" else
             PretrainedMultiscaleStructureRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_multiscale_adapter" else
             PretrainedSequencePenultimateRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_sequence_penultimate_adapter" else
             PretrainedPenultimateWarmstartRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant in {
                 "pretrained_penultimate_warmstart",
                 "dual_pretrained_penultimate_warmstart",
             } else
             CrossGatedPenultimateRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "cross_gated_penultimate_adapter" else
             PretrainedRawPenultimateWarmstartRNAFM(
                 backbone,
                 args.dropout,
                 adapter=pretrained_structure_adapter,
                 deepires_token_lookup=deepires_lookup,
             ) if args.variant == "pretrained_raw_penultimate_warmstart" else
             ContextualDeepIRES(backbone, args.dropout) if args.variant == "deepires" else
             ResidueGraphStructIRES(
                 backbone, args.dropout, use_bpp=args.variant in {"mfe_graph", "bpp_graph"}
             ))
    warmstart_checkpoint_hash = None
    if args.variant == "pretrained_refined_deep_adapter":
        if args.warmstart_checkpoint is None:
            raise ValueError(
                "pretrained_refined_deep_adapter requires --warmstart-checkpoint"
            )
        warmstart_payload = torch.load(
            args.warmstart_checkpoint, map_location="cpu", weights_only=True
        )
        warmstart_state = warmstart_payload["model"]
        model_state = model.state_dict()
        refinement_prefixes = (
            "refinement_structure_norm.",
            "refinement_residual.",
            "refinement_gate.",
        )
        required = {
            name for name in model_state if not name.startswith(refinement_prefixes)
        }
        transferred = {
            name: value
            for name, value in warmstart_state.items()
            if name in model_state and model_state[name].shape == value.shape
        }
        missing = sorted(required - set(transferred))
        if missing:
            raise ValueError(
                "joint warm-start checkpoint is missing compatible keys: "
                f"{missing[:5]}"
            )
        model_state.update(transferred)
        model.load_state_dict(model_state, strict=True)
        for name, parameter in model.named_parameters():
            if not name.startswith("refinement_"):
                parameter.requires_grad_(False)
        warmstart_checkpoint_hash = sha256(args.warmstart_checkpoint)

    rnafm_classifier_checkpoint_hash = None
    if args.variant in {
        "dual_pretrained_penultimate_warmstart",
        "cross_gated_penultimate_adapter",
    }:
        sequence_payload = torch.load(
            args.rnafm_classifier_checkpoint, map_location="cpu", weights_only=True
        )
        sequence_state = sequence_payload["model"]
        model_state = model.state_dict()
        transferred = {
            name: value
            for name, value in sequence_state.items()
            if (name.startswith("rnafm.") or name.startswith("fc."))
            and name in model_state
            and model_state[name].shape == value.shape
        }
        expected = {
            name for name in model_state
            if name.startswith("rnafm.") or name.startswith("fc.")
        }
        if set(transferred) != expected:
            missing = sorted(expected - set(transferred))
            raise ValueError(
                f"RNA-FM classifier checkpoint is missing compatible keys: {missing[:5]}"
            )
        model_state.update(transferred)
        model.load_state_dict(model_state, strict=True)
        # Preserve the already task-trained 40-D RNA-FM representation while
        # the new joint classifier learns cross-branch weights.
        for parameter in model.fc.parameters():
            parameter.requires_grad_(False)
        rnafm_classifier_checkpoint_hash = sha256(
            args.rnafm_classifier_checkpoint
        )
    device = torch.device(args.device); model = model.to(device)
    teacher = None
    teacher_token_lookup = None
    teacher_checkpoint_hash = None
    if args.distillation_weight:
        teacher_module = deepires_teacher_module()
        deepires = teacher_module.load_deepires_module()
        teacher = teacher_module.GatedDeepIRESStructure(deepires, dropout=.2)
        teacher_payload = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True)
        teacher.load_state_dict(teacher_payload["model"], strict=True)
        teacher.eval().to(device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        vocabulary_size = max(alphabet.tok_to_idx.values()) + 1
        teacher_token_lookup = torch.zeros(vocabulary_size, dtype=torch.long, device=device)
        for encoded, base in enumerate("ACGU", start=1):
            teacher_token_lookup[alphabet.tok_to_idx[base]] = encoded
        teacher_checkpoint_hash = sha256(args.teacher_checkpoint)
    validation_loader = utility.make_loader(data_module, alphabet, validation, sequences, args.tokens_per_batch)
    test_loader = (None if args.skip_test else
                   utility.make_loader(data_module, alphabet, test, sequences, args.tokens_per_batch))
    counts = np.bincount(labels[train], minlength=2).astype(np.float32)
    # Numerically identical class ratio to the upstream implementation, while
    # retaining its explicit n/(2*n_class) definition for provenance.
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(train.size / (2. * counts), device=device))
    backbone_parameters, adapter_parameters, head_parameters = configure_trainable_parameters(
        model, args.unfreeze_last_layers
    )
    head_lr = args.lr if args.head_lr is None else args.head_lr
    backbone_lr = args.lr if args.backbone_lr is None else args.backbone_lr
    adapter_lr = head_lr if args.adapter_lr is None else args.adapter_lr
    parameter_groups = [{"params": head_parameters, "lr": head_lr, "base_lr": head_lr, "role": "head"}]
    if adapter_parameters:
        parameter_groups.append({"params": adapter_parameters, "lr": adapter_lr,
                                 "base_lr": adapter_lr, "role": "adapter"})
    parameter_groups.extend(layerwise_backbone_groups(
        model, backbone_parameters, backbone_lr, args.layerwise_lr_decay
    ))
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(parameter_groups, weight_decay=args.weight_decay if args.optimizer == "adamw" else 0.)
    scheduler = (torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1., end_factor=.5, total_iters=30
    ) if args.lr_schedule == "linear" else None)
    batches_per_epoch = len(shuffled_train_loader(
        utility, data_module, alphabet, train, sequences, args.tokens_per_batch,
        args.mask_prob, args.seed + 1,
    ))
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    total_updates = args.epochs * updates_per_epoch
    update = 0
    mask_token_id = int(alphabet.tok_to_idx["<mask>"])
    best, best_state, best_threshold, best_validation, curve, stale = -np.inf, None, .5, None, [], 0
    warmstart_variants = {
        "pretrained_refined_deep_adapter",
        "pretrained_penultimate_warmstart",
        "pretrained_raw_penultimate_warmstart",
        "dual_pretrained_penultimate_warmstart",
        "cross_gated_penultimate_adapter",
    }
    if args.variant in warmstart_variants:
        initial_index, initial_score = probability(
            model, validation_loader, labels, variant=args.variant, offsets=offsets,
            pairs=pairs, pair_probabilities=pair_probabilities, profiles=profiles,
            device=device, truncate_num=args.truncate_num,
        )
        initial_threshold = utility.best_f1_threshold(
            labels[initial_index], initial_score
        )
        initial_report = utility.metrics(
            labels[initial_index], initial_score, initial_threshold
        )
        curve.append({"epoch": 0, **initial_report})
        print(json.dumps({"epoch": 0, "validation": initial_report}), flush=True)
        if wandb_run is not None:
            wandb_run.log(
                {f"validation/{key}": value for key, value in initial_report.items()},
                step=0,
            )
        best = initial_report[args.selection_metric]
        best_threshold = initial_threshold
        best_validation = dict(initial_report)
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loader = shuffled_train_loader(utility, data_module, alphabet, train, sequences,
                                             args.tokens_per_batch, args.mask_prob, args.seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        for batch_number, (index, _, _, clean_tokens, masked_tokens, _) in enumerate(train_loader, start=1):
            if args.lr_schedule == "warmup_cosine" and (batch_number - 1) % args.gradient_accumulation_steps == 0:
                factor = warmup_cosine_factor(update, total_updates, args.warmup_fraction)
                for group in optimizer.param_groups:
                    active = not (
                        group["role"].startswith("backbone")
                        and epoch <= args.head_only_epochs
                    )
                    group["lr"] = group["base_lr"] * factor if active else 0.0
            index = np.asarray(index, dtype=np.int64)
            clean = clean_tokens[:, :args.truncate_num].to(device)
            masked = masked_tokens[:, :args.truncate_num].to(device)
            classification_tokens = clean if args.classification_input == "clean" else masked
            if args.variant in {"sequence", "deepires"}:
                logits, lm_logits = model.forward_with_lm(classification_tokens)
            elif args.variant == "contact":
                logits, lm_logits = model.forward_with_lm(classification_tokens, index, offsets, pairs)
            elif args.variant in {"graph_sequence", "mfe_graph", "bpp_graph"}:
                logits, lm_logits = model.forward_with_lm(
                    classification_tokens, index, offsets, pairs, pair_probabilities
                )
            elif args.variant in structure_aux_variants:
                profile = torch.from_numpy(np.asarray(profiles[index])).to(device)
                logits, lm_logits, structure_prediction, structure_valid = model.forward_with_aux(
                    classification_tokens, profile
                )
            else:
                profile = torch.from_numpy(np.asarray(profiles[index])).to(device)
                logits, lm_logits = model.forward_with_lm(classification_tokens, profile)
            target = torch.as_tensor(labels[index], dtype=torch.long, device=device)
            masked_target = torch.full_like(clean, -1)
            mask = masked == mask_token_id
            masked_target[mask] = clean[mask]
            if args.classification_input == "clean" and args.mlm_loss_weight:
                lm_logits = model.rnafm(masked, [12])["logits"]
            classification_loss = criterion(logits, target)
            if args.pairwise_auc_weight:
                ranking_loss = pairwise_auc_loss(logits, target, args.pairwise_margin)
                classification_loss = (
                    (1.0 - args.pairwise_auc_weight) * classification_loss
                    + args.pairwise_auc_weight * ranking_loss
                )
            loss = args.classification_loss_weight * classification_loss
            if args.mlm_loss_weight:
                loss = loss + args.mlm_loss_weight * nn.functional.cross_entropy(
                    lm_logits.transpose(1, 2), masked_target, ignore_index=-1
                )
            if args.variant in structure_aux_variants and args.structure_aux_weight:
                structure_target = profile[:, :structure_prediction.shape[1], :3]
                auxiliary = nn.functional.smooth_l1_loss(
                    structure_prediction[structure_valid],
                    structure_target[structure_valid],
                )
                loss = loss + args.structure_aux_weight * auxiliary
            if teacher is not None:
                assert teacher_token_lookup is not None
                biological = clean[:, 1:]
                non_padding = (clean != 1).sum(dim=1)
                has_eos = (clean == 2).any(dim=1).to(non_padding.dtype)
                teacher_lengths = (non_padding - 1 - has_eos).clamp(min=1, max=174)
                teacher_tokens = torch.zeros(
                    (len(clean), 174), dtype=torch.long, device=device
                )
                copy_length = min(174, biological.shape[1])
                teacher_tokens[:, :copy_length] = teacher_token_lookup[
                    biological[:, :copy_length]
                ]
                with torch.no_grad():
                    teacher_logits, _ = teacher(
                        teacher_tokens,
                        teacher_lengths,
                        profile[:, :174, :4],
                    )
                temperature = args.distillation_temperature
                teacher_target = torch.sigmoid(teacher_logits / temperature)
                student_margin = (logits[:, 1] - logits[:, 0]) / temperature
                distillation = nn.functional.binary_cross_entropy_with_logits(
                    student_margin, teacher_target
                ) * temperature ** 2
                loss = loss + args.distillation_weight * distillation
            (loss / args.gradient_accumulation_steps).backward()
            should_update = (batch_number % args.gradient_accumulation_steps == 0 or
                             batch_number == len(train_loader))
            if should_update:
                if args.gradient_clip > 0.:
                    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); update += 1
        if scheduler is not None:
            scheduler.step()
        val_index, val_score = probability(
            model, validation_loader, labels, variant=args.variant, offsets=offsets,
            pairs=pairs, pair_probabilities=pair_probabilities, profiles=profiles,
            device=device, truncate_num=args.truncate_num,
        )
        threshold = utility.best_f1_threshold(labels[val_index], val_score)
        report = utility.metrics(labels[val_index], val_score, threshold)
        curve.append({"epoch": epoch, **report}); print(json.dumps({"epoch": epoch, "validation": report}), flush=True)
        if wandb_run is not None:
            wandb_run.log({f"validation/{key}": value for key, value in report.items()}, step=epoch)
        selection_value = report[args.selection_metric]
        if selection_value > best:
            best, best_threshold, stale = selection_value, threshold, 0
            best_validation = dict(report)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                print(json.dumps({"early_stopping": True, "epoch": epoch,
                                  "selection_metric": args.selection_metric,
                                  "best_validation_selection_value": best}), flush=True)
                break
    assert best_state is not None and best_validation is not None; model.load_state_dict(best_state)
    validation_index, validation_score = probability(
        model, validation_loader, labels, variant=args.variant, offsets=offsets,
        pairs=pairs, pair_probabilities=pair_probabilities, profiles=profiles,
        device=device, truncate_num=args.truncate_num,
    )
    result = None
    if not args.skip_test:
        assert test_loader is not None
        test_index, test_score = probability(
            model, test_loader, labels, variant=args.variant, offsets=offsets,
            pairs=pairs, pair_probabilities=pair_probabilities, profiles=profiles,
            device=device, truncate_num=args.truncate_num,
        )
        result = utility.metrics(labels[test_index], test_score, best_threshold)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        payload = {"best_validation": best_validation,
                   "best_validation_selection_metric": args.selection_metric,
                   "best_validation_selection_value": best,
                   "epochs_completed": sum(item["epoch"] > 0 for item in curve),
                   "validation_curve": curve}
        if result is not None:
            payload["native_validation_selected"] = result
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
    if result is not None:
        with gzip.open(args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_index", "label", "probability")); writer.writeheader()
            for index, score in zip(test_index, test_score):
                writer.writerow({"sample_index": int(index), "label": int(labels[index]), "probability": float(score)})
    torch.save({"model": best_state, "fold": args.fold, "variant": args.variant,
                "best_validation_selection_metric": args.selection_metric,
                "best_validation_selection_value": best}, args.output_dir / "best_model.pt")
    manifest = {"schema_version": 1, "experiment": "native_rnafm_structires_retraining",
                "variant": args.variant, "fold": args.fold, "seed": args.seed,
                "split_seed": split_seed,
                "dataset_sha256": dataset_hash, "rnafm_base_sha256": sha256(args.rnafm_base),
                "contact_manifest_sha256": contact_manifest_hash, "profile_manifest_sha256": profile_manifest_hash, "validation_fraction": args.validation_fraction,
                "profile_permutation_seed": args.profile_permutation_seed,
                "profile_mapping_sha256": profile_mapping_hash,
                "profile_mapping_fraction_changed": profile_mapping_fraction_changed,
                "drop_profile_channel": args.drop_profile_channel,
                "test_labels_used_for_selection": False,
                "selection": f"validation {args.selection_metric.upper()}",
                "test_evaluation_skipped": args.skip_test,
                "architecture": "author-style RNA-FM t12 BOS 640->40->2" if args.variant == "sequence" else
                "author-style RNA-FM t12 BOS 640->40->2 plus sparse MFE contact encoder and gated residual" if args.variant == "contact" else
                "author-style RNA-FM t12 BOS 640->40->2 plus train-split-normalized ViennaRNA ensemble-profile CNN and gated residual" if args.variant == "ensemble" else
                "author-style RNA-FM t12 with nucleotide-level structure-conditioned attention pooling and gated residual" if args.variant == "conditioned" else
                "RNA-FM t12 nucleotide embeddings with DeepIRES-style multi-kernel dilated CNN, BiGRU and attention head" if args.variant == "deepires" else
                "RNA-FM t12 plus a DeepIRES-style multi-kernel CNN, BiGRU and attention structure adapter with gated residual" if args.variant == "deep_structure" else
                "RNA-FM t12 plus a raw nucleotide/structure DeepIRES multi-kernel CNN, BiGRU and attention branch with gated residual" if args.variant == "hybrid_structure" else
                "RNA-FM t12 with per-nucleotide ViennaRNA profile injection, contextual self-attention pooling, and structure reconstruction auxiliary head" if args.variant == "token_structure_aux" else
                "learned RNA-FM layer-6/8/10/12 mixture with per-nucleotide ViennaRNA profile injection and structure reconstruction auxiliary head" if args.variant == "multilayer_structure_aux" else
                "RNA-FM t12 with a pretrained local sequence/structure representation adapter and one newly trained joint classification head" if args.variant == "pretrained_deep_adapter" else
                "same-fold warm-started RNA-FM/local-adapter joint classifier with a zero-initialized gated structural correction inside the RNA-FM hidden state" if args.variant == "pretrained_refined_deep_adapter" else
                "RNA-FM t12 with separately exposed pretrained local-sequence and gated-structure hidden states consumed by one joint classification head" if args.variant == "pretrained_separated_deep_adapter" else
                "RNA-FM t12 internally conditioned by a pretrained 16-D structure representation through a feature gate and residual, followed by one joint classification head" if args.variant == "pretrained_gated_deep_adapter" else
                "RNA-FM t12 with a pretrained local sequence-only 16-D representation adapter as a matched structure ablation" if args.variant == "pretrained_sequence_deep_adapter" else
                "RNA-FM t12 BOS and masked-mean representations with a pretrained local sequence/structure adapter and one joint classification head" if args.variant == "pretrained_multipool_adapter" else
                "RNA-FM t12 with a pretrained local sequence/structure representation adapter whose structural gate and delta are jointly tuned at a separate low learning rate" if args.variant == "pretrained_tunable_structure_adapter" else
                "RNA-FM t12 with a pretrained structure-aware penultimate representation adapter and one newly trained joint classification head; no teacher logits are used" if args.variant == "pretrained_penultimate_adapter" else
                "RNA-FM t12 with complementary pretrained 16-D and 32-D hidden states from one structure-aware adapter and one newly trained joint classification head; no teacher logits are used" if args.variant == "pretrained_multiscale_adapter" else
                "RNA-FM t12 with a pretrained sequence-only penultimate representation adapter as a matched structure ablation" if args.variant == "pretrained_sequence_penultimate_adapter" else
                "one joint RNA-FM/structure classifier over concatenated hidden representations, initialized to preserve the pretrained structure decision boundary; no separate logits are combined" if args.variant == "pretrained_penultimate_warmstart" else
                "one joint linear classifier over raw RNA-FM BOS and structure-aware penultimate features, initialized to preserve the structure decision boundary; no separate logits are combined" if args.variant == "pretrained_raw_penultimate_warmstart" else
                "one joint classifier over separately pretrained RNA-FM and structure-aware hidden representations, initialized without calling or combining their original output logits" if args.variant == "dual_pretrained_penultimate_warmstart" else
                "task-trained RNA-FM hidden features injected through a zero-start cross-gated residual into a frozen structure-aware penultimate representation and one classifier; no branch logits are computed or combined" if args.variant == "cross_gated_penultimate_adapter" else
                "RNA-FM t12 plus a zero-start residue-graph adapter without structural edges" if args.variant == "graph_sequence" else
                "RNA-FM t12 plus a zero-start residue-graph adapter over MFE base-pair edges" if args.variant == "mfe_graph" else
                "RNA-FM t12 plus a zero-start probability-weighted BPP residue-graph adapter",
                "training_objective": f"classification CE/pairwise-AUC({args.pairwise_auc_weight:g})*{args.classification_loss_weight:g} plus masked-LM CE*{args.mlm_loss_weight:g} plus structure SmoothL1*{args.structure_aux_weight:g}", "mask_probability": args.mask_prob,
                "classification_input": args.classification_input,
                "pairwise_auc_weight": args.pairwise_auc_weight,
                "pairwise_margin": args.pairwise_margin,
                "adapter_init": args.adapter_init if args.variant in {"conditioned", "deep_structure", "hybrid_structure", *structure_aux_variants} else None,
                "structure_aux_weight": args.structure_aux_weight,
                "distillation_weight": args.distillation_weight,
                "distillation_temperature": args.distillation_temperature,
                "teacher_checkpoint_sha256": teacher_checkpoint_hash,
                "warmstart_checkpoint_sha256": warmstart_checkpoint_hash,
                "pretrained_adapter_checkpoint_sha256": pretrained_adapter_checkpoint_hash,
                "rnafm_classifier_checkpoint_sha256": rnafm_classifier_checkpoint_hash,
                "epochs_requested": args.epochs, "tokens_per_batch": args.tokens_per_batch,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_tokens_per_update": args.tokens_per_batch * args.gradient_accumulation_steps,
                "batch_order": "length-aware batches reshuffled per epoch",
                "optimizer": args.optimizer, "weight_decay": args.weight_decay if args.optimizer == "adamw" else 0.0,
                "head_lr": head_lr, "backbone_lr": backbone_lr,
                "layerwise_lr_decay": args.layerwise_lr_decay,
                "adapter_lr": adapter_lr if adapter_parameters else None,
                "unfreeze_last_layers": args.unfreeze_last_layers,
                "head_only_epochs": args.head_only_epochs,
                "initial_validation_checkpoint_candidate": args.variant in warmstart_variants,
                "gradient_clip": args.gradient_clip,
                "lr_schedule": ("LinearLR 1.0->0.5 over 30 epochs" if args.lr_schedule == "linear" else
                                f"per-update warmup {args.warmup_fraction:.3f} then cosine decay")}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with gzip.open(args.output_dir / "validation_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "label", "probability")); writer.writeheader()
        for index, score in zip(validation_index, validation_score):
            writer.writerow({"sample_index": int(index), "label": int(labels[index]), "probability": float(score)})
    if wandb_run is not None:
        for key, value in best_validation.items():
            wandb_run.summary[f"best_validation/{key}"] = value
        if result is not None:
            for key, value in result.items():
                wandb_run.summary[f"test/{key}"] = value
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
