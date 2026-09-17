#!/usr/bin/env python3
"""Evaluate a locked native RNA-FM structure-adapter checkpoint once.

The command deliberately requires a selection-lock manifest created before
test evaluation.  It reconstructs the inner validation split and structure
normalization, verifies the saved checkpoint against its recorded validation
metrics, and only then scores the untouched official test fold.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


SUPPORTED_VARIANTS = {
    "sequence": "AuthorStyleRNAFM",
    "pretrained_deep_adapter": "PretrainedDeepStructureRNAFM",
    "pretrained_refined_deep_adapter": "WarmstartedStructureRefinementRNAFM",
    "pretrained_separated_deep_adapter": "PretrainedSeparatedDeepStructureRNAFM",
    "pretrained_gated_deep_adapter": "PretrainedGatedDeepStructureRNAFM",
    "pretrained_sequence_deep_adapter": "PretrainedSequenceDeepRNAFM",
    "pretrained_multipool_adapter": "PretrainedMultiPoolStructureRNAFM",
    "pretrained_tunable_structure_adapter": "TunablePretrainedStructureRNAFM",
    "pretrained_penultimate_adapter": "PretrainedPenultimateStructureRNAFM",
    "pretrained_multiscale_adapter": "PretrainedMultiscaleStructureRNAFM",
    "pretrained_sequence_penultimate_adapter": "PretrainedSequencePenultimateRNAFM",
    "pretrained_penultimate_warmstart": "PretrainedPenultimateWarmstartRNAFM",
    "pretrained_raw_penultimate_warmstart": "PretrainedRawPenultimateWarmstartRNAFM",
    "dual_pretrained_penultimate_warmstart": "PretrainedPenultimateWarmstartRNAFM",
    "cross_gated_penultimate_adapter": "CrossGatedPenultimateRNAFM",
}
METRICS = ("auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc", "ece10")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--rnafm-base", type=Path, required=True)
    parser.add_argument("--upstream-fm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens-per-batch", type=int, default=4096)
    parser.add_argument("--truncate-num", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--metric-tolerance", type=float, default=2e-6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_module():
    source = Path(__file__).with_name("train_structires_native_contact_fusion.py")
    spec = importlib.util.spec_from_file_location("locked_native_adapter", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_lock(lock: dict[str, object], run_dir: Path, manifest: dict[str, object]) -> None:
    if lock.get("status") != "selected_for_formal_evaluation":
        raise ValueError("selection lock is not frozen for formal evaluation")
    if lock.get("selection_complete_before_test") is not True:
        raise ValueError("selection lock must assert selection_complete_before_test=true")
    selected = lock.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("selection lock has no selected configuration")
    allowed_variants = {selected.get("trainer_variant")}
    comparators = lock.get("formal_comparators", [])
    if not isinstance(comparators, list) or any(
        not isinstance(value, dict) or not isinstance(value.get("trainer_variant"), str)
        for value in comparators
    ):
        raise ValueError("selection lock formal_comparators must be a list of configurations")
    allowed_variants.update(value["trainer_variant"] for value in comparators)
    if manifest.get("variant") not in allowed_variants:
        raise ValueError("development checkpoint variant is not locked for formal evaluation")
    locked_runs = lock.get("source_dev_run_dirs")
    if not isinstance(locked_runs, list) or str(run_dir.resolve()) not in {
        str(Path(value).resolve()) for value in locked_runs
    }:
        raise ValueError("development run is not included in the selection lock")
    if manifest.get("test_evaluation_skipped") is not True:
        raise ValueError("source checkpoint is not a development-only run")
    if manifest.get("test_labels_used_for_selection") is not False:
        raise ValueError("source checkpoint does not certify test-independent selection")


def write_predictions(path: Path, indices: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "label", "probability"))
        writer.writeheader()
        for index, score in zip(indices, scores):
            writer.writerow(
                {
                    "sample_index": int(index),
                    "label": int(labels[index]),
                    "probability": float(score),
                }
            )


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    source_manifest_path = args.run_dir / "run_manifest.json"
    source_metrics_path = args.run_dir / "metrics.json"
    source_checkpoint_path = args.run_dir / "best_model.pt"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    validate_lock(lock, args.run_dir, source_manifest)

    variant = str(source_manifest["variant"])
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported locked adapter variant: {variant}")
    if sha256(args.dataset) != source_manifest.get("dataset_sha256"):
        raise ValueError("dataset hash differs from the development run")
    if sha256(args.rnafm_base) != source_manifest.get("rnafm_base_sha256"):
        raise ValueError("RNA-FM base checkpoint differs from the development run")
    uses_structure_adapter = variant != "sequence"
    if uses_structure_adapter:
        if args.profile_dir is None or args.teacher_checkpoint is None:
            raise ValueError("structure adapter evaluation requires profile and teacher paths")
        if sha256(args.profile_dir / "manifest.json") != source_manifest.get("profile_manifest_sha256"):
            raise ValueError("structure-profile manifest differs from the development run")
        if sha256(args.teacher_checkpoint) != source_manifest.get("pretrained_adapter_checkpoint_sha256"):
            raise ValueError("structure-adapter checkpoint differs from the development run")

    fold = int(source_manifest["fold"])
    seed = int(source_manifest["seed"])
    validation_fraction = float(source_manifest["validation_fraction"])
    training = load_training_module()
    utility = training.utility_module()
    frame = utility.load_fold_rows(args.dataset, fold)
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    labels = frame.label.to_numpy(dtype=np.int64)
    sequences = frame.Sequence.astype(str).tolist()
    source_train = frame.type.to_numpy() == "train"
    source_test = frame.type.to_numpy() == "test"
    upstream_train = np.flatnonzero(source_train)
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    train_local, validation_local = next(
        split.split(upstream_train, labels[upstream_train])
    )
    train = upstream_train[train_local]
    validation = upstream_train[validation_local]
    test = np.flatnonzero(source_test)

    profiles = None
    if uses_structure_adapter:
        profile_ids = np.load(
            args.profile_dir / "sequence_ids.npy", allow_pickle=False
        ).astype(str)
        raw_profiles = np.load(args.profile_dir / "profiles.npy", mmap_mode="r")
        if list(profile_ids) != frame.ID.astype(str).tolist():
            raise ValueError("structure-profile IDs and native fold rows differ")
        expected_shape = (len(frame), args.truncate_num, 5)
        if raw_profiles.shape != expected_shape:
            raise ValueError(
                f"structure-profile shape {raw_profiles.shape} differs from {expected_shape}"
            )
        values = np.asarray(raw_profiles[train], dtype=np.float64)
        mask = values[:, :, 4:5]
        count = float(mask.sum())
        mean = (values[:, :, :4] * mask).sum(axis=(0, 1)) / count
        variance = ((values[:, :, :4] ** 2) * mask).sum(axis=(0, 1)) / count - mean ** 2
        std = np.sqrt(np.maximum(variance, 1e-6))

        class NormalizedProfiles:
            def __getitem__(self, index):
                output = np.asarray(raw_profiles[index], dtype=np.float32).copy()
                output[..., :4] = (
                    output[..., :4] - mean.astype(np.float32)
                ) / std.astype(np.float32)
                output[..., :4] *= output[..., 4:5]
                return output

        profiles = NormalizedProfiles()
    data_module, pretrained = utility.load_fm(args.upstream_fm_dir)
    backbone, alphabet = pretrained.rna_fm_t12(str(args.rnafm_base))
    model_class = getattr(training, SUPPORTED_VARIANTS[variant])
    if uses_structure_adapter:
        adapter_module = training.deepires_teacher_module()
        deepires = adapter_module.load_deepires_module()
        adapter = adapter_module.GatedDeepIRESStructure(deepires, dropout=.2)
        adapter_payload = torch.load(
            args.teacher_checkpoint, map_location="cpu", weights_only=True
        )
        adapter.load_state_dict(adapter_payload["model"], strict=True)
        lookup = [0] * (max(alphabet.tok_to_idx.values()) + 1)
        for encoded, base in enumerate("ACGU", start=1):
            lookup[alphabet.tok_to_idx[base]] = encoded
        model = model_class(
            backbone,
            args.dropout,
            adapter=adapter,
            deepires_token_lookup=lookup,
        )
    else:
        model = model_class(backbone, args.dropout)
    checkpoint = torch.load(
        source_checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("fold") != fold or checkpoint.get("variant") != variant:
        raise ValueError("checkpoint fold or variant differs from its run manifest")
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model = model.to(device)

    validation_loader = utility.make_loader(
        data_module, alphabet, validation, sequences, args.tokens_per_batch
    )
    test_loader = utility.make_loader(
        data_module, alphabet, test, sequences, args.tokens_per_batch
    )
    validation_index, validation_score = training.probability(
        model,
        validation_loader,
        labels,
        variant=variant,
        offsets=None,
        pairs=None,
        pair_probabilities=None,
        profiles=profiles,
        device=device,
        truncate_num=args.truncate_num,
    )
    threshold = float(source_metrics["best_validation"]["threshold"])
    validation_report = utility.metrics(
        labels[validation_index], validation_score, threshold
    )
    mismatch = {
        metric: (
            float(source_metrics["best_validation"][metric]),
            float(validation_report[metric]),
        )
        for metric in METRICS
        if abs(
            float(source_metrics["best_validation"][metric])
            - float(validation_report[metric])
        ) > args.metric_tolerance
    }
    if mismatch:
        raise ValueError(f"development checkpoint validation mismatch: {mismatch}")

    test_index, test_score = training.probability(
        model,
        test_loader,
        labels,
        variant=variant,
        offsets=None,
        pairs=None,
        pair_probabilities=None,
        profiles=profiles,
        device=device,
        truncate_num=args.truncate_num,
    )
    test_report = utility.metrics(labels[test_index], test_score, threshold)
    args.output_dir.mkdir(parents=True)
    write_predictions(
        args.output_dir / "validation_predictions.csv.gz",
        validation_index,
        labels,
        validation_score,
    )
    write_predictions(
        args.output_dir / "test_predictions.csv.gz", test_index, labels, test_score
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_validation_threshold": threshold,
                "source_best_validation": source_metrics["best_validation"],
                "validation_recomputed": validation_report,
                "native_validation_selected": test_report,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "experiment": "native_rnafm_structires_locked_checkpoint_evaluation",
        "variant": variant,
        "fold": fold,
        "seed": seed,
        "dataset_sha256": sha256(args.dataset),
        "profile_manifest_sha256": (
            sha256(args.profile_dir / "manifest.json") if uses_structure_adapter else None
        ),
        "rnafm_base_sha256": sha256(args.rnafm_base),
        "teacher_checkpoint_sha256": (
            sha256(args.teacher_checkpoint) if uses_structure_adapter else None
        ),
        "source_dev_manifest_sha256": sha256(source_manifest_path),
        "source_dev_metrics_sha256": sha256(source_metrics_path),
        "source_dev_checkpoint_sha256": sha256(source_checkpoint_path),
        "selection_lock_sha256": sha256(args.selection_lock),
        "source_dev_run": str(args.run_dir.resolve()),
        "validation_metrics_recomputed": True,
        "test_labels_used_for_selection": False,
        "official_test_fold_evaluated_once_after_lock": True,
        "n_validation": int(len(validation_index)),
        "n_test": int(len(test_index)),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"fold": fold, "test": test_report}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
