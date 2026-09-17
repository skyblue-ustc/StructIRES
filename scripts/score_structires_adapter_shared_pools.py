#!/usr/bin/env python3
"""Score frozen design pools with one frozen StructIRES-Adapter checkpoint.

This is proposal scoring only.  It does not fit parameters, choose a threshold,
or read candidate activity labels.  Fold-specific normalization is recovered
from the original training partition and immutable release-profile cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-profile-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--release-profile-dir", type=Path, required=True)
    parser.add_argument("--rnafm-base", type=Path, required=True)
    parser.add_argument("--upstream-fm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens-per-batch", type=int, default=4096)
    parser.add_argument("--truncate-num", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
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
    spec = importlib.util.spec_from_file_location("candidate_structires_adapter", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_statistics(raw_profiles: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(4, dtype=np.float64)
    squared = np.zeros(4, dtype=np.float64)
    count = 0.0
    for start in range(0, len(train), 256):
        values = np.asarray(raw_profiles[train[start : start + 256]], dtype=np.float64)
        mask = values[:, :, 4:5]
        total += (values[:, :, :4] * mask).sum(axis=(0, 1))
        squared += ((values[:, :, :4] ** 2) * mask).sum(axis=(0, 1))
        count += float(mask.sum())
    if count <= 0:
        raise RuntimeError("training profile cache contains no valid positions")
    mean = total / count
    variance = squared / count - mean**2
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    source_manifest_path = args.run_dir / "run_manifest.json"
    source_checkpoint_path = args.run_dir / "best_model.pt"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("variant") != "pretrained_deep_adapter":
        raise ValueError("run-dir is not a pretrained_deep_adapter experiment")
    if sha256(args.dataset) != source_manifest.get("dataset_sha256"):
        raise ValueError("dataset hash differs from frozen adapter run")
    if sha256(args.rnafm_base) != source_manifest.get("rnafm_base_sha256"):
        raise ValueError("RNA-FM base hash differs from frozen adapter run")
    if sha256(args.release_profile_dir / "manifest.json") != source_manifest.get(
        "profile_manifest_sha256"
    ):
        raise ValueError("release profile manifest differs from frozen adapter run")
    if sha256(args.teacher_checkpoint) != source_manifest.get(
        "pretrained_adapter_checkpoint_sha256"
    ):
        raise ValueError("teacher checkpoint differs from frozen adapter run")

    candidates: list[dict[str, object]] = []
    pool_audits: list[dict[str, object]] = []
    for pool in args.pool:
        path = pool / "candidates.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        candidates.extend({"pool_dir": str(pool.resolve()), **row} for row in rows)
        pool_audits.append({"path": str(path.resolve()), "sha256": sha256(path), "n": len(rows)})
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs are not unique")
    profile_ids = np.load(
        args.candidate_profile_dir / "candidate_ids.npy", allow_pickle=False
    ).astype(str)
    if candidate_ids != profile_ids.tolist():
        raise ValueError("candidate profile IDs differ from pool order")
    profile_manifest_path = args.candidate_profile_dir / "manifest.json"
    candidate_profile_manifest = json.loads(profile_manifest_path.read_text(encoding="utf-8"))
    if candidate_profile_manifest.get("n_candidates") != len(candidates):
        raise ValueError("candidate profile count differs from pools")
    expected_pool_hashes = [entry["sha256"] for entry in pool_audits]
    observed_pool_hashes = [entry["sha256"] for entry in candidate_profile_manifest["pools"]]
    if expected_pool_hashes != observed_pool_hashes:
        raise ValueError("candidate profile cache was built from different pools")

    training = load_training_module()
    utility = training.utility_module()
    fold = int(source_manifest["fold"])
    seed = int(source_manifest["seed"])
    frame = utility.load_fold_rows(args.dataset, fold)
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    labels = frame.label.to_numpy(dtype=np.int64)
    source_train = frame.type.to_numpy() == "train"
    upstream_train = np.flatnonzero(source_train)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(source_manifest["validation_fraction"]),
        random_state=seed,
    )
    train_local, _ = next(splitter.split(upstream_train, labels[upstream_train]))
    train = upstream_train[train_local]

    release_ids = np.load(
        args.release_profile_dir / "sequence_ids.npy", allow_pickle=False
    ).astype(str)
    if release_ids.tolist() != frame.ID.astype(str).tolist():
        raise ValueError("release profile IDs differ from frozen fold order")
    release_profiles = np.load(args.release_profile_dir / "profiles.npy", mmap_mode="r")
    if release_profiles.shape != (len(frame), args.truncate_num, 5):
        raise ValueError(f"unexpected release profile shape: {release_profiles.shape}")
    mean, std = profile_statistics(release_profiles, train)

    candidate_profiles = np.load(
        args.candidate_profile_dir / "profiles.npy", mmap_mode="r"
    )
    if candidate_profiles.shape != (len(candidates), 174, 5):
        raise ValueError(f"unexpected candidate profile shape: {candidate_profiles.shape}")

    class NormalizedCandidateProfiles:
        def __getitem__(self, index):
            output = np.asarray(candidate_profiles[index], dtype=np.float32).copy()
            output[..., :4] = (output[..., :4] - mean) / std
            output[..., :4] *= output[..., 4:5]
            return output

    data_module, pretrained = utility.load_fm(args.upstream_fm_dir)
    backbone, alphabet = pretrained.rna_fm_t12(str(args.rnafm_base))
    adapter_module = training.deepires_teacher_module()
    deepires = adapter_module.load_deepires_module()
    adapter = adapter_module.GatedDeepIRESStructure(deepires, dropout=0.2)
    adapter_payload = torch.load(
        args.teacher_checkpoint, map_location="cpu", weights_only=True
    )
    adapter.load_state_dict(adapter_payload["model"], strict=True)
    lookup = [0] * (max(alphabet.tok_to_idx.values()) + 1)
    for encoded, base in enumerate("ACGU", start=1):
        lookup[alphabet.tok_to_idx[base]] = encoded
    model = training.PretrainedDeepStructureRNAFM(
        backbone,
        args.dropout,
        adapter=adapter,
        deepires_token_lookup=lookup,
    )
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("fold") != fold or checkpoint.get("variant") != "pretrained_deep_adapter":
        raise ValueError("checkpoint identity differs from run manifest")
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = model.to(device)

    sequences = [str(row["sequence"]).upper().replace("T", "U") for row in candidates]
    indices = np.arange(len(candidates), dtype=np.int64)
    loader = utility.make_loader(
        data_module, alphabet, indices, sequences, args.tokens_per_batch
    )
    scored_index, probability = training.probability(
        model,
        loader,
        np.zeros(len(candidates), dtype=np.int64),
        variant="pretrained_deep_adapter",
        offsets=None,
        pairs=None,
        pair_probabilities=None,
        profiles=NormalizedCandidateProfiles(),
        device=device,
        truncate_num=args.truncate_num,
    )
    if len(scored_index) != len(candidates) or len(set(scored_index.tolist())) != len(candidates):
        raise RuntimeError("candidate scoring did not return every candidate exactly once")
    score_by_index = dict(zip(scored_index.tolist(), probability.tolist()))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, row in enumerate(candidates):
        sequence = str(row["sequence"]).upper().replace("T", "U")
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "parent_id": row["parent_id"],
                "run_seed": row.get("run_seed", row["seed"]),
                "pool_dir": row["pool_dir"],
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                f"structires_adapter_probability_fold{fold}": score_by_index[index],
            }
        )
    output_path = args.output_dir / f"structires_adapter_pool_scores_fold{fold}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "experiment": "frozen_structires_adapter_score_on_frozen_shared_pools",
        "scope": "classification proposal score only; no candidate activity labels used",
        "fold": fold,
        "seed": seed,
        "pools": pool_audits,
        "candidate_profile_manifest": {
            "path": str(profile_manifest_path.resolve()),
            "sha256": sha256(profile_manifest_path),
        },
        "source_adapter_run": str(args.run_dir.resolve()),
        "source_adapter_manifest_sha256": sha256(source_manifest_path),
        "source_adapter_checkpoint_sha256": sha256(source_checkpoint_path),
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256(args.teacher_checkpoint),
        "dataset_sha256": sha256(args.dataset),
        "release_profile_manifest_sha256": sha256(
            args.release_profile_dir / "manifest.json"
        ),
        "rnafm_base_sha256": sha256(args.rnafm_base),
        "normalization": "four profile channels normalized with this fold's inner-training records only",
        "profile_mean": mean.tolist(),
        "profile_std": std.tolist(),
        "rnafm_input": "full candidate up to the frozen 1024-token preprocessing limit",
        "local_adapter_input": "leftmost 174 nt; profile computed by folding the full candidate before prefix cropping",
        "n_candidates": len(candidates),
        "tokens_per_batch": args.tokens_per_batch,
        "device": str(device),
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fold": fold,
                "n_candidates": len(rows),
                "mean_probability": float(np.mean(probability)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
