#!/usr/bin/env python3
"""Frozen-embedding smoke/probe for the local RNA autoregressive reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ires_design.prediction import load_ires_ai_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer directory; defaults to --model when files are colocated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_sample(records, max_records: int, seed: int):
    if max_records <= 0 or max_records >= len(records):
        return records
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault((record.test_fold, record.label), []).append(index)
    selected: list[int] = []
    for indices in groups.values():
        target = max(1, round(max_records * len(indices) / len(records)))
        selected.extend(rng.choice(indices, size=min(target, len(indices)), replace=False).tolist())
    if len(selected) > max_records:
        selected = rng.choice(selected, size=max_records, replace=False).tolist()
    return [records[index] for index in sorted(selected)]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local RNA embedding probe")
    device = torch.device("cuda:0")
    model_config = AutoConfig.from_pretrained(args.model)
    if getattr(model_config, "position_embedding_type", "rope") == "hope":
        from transformers.models.llama import modeling_llama

        if not hasattr(modeling_llama, "LlamaHyperbolicEmbedding"):
            raise RuntimeError(
                "HoPE checkpoint detected, but this Transformers build has no "
                "LlamaHyperbolicEmbedding. Refusing to ignore bias_raw weights; restore the "
                "version-pinned custom HoPE implementation before running this probe."
            )
    records = stratified_sample(
        load_ires_ai_records(args.dataset), args.max_records, args.seed
    )
    tokenizer_path = args.tokenizer or args.model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except ValueError as exc:
        if "TokenizersBackend" not in str(exc):
            raise
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16
        )
    model.to(device).eval()

    pooled_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            encoded = tokenizer(
                [record.sequence for record in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            outputs = model.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled_batches.append(pooled.cpu().numpy().astype(np.float32))
            print(f"embedded {min(start + len(batch), len(records))}/{len(records)}", flush=True)

    embeddings = np.concatenate(pooled_batches, axis=0)
    labels = np.asarray([record.label for record in records], dtype=np.int8)
    folds = np.asarray([record.test_fold for record in records], dtype=np.int8)
    np.savez_compressed(
        args.output_dir / "embeddings.npz", embeddings=embeddings, labels=labels, folds=folds
    )
    with (args.output_dir / "records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].to_dict()))
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fold_rows: list[dict[str, float | int]] = []
    for test_fold in sorted(set(folds.tolist())):
        test = folds == test_fold
        validation = folds == ((test_fold + 1) % 10)
        train = ~(test | validation)
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=args.seed + test_fold
            ),
        )
        estimator.fit(embeddings[train], labels[train])
        probability = estimator.predict_proba(embeddings[test])[:, 1]
        fold_rows.append(
            {
                "test_fold": test_fold,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "auc": float(roc_auc_score(labels[test], probability)),
                "aupr": float(average_precision_score(labels[test], probability)),
                "f1_at_0_5": float(f1_score(labels[test], probability >= 0.5)),
            }
        )
    with (args.output_dir / "fold_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    manifest = {
        "schema_version": 1,
        "experiment": "local_rfam_ar_frozen_embedding_smoke",
        "role": "historical supplementary backbone probe",
        "model_path": str(args.model.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "model_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "dataset_path": str(args.dataset.resolve()),
        "n_records": len(records),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "test_labels_used_for_tuning": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(fold_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
