#!/usr/bin/env python3
"""Batch-score a frozen candidate pool with the local RNA language model.

The score is an RNA-language-model plausibility objective, not an assay
activity measurement.  It is intentionally recorded separately from the
direct-RNA evaluation and structural metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidates(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            for key in ("candidate_id", "sequence", "parent_id", "seed"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number} missing {key}")
            rows.append(row)
    if not rows:
        raise ValueError("candidate pool is empty")
    return rows


def render(sequence: str) -> str:
    return f"<|bos|> <|5|> {sequence.replace('U', 'T')} <|3|> <|eos|>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerFast

    rows = read_candidates(args.input)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except ValueError as exc:
        # The local historical checkpoint retains the now-removed
        # ``TokenizersBackend`` class name while its tokenizer.json is valid.
        if "TokenizersBackend" not in str(exc):
            raise
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        model = LlamaForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, attn_implementation="flash_attention_2"
        )
    except Exception:
        model = LlamaForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, attn_implementation="sdpa"
        )
    model = model.to(device).eval()

    scored: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [render(str(row["sequence"])) for row in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(device)
            # PreTrainedTokenizerFast exposes token_type_ids for this legacy
            # BPE tokenizer, whereas Llama does not accept that argument.
            encoded.pop("token_type_ids", None)
            if encoded.input_ids.shape[1] > model.config.max_position_embeddings:
                raise ValueError(
                    f"tokenized length {encoded.input_ids.shape[1]} exceeds model maximum "
                    f"{model.config.max_position_embeddings}"
                )
            logits = model(**encoded).logits[:, :-1]
            token_ids = encoded.input_ids[:, 1:]
            mask = encoded.attention_mask[:, 1:].bool()
            token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
                -1, token_ids.unsqueeze(-1)
            ).squeeze(-1)
            counts = mask.sum(dim=1)
            values = (token_log_probs * mask).sum(dim=1) / counts
            for row, value, count in zip(batch, values.cpu().tolist(), counts.cpu().tolist()):
                scored.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "parent_id": row["parent_id"],
                        "run_seed": row["seed"],
                        "sequence_sha256": hashlib.sha256(str(row["sequence"]).encode("ascii")).hexdigest(),
                        "length": len(str(row["sequence"])),
                        "lm_log_likelihood_per_token": float(value),
                        "n_scored_tokens": int(count),
                    }
                )
            print(f"scored {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    manifest = {
        "schema_version": 1,
        "experiment": "rfam_llama_likelihood_scoring",
        "scope": "proposal plausibility only; not an activity oracle",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "model_path": str(args.model_path),
        "model_config_sha256": sha256(args.model_path / "config.json"),
        "model_weights_sha256": sha256(args.model_path / "model.safetensors"),
        "device": str(device),
        "torch_version": torch.__version__,
        "n_candidates": len(scored),
        "batch_size": args.batch_size,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
