#!/usr/bin/env python3
"""Enumerate exact high-identity pairs for one sequence length on a bounded GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ires_design.prediction import load_ires_ai_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=174)
    parser.add_argument("--identity", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-memory-fraction", type=float, default=0.10)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_sequences(sequences: list[str]) -> np.ndarray:
    lookup = np.full(256, 255, dtype=np.uint8)
    for index, base in enumerate(b"ACGU"):
        lookup[base] = index
    encoded = np.vstack(
        [lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)] for sequence in sequences]
    )
    if np.any(encoded > 3):
        raise ValueError("all sequences must use the canonical A/C/G/U alphabet")
    return encoded


def main() -> int:
    args = parse_args()
    if not 0 < args.identity <= 1:
        raise ValueError("identity must be in (0, 1]")
    if not 0 < args.max_memory_fraction <= 1:
        raise ValueError("max-memory-fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact all-pairs enumeration")
    torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, device=0)
    device = torch.device("cuda:0")

    records = [row for row in load_ires_ai_records(args.dataset) if row.length == args.length]
    sequences = [row.sequence for row in records]
    encoded = torch.from_numpy(encode_sequences(sequences)).to(device=device, dtype=torch.long)
    one_hot = functional.one_hot(encoded, num_classes=4).reshape(len(records), -1).to(torch.float16)
    del encoded
    minimum_matches = int(math.ceil(args.identity * args.length - 1e-12))

    edge_path = args.output_dir / "identity_edges.csv.gz"
    import gzip

    n_edges = 0
    with gzip.open(edge_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("left_sequence_id", "right_sequence_id", "matches", "length", "identity")
        )
        with torch.inference_mode():
            for start in range(0, len(records), args.batch_size):
                stop = min(start + args.batch_size, len(records))
                scores = one_hot[start:stop] @ one_hot.T
                for local_index in range(stop - start):
                    global_index = start + local_index
                    columns = torch.nonzero(
                        scores[local_index, global_index + 1 :] >= minimum_matches,
                        as_tuple=False,
                    ).flatten()
                    if not len(columns):
                        continue
                    columns = columns + global_index + 1
                    matches = scores[local_index, columns].to(torch.int16).cpu().tolist()
                    for right_index, match_count in zip(columns.cpu().tolist(), matches):
                        writer.writerow(
                            (
                                records[global_index].sequence_id,
                                records[right_index].sequence_id,
                                match_count,
                                args.length,
                                match_count / args.length,
                            )
                        )
                        n_edges += 1
                del scores
                if start == 0 or stop == len(records) or (start // args.batch_size + 1) % 20 == 0:
                    allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
                    print(
                        f"processed={stop}/{len(records)} edges={n_edges} "
                        f"peak_allocated_mib={allocated:.1f}",
                        flush=True,
                    )

    properties = torch.cuda.get_device_properties(device)
    manifest = {
        "schema_version": 1,
        "experiment": "exact_equal_length_hamming_graph",
        "dataset_path": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "sequence_length": args.length,
        "n_sequences": len(records),
        "identity_threshold": args.identity,
        "minimum_matching_positions": minimum_matches,
        "n_edges": n_edges,
        "batch_size": args.batch_size,
        "max_memory_fraction": args.max_memory_fraction,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": properties.name,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "definition": (
            "Exact ungapped Hamming identity for equal-length sequences. "
            "Cross-length and gapped similarity are outside this graph."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
