#!/usr/bin/env python3
"""Create a label-free, fixed-width ViennaRNA profile cache for IRES-RNAFM.

The released IRES benchmark has predominantly 174-nt sequences but includes a
small number of longer records.  This tool folds each *unique* public sequence
once, truncates only at the same token budget used by the upstream classifier,
and writes a memory-mappable fixed-width array.  The final channel is an
explicit valid-position mask, so downstream encoders do not mistake zero
padding for an unpaired RNA state.  Activity labels are deliberately neither
read nor written.

The array is external run data rather than a Git asset.  It is designed for a
subsequent position-aware StructIRES classifier evaluated on the full released
ten-fold protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


CHANNELS = (
    "ensemble_pairing_probability",
    "mfe_paired_state",
    "centroid_paired_state",
    "normalized_position",
    "valid_position_mask",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) // 2)))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_unique_sequences(dataset: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(dataset) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv") and "__MACOSX" not in name]
        if len(members) != 1:
            raise ValueError("expected exactly one CSV inside the benchmark archive")
        unique: dict[str, str] = {}
        with archive.open(members[0], "r") as binary:
            reader = csv.DictReader((line.decode("utf-8") for line in binary))
            for row in reader:
                identifier = str(row["ID"])
                sequence = str(row["Sequence"]).upper().replace("T", "U")
                if not sequence or set(sequence) - set("ACGU"):
                    raise ValueError(f"non-canonical sequence for {identifier}")
                previous = unique.setdefault(identifier, sequence)
                if previous != sequence:
                    raise ValueError(f"ID {identifier} maps to inconsistent sequences across folds")
    if len(unique) != 46774:
        raise ValueError(f"expected 46774 unique benchmark sequences, observed {len(unique)}")
    return sorted(unique.items())


def fold_profile(task: tuple[str, int]) -> tuple[np.ndarray, int]:
    sequence, max_length = task
    sequence = sequence[:max_length]
    # This environment is intentionally independent from the training conda
    # environment because ViennaRNA's binary bindings are pinned separately.
    import RNA

    compound = RNA.fold_compound(sequence)
    mfe_structure, mfe = compound.mfe()
    compound.exp_params_rescale(mfe)
    compound.pf()
    centroid_structure, _ = compound.centroid()
    bpp = compound.bpp()
    length = len(sequence)
    profile = np.zeros((max_length, len(CHANNELS)), dtype=np.float32)
    paired = np.zeros(length, dtype=np.float32)
    for left in range(1, length + 1):
        for right in range(left + 1, length + 1):
            probability = float(bpp[left][right])
            if probability:
                paired[left - 1] += probability
                paired[right - 1] += probability
    profile[:length, 0] = paired
    profile[:length, 1] = np.fromiter((item != "." for item in mfe_structure), dtype=np.float32, count=length)
    profile[:length, 2] = np.fromiter((item != "." for item in centroid_structure), dtype=np.float32, count=length)
    profile[:length, 3] = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    profile[:length, 4] = 1.0
    if not np.all(np.isfinite(profile)):
        raise RuntimeError("ViennaRNA produced a non-finite position profile")
    return profile, length


def main() -> int:
    args = arguments()
    if args.max_length < 1:
        raise ValueError("max-length must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    rows = load_unique_sequences(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    values_path = args.output_dir / "profiles.npy"
    profile_array = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float32,
                                                shape=(len(rows), args.max_length, len(CHANNELS)))
    try:
        tasks = ((sequence, args.max_length) for _, sequence in rows)
        lengths: list[int] = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, (profile, length) in enumerate(executor.map(fold_profile, tasks, chunksize=16), start=1):
                profile_array[index - 1] = profile
                lengths.append(length)
                if index % 1000 == 0 or index == len(rows):
                    print(f"folded {index}/{len(rows)}", flush=True)
        profile_array.flush()
    finally:
        del profile_array
    np.save(args.output_dir / "sequence_ids.npy", np.asarray([identifier for identifier, _ in rows]))
    manifest = {
        "schema_version": 1,
        "feature_family": "ViennaRNA fixed-width position-aware ensemble profiles",
        "channels": list(CHANNELS),
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "n_unique_sequences": len(rows),
        "max_length": args.max_length,
        "profile_shape": [len(rows), args.max_length, len(CHANNELS)],
        "input_length_min": min(lengths),
        "input_length_max_after_truncation": max(lengths),
        "n_truncated": sum(len(sequence) > args.max_length for _, sequence in rows),
        "uses_activity_labels": False,
        "padding_semantics": "zeros with a dedicated valid_position_mask channel",
        "viennarna_version": __import__("RNA").__version__,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "shape": manifest["profile_shape"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
