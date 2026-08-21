#!/usr/bin/env python3
"""Prepare an immutable workspace for an upstream IRES-RNAFM native-fold rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=False)
    (args.run_root / "Data").mkdir()
    (args.run_root / "models").mkdir()
    (args.run_root / "results").mkdir()
    shutil.copytree(args.upstream_root / "Script", args.run_root / "Script")
    fm_init = args.run_root / "Script" / "fm" / "__init__.py"
    original_fm_init = fm_init.read_text(encoding="utf-8")
    unused_import = "from .downstream import *\n"
    if unused_import not in original_fm_init:
        raise ValueError("expected upstream fm.downstream wildcard import was not found")
    fm_init.write_text(original_fm_init.replace(unused_import, ""), encoding="utf-8")

    csv_name = "v2_dataset_with_unified_stratified_shuffle_train_test_split.csv"
    csv_path = args.run_root / "Data" / csv_name
    with zipfile.ZipFile(args.dataset_zip) as archive:
        members = [name for name in archive.namelist() if name.endswith(csv_name) and not name.startswith("__MACOSX")]
        if len(members) != 1:
            raise ValueError(f"expected one canonical dataset member, found {members}")
        with archive.open(members[0]) as source, csv_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    prefix = f"reproduction_RNAFM_native_folds{args.folds}_seed{args.seed}"
    torchrun_args = [
        "--nproc_per_node=1",
        "--master_port=28890",
        "IRES_RNAFM.py",
        "--device_ids", "0",
        "--seed", str(args.seed),
        "--prefix", prefix,
        "--folds", str(args.folds),
        "--epochs", str(args.epochs),
        "--bos_emb",
        "--truncate",
        "--finetune_esm",
    ]
    manifest = {
        "schema_version": 1,
        "experiment": "published_ires_rnafm_native_protocol_rerun",
        "status": "prepared",
        "scope": "classification-only reproduction; no sequence generation or wet-lab work",
        "upstream_root": str(args.upstream_root),
        "upstream_commit": git_commit(args.upstream_root),
        "upstream_script": "Script/IRES_RNAFM.py",
        "upstream_script_sha256": sha256(args.upstream_root / "Script" / "IRES_RNAFM.py"),
        "compatibility_patch": {
            "file": "Script/fm/__init__.py",
            "change": "Removed the unused fm.downstream wildcard import, which otherwise requires ptflops before the RNA-FM classifier can load.",
            "upstream_sha256": sha256(args.upstream_root / "Script" / "fm" / "__init__.py"),
            "executed_sha256": sha256(fm_init),
            "training_or_model_logic_changed": False,
        },
        "dataset_zip": str(args.dataset_zip),
        "dataset_zip_sha256": sha256(args.dataset_zip),
        "extracted_dataset_sha256": sha256(csv_path),
        "node": args.node,
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": str(args.physical_gpu),
        "logical_device_ids_argument": "0",
        "folds": args.folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "prefix": prefix,
        "torchrun_args": torchrun_args,
        "protocol_warning": "The upstream script evaluates test data every epoch and selects the checkpoint on test AUPR; this rerun reproduces that released protocol and is not an independent blind test.",
    }
    (args.run_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
