#!/usr/bin/env python3
"""Audit exact sequence overlap in the released DeepCIP data splits.

This is a provenance audit only.  It does not train a model or tune a threshold.
Sequences are canonicalized to upper-case DNA so RNA/DNA alphabet differences do
not hide exact matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepcip-root", type=Path, required=True)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--direct-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(sequence: str) -> str:
    return "".join(sequence.split()).upper().replace("U", "T")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, canonicalize("".join(chunks))))
            name = line[1:].split()[0]
            chunks = []
        else:
            if name is None:
                raise ValueError(f"sequence before FASTA header in {path}")
            chunks.append(line)
    if name is not None:
        records.append((name, canonicalize("".join(chunks))))
    return records


def load_deepcip_split(root: Path, split_name: str) -> pd.DataFrame:
    if split_name == "test":
        fasta_path = root / "process_data" / "seq" / "test_data.fa"
        label_path = root / "process_data" / "label" / "test_label.npy"
        role = "released_test"
    else:
        fasta_path = root / "process_data" / "seq" / f"{split_name}_data.fa"
        label_path = root / "process_data" / "label" / f"train_{split_name}_label.npy"
        role = "released_train"
    records = read_fasta(fasta_path)
    labels = np.load(label_path).reshape(-1)
    if len(records) != len(labels):
        raise ValueError(
            f"{split_name}: {len(records)} FASTA records but {len(labels)} labels"
        )
    frame = pd.DataFrame(
        {
            "deepcip_split": split_name,
            "deepcip_role": role,
            "record_id": [record[0] for record in records],
            "sequence_dna": [record[1] for record in records],
            "deepcip_label": labels.astype(int),
        }
    )
    if not frame["deepcip_label"].isin([0, 1]).all():
        raise ValueError(f"non-binary labels in {label_path}")
    if frame["sequence_dna"].duplicated().any():
        raise ValueError(f"duplicate exact sequences within DeepCIP {split_name}")
    frame["length"] = frame["sequence_dna"].str.len()
    return frame


def load_ireslm(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.endswith("v2_dataset_with_unified_stratified_shuffle_train_test_split.csv")
            and not name.startswith("__MACOSX")
        ]
        if len(members) != 1:
            raise ValueError(f"expected one canonical CSV member, found {members}")
        with archive.open(members[0]) as handle:
            long_data = pd.read_csv(
                handle,
                usecols=["idx", "ID", "Sequence", "IRES_class_600", "Source"],
            )
    data = long_data.drop_duplicates("idx").copy()
    data["sequence_dna"] = data["Sequence"].map(canonicalize)
    if data["sequence_dna"].duplicated().any():
        raise ValueError("duplicate exact sequences in canonical IRES-LM data")
    return data[["idx", "ID", "sequence_dna", "IRES_class_600", "Source"]]


def split_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "n": int(len(frame)),
        "n_positive": int(frame["deepcip_label"].sum()),
        "positive_fraction": float(frame["deepcip_label"].mean()),
        "length_min": int(frame["length"].min()),
        "length_max": int(frame["length"].max()),
        "length_counts": {
            str(int(key)): int(value)
            for key, value in frame["length"].value_counts().sort_index().items()
        },
    }


def overlap_summary(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, object]:
    overlap = left.merge(right, on="sequence_dna", suffixes=("_left", "_right"))
    summary: dict[str, object] = {
        "n_exact": int(len(overlap)),
        "fraction_of_left": float(len(overlap) / len(left)),
        "fraction_of_right": float(len(overlap) / len(right)),
    }
    if {"deepcip_label_left", "deepcip_label_right"}.issubset(overlap.columns):
        summary["label_agreement"] = (
            float(
                (overlap["deepcip_label_left"] == overlap["deepcip_label_right"]).mean()
            )
            if len(overlap)
            else None
        )
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    splits = {
        name: load_deepcip_split(args.deepcip_root, name)
        for name in ("set1", "set2", "set3", "test")
    }
    ireslm = load_ireslm(args.dataset_zip)

    pairwise: dict[str, object] = {}
    names = list(splits)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            pairwise[f"{left_name}__{right_name}"] = overlap_summary(
                splits[left_name], splits[right_name]
            )

    occurrence = pd.concat(splits.values(), ignore_index=True)
    unique_deepcip = occurrence.drop_duplicates("sequence_dna").copy()
    label_consistency = occurrence.groupby("sequence_dna")["deepcip_label"].nunique()
    if (label_consistency > 1).any():
        raise ValueError("DeepCIP assigns conflicting labels to an exact sequence")
    train_shared_by_label = {}
    for label in (0, 1):
        sequence_sets = [
            set(splits[name].loc[splits[name]["deepcip_label"] == label, "sequence_dna"])
            for name in ("set1", "set2", "set3")
        ]
        train_shared_by_label[str(label)] = int(len(set.intersection(*sequence_sets)))

    benchmark_rows: list[pd.DataFrame] = []
    benchmark_summary: dict[str, object] = {}
    for name, frame in splits.items():
        overlap = frame.merge(ireslm, on="sequence_dna", how="inner", validate="one_to_one")
        overlap["label_agreement"] = (
            overlap["deepcip_label"] == overlap["IRES_class_600"].astype(int)
        )
        contingency = pd.crosstab(
            overlap["deepcip_label"], overlap["IRES_class_600"].astype(int)
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        benchmark_summary[name] = {
            "n_exact": int(len(overlap)),
            "fraction_of_deepcip_split": float(len(overlap) / len(frame)),
            "cross_assay_label_agreement": float(overlap["label_agreement"].mean())
            if len(overlap)
            else None,
            "label_contingency_deepcip_rows_ireslm_columns": {
                "deepcip_0__ireslm_0": int(contingency.loc[0, 0]),
                "deepcip_0__ireslm_1": int(contingency.loc[0, 1]),
                "deepcip_1__ireslm_0": int(contingency.loc[1, 0]),
                "deepcip_1__ireslm_1": int(contingency.loc[1, 1]),
            },
            "ireslm_source_counts": overlap["Source"].value_counts().to_dict(),
        }
        benchmark_rows.append(overlap)

    benchmark_overlap = pd.concat(benchmark_rows, ignore_index=True)
    benchmark_overlap.to_csv(
        args.output_dir / "deepcip_ireslm_exact_overlaps.csv.gz",
        index=False,
        compression="gzip",
    )

    direct_summary: dict[str, object] | None = None
    if args.direct_data is not None:
        direct = pd.read_csv(args.direct_data)
        sequence_column = "sequence" if "sequence" in direct.columns else "Sequence"
        direct["sequence_dna"] = direct[sequence_column].map(canonicalize)
        direct_unique = direct.drop_duplicates("sequence_dna")
        direct_overlap = unique_deepcip.merge(direct_unique, on="sequence_dna", how="inner")
        direct_summary = {
            "n_direct_rows": int(len(direct)),
            "n_direct_unique_sequences": int(len(direct_unique)),
            "n_exact_deepcip_overlap": int(len(direct_overlap)),
            "fraction_of_direct_unique": float(len(direct_overlap) / len(direct_unique)),
        }

    files = []
    for name in splits:
        fasta_name = "test_data.fa" if name == "test" else f"{name}_data.fa"
        label_name = "test_label.npy" if name == "test" else f"train_{name}_label.npy"
        for path in (
            args.deepcip_root / "process_data" / "seq" / fasta_name,
            args.deepcip_root / "process_data" / "label" / label_name,
        ):
            files.append({"path": str(path), "sha256": sha256(path)})

    summary = {
        "schema_version": 1,
        "experiment": "deepcip_released_data_exact_overlap_audit",
        "deepcip_revision": "8ba514944925be741013fb248918da6b7afb5b22",
        "split_summaries": {name: split_summary(frame) for name, frame in splits.items()},
        "pairwise_exact_overlaps": pairwise,
        "n_deepcip_occurrences": int(len(occurrence)),
        "n_deepcip_unique_sequences": int(len(unique_deepcip)),
        "n_sequences_shared_across_all_three_train_sets_by_label": train_shared_by_label,
        "all_deepcip_duplicate_labels_consistent": True,
        "ireslm_exact_overlaps": benchmark_summary,
        "direct_assay_exact_overlap": direct_summary,
        "inputs": {
            "dataset_zip": str(args.dataset_zip),
            "dataset_zip_sha256": sha256(args.dataset_zip),
            "direct_data": str(args.direct_data) if args.direct_data else None,
            "direct_data_sha256": sha256(args.direct_data) if args.direct_data else None,
            "deepcip_files": files,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
