"""Utilities for the assay-labelled IRES identification benchmark.

The released IRES-AI random split is stored in long form.  Audit shows that it
contains ten repeated 90/10 holdouts, not a mutually exclusive ten-fold
partition: a sequence can be in several test sets or in none.  This module
recovers one canonical biological record, preserves all published test-set
assignments for provenance, and constructs a deterministic project partition
for leakage-aware pilot experiments.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO

from .metrics import longest_homopolymer, shannon_entropy
from .schemas import normalize_rna


@dataclass(frozen=True)
class PredictionRecord:
    sequence_id: str
    sequence: str
    label: int
    source: str
    length: int
    test_fold: int
    published_test_folds: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _open_csv_text(path: Path) -> tuple[TextIO, object | None]:
    """Open a plain, gzip, or zip-compressed CSV and return its owner handle."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        archive = zipfile.ZipFile(path)
        members = [
            info
            for info in archive.infolist()
            if info.filename.lower().endswith(".csv")
            and not info.filename.startswith("__MACOSX/")
        ]
        if not members:
            archive.close()
            raise ValueError(f"no CSV member found in {path}")
        member = max(members, key=lambda info: info.file_size)
        raw = archive.open(member)
        return io.TextIOWrapper(raw, encoding="utf-8", newline=""), archive
    if suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline=""), None
    return path.open(encoding="utf-8", newline=""), None


def _length_group(length: int) -> str:
    if length < 174:
        return "<174"
    if length == 174:
        return "174"
    if length <= 200:
        return "175-200"
    return ">200"


def _stable_group_offset(group: tuple[int, str, str], n_folds: int) -> int:
    encoded = "\x1f".join(map(str, group)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % n_folds


def load_ires_ai_records(path: Path, n_folds: int = 10) -> list[PredictionRecord]:
    """Recover unique records and construct a deterministic stratified split.

    The reconstructed fold is stratified by label, source and coarse length
    group.  It is suitable for project smoke tests and random-split pilots, but
    it is not a substitute for the planned sequence-cluster/family split.
    """
    handle, owner = _open_csv_text(path)
    canonical: dict[str, dict[str, object]] = {}
    try:
        reader = csv.DictReader(handle)
        required = {"fold", "type", "ID", "Sequence", "IRES_class_600", "Source"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        for row in reader:
            sequence = normalize_rna(row["Sequence"])
            sequence_id = row["ID"].strip()
            current = {
                "sequence_id": sequence_id,
                "sequence": sequence,
                "label": int(row["IRES_class_600"]),
                "source": row["Source"].strip(),
                "length": int(float(row.get("Length") or len(sequence))),
            }
            previous = canonical.get(sequence_id)
            if previous is None:
                current["published_test_folds"] = []
                current["observed_folds"] = []
                canonical[sequence_id] = current
                previous = current
            elif any(previous[key] != current[key] for key in current):
                raise ValueError(f"inconsistent repeated rows for ID {sequence_id}")
            observed_folds = previous["observed_folds"]
            assert isinstance(observed_folds, list)
            fold = int(row["fold"])
            if fold in observed_folds:
                raise ValueError(f"duplicate long-form row for ID {sequence_id}, fold {fold}")
            observed_folds.append(fold)
            if row["type"].strip().lower() == "test":
                test_folds = previous["published_test_folds"]
                assert isinstance(test_folds, list)
                test_folds.append(fold)
            elif row["type"].strip().lower() != "train":
                raise ValueError(f"unknown split type: {row['type']!r}")
    finally:
        handle.close()
        if owner is not None:
            owner.close()
    if not canonical:
        raise ValueError(f"no records recovered from {path}")
    expected_folds = set(range(n_folds))
    for sequence_id, row in canonical.items():
        if set(row["observed_folds"]) != expected_folds:
            raise ValueError(f"incomplete fold coverage for ID {sequence_id}")

    sequence_to_id: dict[str, str] = {}
    for sequence_id, row in canonical.items():
        sequence = str(row["sequence"])
        if sequence in sequence_to_id and sequence_to_id[sequence] != sequence_id:
            raise ValueError(
                f"identical sequence appears under IDs {sequence_to_id[sequence]} and {sequence_id}"
            )
        sequence_to_id[sequence] = sequence_id

    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in canonical.values():
        group = (int(row["label"]), str(row["source"]), _length_group(int(row["length"])))
        grouped[group].append(row)

    records: list[PredictionRecord] = []
    for group in sorted(grouped):
        offset = _stable_group_offset(group, n_folds)
        ordered = sorted(
            grouped[group],
            key=lambda row: hashlib.sha256(str(row["sequence"]).encode("utf-8")).digest(),
        )
        for index, row in enumerate(ordered):
            published = tuple(sorted(set(int(fold) for fold in row["published_test_folds"])))
            records.append(
                PredictionRecord(
                    sequence_id=str(row["sequence_id"]),
                    sequence=str(row["sequence"]),
                    label=int(row["label"]),
                    source=str(row["source"]),
                    length=int(row["length"]),
                    test_fold=(index + offset) % n_folds,
                    published_test_folds=published,
                )
            )
    records.sort(key=lambda row: row.sequence_id)
    return records


def dataset_audit(records: Iterable[PredictionRecord]) -> dict[str, object]:
    rows = list(records)
    label_counts = Counter(row.label for row in rows)
    source_label: dict[str, Counter[int]] = defaultdict(Counter)
    fold_label: dict[int, Counter[int]] = defaultdict(Counter)
    length_groups: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        source_label[row.source][row.label] += 1
        fold_label[row.test_fold][row.label] += 1
        group = _length_group(row.length)
        length_groups[group][row.label] += 1
    published_counts = Counter(len(row.published_test_folds) for row in rows)
    return {
        "n_sequences": len(rows),
        "n_positive": label_counts[1],
        "n_negative": label_counts[0],
        "positive_prevalence": label_counts[1] / len(rows),
        "n_unique_sequences": len({row.sequence for row in rows}),
        "length_min": min(row.length for row in rows),
        "length_max": max(row.length for row in rows),
        "length_174": sum(row.length == 174 for row in rows),
        "published_split_semantics": "10 repeated stratified 90/10 holdouts; not mutually exclusive K-fold",
        "published_test_assignment_count": {
            str(count): frequency for count, frequency in sorted(published_counts.items())
        },
        "n_never_in_published_test": published_counts[0],
        "source_by_label": {
            source: {str(label): counts[label] for label in (0, 1)}
            for source, counts in sorted(source_label.items())
        },
        "fold_by_label": {
            str(fold): {str(label): counts[label] for label in (0, 1)}
            for fold, counts in sorted(fold_label.items())
        },
        "length_group_by_label": {
            group: {str(label): counts[label] for label in (0, 1)}
            for group, counts in length_groups.items()
        },
    }


def composition_features(sequences: Iterable[str], max_k: int = 3):
    """Return stateless sequence-composition features for a shortcut audit."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in analysis environments
        raise RuntimeError("composition features require numpy") from exc

    alphabet = "ACGU"
    kmers: list[str] = []
    for k in range(1, max_k + 1):
        current = [""]
        for _ in range(k):
            current = [prefix + base for prefix in current for base in alphabet]
        kmers.extend(current)

    rows: list[list[float]] = []
    for raw_sequence in sequences:
        sequence = normalize_rna(raw_sequence)
        length = max(len(sequence), 1)
        values = [
            math.log1p(len(sequence)),
            (sequence.count("G") + sequence.count("C")) / length,
            longest_homopolymer(sequence) / length,
            shannon_entropy(sequence),
        ]
        for kmer in kmers:
            denominator = max(len(sequence) - len(kmer) + 1, 1)
            values.append(sequence.count(kmer) / denominator)
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)
