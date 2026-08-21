"""Portable FASTA, CSV, and candidate-JSONL I/O."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .schemas import CandidateRecord, normalize_rna


FASTA_SUFFIXES = {".fa", ".fasta", ".fna", ".fas"}
SEQUENCE_COLUMN_CANDIDATES = ("sequence", "seq", "rna", "ires_sequence", "MT")


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    name: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, normalize_rna("".join(chunks))
                name = line[1:].strip() or f"record_{len(chunks)}"
                chunks = []
            else:
                if name is None:
                    raise ValueError(f"FASTA sequence found before header in {path}")
                chunks.append(line)
    if name is not None:
        yield name, normalize_rna("".join(chunks))


def _read_csv_sequences(path: Path, sequence_column: str | None) -> Iterator[tuple[str, str, dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        column = sequence_column
        if column is None:
            column = next((name for name in SEQUENCE_COLUMN_CANDIDATES if name in reader.fieldnames), None)
        if column is None or column not in reader.fieldnames:
            raise ValueError(
                f"Cannot identify sequence column in {path}; choose --sequence-column from {reader.fieldnames}"
            )
        for index, row in enumerate(reader):
            sequence = normalize_rna(row.get(column, ""))
            record_id = row.get("candidate_id") or row.get("id") or f"row_{index:06d}"
            metadata = {key: value for key, value in row.items() if key != column and value not in (None, "")}
            yield str(record_id), sequence, metadata


def load_external_records(
    path: Path,
    *,
    method: str,
    task: str,
    seed: int,
    sequence_column: str | None = None,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    if path.suffix.lower() in FASTA_SUFFIXES:
        source_rows = ((name, sequence, {}) for name, sequence in iter_fasta(path))
    elif path.suffix.lower() == ".csv":
        source_rows = _read_csv_sequences(path, sequence_column)
    else:
        raise ValueError(f"Unsupported input extension: {path.suffix}")

    for index, (source_id, sequence, metadata) in enumerate(source_rows):
        records.append(
            CandidateRecord(
                candidate_id=f"{method}__s{seed}__{index:06d}",
                sequence=sequence,
                method=method,
                task=task,
                seed=seed,
                parent_id=source_id if task == "mutation" else None,
                source_path=str(path),
                metadata={"source_id": source_id, **metadata},
            )
        )
    return records


def write_jsonl(records: Iterable[CandidateRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[CandidateRecord]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                yield CandidateRecord(**payload)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid candidate JSONL at {path}:{line_number}: {exc}") from exc

