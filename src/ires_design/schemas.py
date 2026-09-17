"""Shared, dependency-light schemas used by every baseline adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_TASKS = {"de_novo", "mutation"}


def normalize_rna(sequence: str) -> str:
    """Return an uppercase RNA sequence while preserving non-alphabet symbols for auditing."""
    return "".join(str(sequence).split()).upper().replace("T", "U")


@dataclass(frozen=True)
class CandidateRecord:
    """Canonical record emitted by all generation and mutation methods."""

    candidate_id: str
    sequence: str
    method: str
    task: str
    seed: int
    parent_id: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.method:
            raise ValueError("method must not be empty")
        if self.task not in VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
        object.__setattr__(self, "sequence", normalize_rna(self.sequence))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

