"""Canonical baseline registry and local-asset checks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "configs" / "baselines.json"
VALID_ROLES = {
    "proposal_backbone",
    "paired_backbone",
    "strict_task_baseline",
    "external_reference",
    "lower_bound",
    "shared_optimizer",
    "distribution_anchor",
    "supplementary_reproduction",
}
VALID_ADAPTERS = {"builtin", "released_output", "planned"}


@dataclass(frozen=True)
class BaselineSpec:
    id: str
    name: str
    kind: str
    tasks: tuple[str, ...]
    role: str
    source_url: str | None
    paper_url: str | None
    license_note: str
    asset_env: str | None
    default_asset_path: str | None
    adapter: str
    fairness_note: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineSpec":
        data = dict(payload)
        data["tasks"] = tuple(data.get("tasks", ()))
        spec = cls(**data)
        if spec.role not in VALID_ROLES:
            raise ValueError(f"Unknown role for {spec.id}: {spec.role}")
        if spec.adapter not in VALID_ADAPTERS:
            raise ValueError(f"Unknown adapter state for {spec.id}: {spec.adapter}")
        if not spec.tasks or any(task not in {"de_novo", "mutation"} for task in spec.tasks):
            raise ValueError(f"Invalid tasks for {spec.id}: {spec.tasks}")
        return spec

    def asset_path(self, repository_root: Path = REPOSITORY_ROOT) -> Path | None:
        configured = os.environ.get(self.asset_env) if self.asset_env else None
        raw_path = configured or self.default_asset_path
        if raw_path is None:
            return None
        path = Path(raw_path).expanduser()
        return path if path.is_absolute() else repository_root / path

    def availability(self, repository_root: Path = REPOSITORY_ROOT) -> tuple[str, Path | None]:
        if self.adapter == "builtin":
            return "ready", None
        if self.adapter == "planned":
            return "adapter_pending", self.asset_path(repository_root)
        path = self.asset_path(repository_root)
        return ("ready" if path and path.exists() else "asset_missing"), path


def load_registry(path: Path = DEFAULT_REGISTRY) -> list[BaselineSpec]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported baseline registry schema in {path}")
    specs = [BaselineSpec.from_dict(row) for row in payload.get("baselines", [])]
    ids = [spec.id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Baseline IDs must be unique")
    return specs


def registry_by_id(path: Path = DEFAULT_REGISTRY) -> dict[str, BaselineSpec]:
    return {spec.id: spec for spec in load_registry(path)}
