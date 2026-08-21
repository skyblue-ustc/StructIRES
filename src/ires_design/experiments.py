"""Validation for frozen experiment contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .baselines import DEFAULT_REGISTRY, registry_by_id


VALID_VARIANTS = {"raw", "function_only", "mfe", "structure", "full"}


def load_and_validate_experiment(
    path: Path, registry_path: Path = DEFAULT_REGISTRY
) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("task") not in {"de_novo", "mutation"}:
        errors.append("task must be de_novo or mutation")
    if not payload.get("experiment_id"):
        errors.append("experiment_id is required")
    if not isinstance(payload.get("seeds"), list) or not payload.get("seeds"):
        errors.append("seeds must be a non-empty list")
    unknown_variants = set(payload.get("variants", [])) - VALID_VARIANTS
    if unknown_variants:
        errors.append(f"unknown variants: {sorted(unknown_variants)}")

    registry = registry_by_id(registry_path)
    baseline_ids = payload.get("baseline_ids", [])
    unknown_baselines = set(baseline_ids) - set(registry)
    if unknown_baselines:
        errors.append(f"unknown baseline IDs: {sorted(unknown_baselines)}")
    for baseline_id in set(baseline_ids) & set(registry):
        if payload.get("task") not in registry[baseline_id].tasks:
            errors.append(f"baseline {baseline_id} does not support task {payload.get('task')}")

    controlled = set(payload.get("controlled_baseline_ids", []))
    if not controlled <= set(baseline_ids):
        errors.append("controlled_baseline_ids must be a subset of baseline_ids")

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("protocol must be an object")
    elif payload.get("task") == "de_novo":
        for key in ("candidate_pool_size", "evaluation_size", "oracle_query_budget"):
            if not isinstance(protocol.get(key), int) or protocol[key] <= 0:
                errors.append(f"protocol.{key} must be a positive integer")
        if protocol.get("evaluation_size", 0) > protocol.get("candidate_pool_size", 0):
            errors.append("evaluation_size cannot exceed candidate_pool_size")
    elif payload.get("task") == "mutation":
        for key in ("max_edit_distance", "oracle_query_budget_per_seed"):
            if not isinstance(protocol.get(key), int) or protocol[key] <= 0:
                errors.append(f"protocol.{key} must be a positive integer")

    if errors:
        raise ValueError("Invalid experiment config:\n- " + "\n- ".join(errors))
    return payload

