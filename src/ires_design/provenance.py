"""Create compact, machine-readable run manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def build_run_manifest(config_path: Path, repository_root: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config.get("experiment_id"),
        "task": config.get("task"),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git_commit": git_commit(repository_root),
        "python": sys.version,
        "platform": platform.platform(),
        "seeds": config.get("seeds"),
        "baseline_ids": config.get("baseline_ids"),
        "status": "initialized",
    }

