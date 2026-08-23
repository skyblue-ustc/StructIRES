#!/usr/bin/env python3
"""Create the frozen shared mutation pool used by every primary design arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ires_design.design import make_mutation_pool
from ires_design.io import write_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--candidates-per-parent", type=int, default=512)
    parser.add_argument("--max-edit-distance", type=int, default=10)
    parser.add_argument("--max-edit-fraction", type=float, default=0.03)
    args = parser.parse_args()

    panel = json.loads(args.seed_panel.read_text(encoding="utf-8"))
    records = panel.get("records", [])
    parent_ids = [record["seed_id"] for record in records]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("seed-panel IDs must be unique before pool generation")
    pool = make_mutation_pool(
        [(record["seed_id"], record["sequence"]) for record in records],
        experiment_id=args.experiment_id,
        run_seed=args.run_seed,
        candidates_per_parent=args.candidates_per_parent,
        max_edit_distance=args.max_edit_distance,
        max_edit_fraction=args.max_edit_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(pool, args.output)
    manifest = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "role": "shared_candidate_pool",
        "seed_panel": str(args.seed_panel),
        "seed_panel_sha256": sha256(args.seed_panel),
        "run_seed": args.run_seed,
        "n_parents": len(records),
        "candidates_per_parent": args.candidates_per_parent,
        "candidate_count": len(pool),
        "max_edit_distance": args.max_edit_distance,
        "max_edit_fraction": args.max_edit_fraction,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
