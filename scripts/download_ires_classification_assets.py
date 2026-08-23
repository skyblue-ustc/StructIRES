#!/usr/bin/env python3
"""Download the released Google Drive assets required for IRES-LM classification."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--include-role", action="append", default=[])
    parser.add_argument("--include-name", action="append", default=[])
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    import gdown

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    all_assets = [row for row in manifest["assets"] if row["provider"] == "google_drive"]
    if args.include_role:
        all_assets = [row for row in all_assets if row["role"] in set(args.include_role)]
    if args.include_name:
        all_assets = [row for row in all_assets if row["name"] in set(args.include_name)]
    assets = [
        row for index, row in enumerate(all_assets) if index % args.num_shards == args.shard_index
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def download_one(asset: dict[str, object]) -> dict[str, object]:
        destination = args.output_dir / str(asset["name"])
        error = "download returned no file"
        for attempt in range(1, args.max_attempts + 1):
            try:
                result = gdown.download(
                    id=str(asset["id"]),
                    output=str(destination),
                    quiet=False,
                    proxy=args.proxy,
                    use_cookies=False,
                    resume=True,
                )
                if result is not None and destination.is_file():
                    break
                error = "download returned no file"
            except Exception as exc:  # network/proxy failures are retried and recorded
                error = f"{type(exc).__name__}: {exc}"
            print(
                f"retry {attempt}/{args.max_attempts} for {asset['name']}: {error}",
                flush=True,
            )
            time.sleep(5 * attempt)
        else:
            return {**asset, "status": "failed", "error": error}
        return {
            **asset,
            "status": "complete",
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }

    results: list[dict[str, object]] = []
    if args.include_name:
        role_tag = "names_" + "_".join(sorted(Path(name).stem for name in args.include_name))
    else:
        role_tag = "_".join(sorted(args.include_role)) if args.include_role else "all"
    output_tag = f"shard{args.shard_index}_{role_tag}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, asset): asset for asset in assets}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            (args.output_dir / f"download_manifest.{output_tag}.partial.json").write_text(
                json.dumps(sorted(results, key=lambda row: str(row["name"])), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    results.sort(key=lambda row: str(row["name"]))
    summary = {
        "schema_version": 1,
        "source_manifest": str(args.manifest),
        "requested": len(assets),
        "complete": sum(row["status"] == "complete" for row in results),
        "failed": sum(row["status"] != "complete" for row in results),
        "assets": results,
    }
    (args.output_dir / f"download_manifest.{output_tag}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("requested", "complete", "failed")}, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
