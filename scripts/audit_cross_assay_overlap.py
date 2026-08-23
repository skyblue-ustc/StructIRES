#!/usr/bin/env python3
"""Audit sequence overlap between IRES-AI and direct-RNA IRES-TrAPPr data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ires_design.assay_audit import run_cross_assay_overlap_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-data", type=Path, required=True)
    parser.add_argument("--direct-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_cross_assay_overlap_audit(args.legacy_data, args.direct_data, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
