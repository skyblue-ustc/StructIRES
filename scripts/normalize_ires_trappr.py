#!/usr/bin/env python3
"""Normalize IRES-TrAPPr supplementary workbooks into assay-labelled CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ires_design.trappr import write_normalized_trappr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = write_normalized_trappr(args.input_dir, args.output_dir)
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
