#!/usr/bin/env python3
"""Build a provenance-checked CSV and LaTeX ledger from frozen result artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


METRICS = ("auc", "aupr", "f1", "accuracy", "sensitivity", "specificity", "mcc", "ece10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--external-run-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(frame: pd.DataFrame, selector: dict[str, object]) -> pd.DataFrame:
    selected = frame
    for column, value in selector.items():
        if column not in selected:
            raise ValueError(f"selector column {column!r} is missing")
        selected = selected[selected[column].astype(str) == str(value)]
    if selected.empty:
        raise ValueError(f"selector matched no rows: {selector}")
    return selected


def local_row(entry: dict[str, object], run_root: Path) -> dict[str, object]:
    source = entry["source"]
    if not isinstance(source, dict):
        raise TypeError("local source must be an object")
    run_id = str(source["run_id"])
    run_dir = run_root / run_id
    metric_path = run_dir / str(source["metric_file"])
    manifest_path = run_dir / "run_manifest.json"
    if not metric_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete ledger source run: {run_dir}")
    frame = select_rows(pd.read_csv(metric_path), source.get("selector", {}))
    aggregation = source.get("aggregation", "first")
    if aggregation == "first" and len(frame) != 1:
        raise ValueError(f"first-row source must select exactly one row: {run_id}")
    if aggregation not in {"first", "mean"}:
        raise ValueError(f"unknown aggregation {aggregation!r}")

    row: dict[str, object] = {}
    mapping = source.get("metric_mapping", {})
    std_mapping = source.get("std_mapping", {})
    ci_mapping = source.get("ci_mapping", {})
    for metric in METRICS:
        column = mapping.get(metric)
        if column is not None:
            row[metric] = (
                float(frame.iloc[0][column])
                if aggregation == "first"
                else float(frame[column].mean())
            )
        std_column = std_mapping.get(metric)
        if std_column is not None:
            row[f"{metric}_std"] = float(frame.iloc[0][std_column])
        elif column is not None and aggregation == "mean" and len(frame) > 1:
            row[f"{metric}_std"] = float(frame[column].std(ddof=1))
        interval = ci_mapping.get(metric)
        if interval is not None:
            low_column, high_column = interval
            row[f"{metric}_ci_low"] = float(frame.iloc[0][low_column])
            row[f"{metric}_ci_high"] = float(frame.iloc[0][high_column])
    row.update(
        {
            "run_id": run_id,
            "metric_file": str(source["metric_file"]),
            "metric_file_sha256": sha256(metric_path),
            "run_manifest_sha256": sha256(manifest_path),
        }
    )
    return row


def build_rows(config: dict[str, object], run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    entries = config.get("entries")
    if not isinstance(entries, list):
        raise TypeError("config entries must be a list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("each config entry must be an object")
        common = {
            "result_id": entry["result_id"],
            "model": entry["model"],
            "result_status": entry["result_status"],
            "evaluation_regime": entry["evaluation_regime"],
            "assay": entry["assay"],
            "note": entry.get("note", ""),
        }
        if entry["result_status"] == "published_reference":
            metrics = entry.get("metrics", {})
            row = {metric: metrics.get(metric, "") for metric in METRICS}
            row.update(
                {
                    "run_id": "",
                    "metric_file": "",
                    "metric_file_sha256": "",
                    "run_manifest_sha256": "",
                }
            )
        else:
            row = local_row(entry, run_root)
        rows.append({**common, **row})
    return rows


def latex_escape(value: object) -> str:
    text = str(value)
    for original, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(original, replacement)
    return text


def formatted_metric(row: dict[str, object], metric: str) -> str:
    value = row.get(metric, "")
    if value == "" or pd.isna(value):
        return "--"
    result = f"{float(value):.3f}"
    std = row.get(f"{metric}_std", "")
    if std != "" and not pd.isna(std):
        result += rf"$\pm${float(std):.3f}"
    low = row.get(f"{metric}_ci_low", "")
    high = row.get(f"{metric}_ci_high", "")
    if low != "" and high != "" and not pd.isna(low) and not pd.isna(high):
        result += f" ({float(low):.3f}--{float(high):.3f})"
    return result


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "% Generated by scripts/build_unified_result_ledger.py; do not edit manually.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Unified IRES classification ledger. Published references, local native-split",
        r"reruns, similarity-aware evaluation, and direct-RNA transfer are intentionally distinct.",
        r"Parentheses are 95\% stratified-bootstrap intervals; $\pm$ is the sample standard",
        r"deviation across released repeated holdouts, which are not independent.}",
        r"\label{tab:unified-results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrrrrrr}",
        r"\toprule",
        r"Status & Regime & Model & Assay & AUC & AUPR & F1 & Acc. & Sens. & Spec. & MCC & ECE \\",
        r"\midrule",
    ]
    for row in rows:
        values = [
            latex_escape(row["result_status"]),
            latex_escape(row["evaluation_regime"]),
            latex_escape(row["model"]),
            latex_escape(row["assay"]),
            *(formatted_metric(row, metric) for metric in METRICS),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = build_rows(config, args.external_run_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    write_latex(args.output_tex, rows)
    print(json.dumps({"rows": len(rows), "output_csv": str(args.output_csv)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
