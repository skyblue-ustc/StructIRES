#!/usr/bin/env python3
"""Validate the small, sequence-free BIBE 2026 release snapshot."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "bibe2026"
METRICS = ("AUROC", "AUPR", "F1", "MCC", "Accuracy")


def read_csv(name: str) -> list[dict[str, str]]:
    with (RELEASE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    per_fold = read_csv("table1_per_fold_metrics.csv")
    aggregate = read_csv("table1_aggregate_metrics.csv")
    assert len(per_fold) == 15, "expected five models by three folds"

    by_model: dict[str, list[dict[str, str]]] = {}
    for row in per_fold:
        assert int(row["N"]) == 6315
        assert int(row["Pos"]) == 1238
        assert int(row["Neg"]) == 5077
        assert int(row["Pos"]) + int(row["Neg"]) == int(row["N"])
        by_model.setdefault(row["Model"], []).append(row)

    assert set(by_model) == {
        "IRESfinder",
        "UTR-LM",
        "DeepIRES",
        "RNA-FM",
        "StructIRES",
    }
    aggregate_by_model = {row["Model"]: row for row in aggregate}

    for model, rows in by_model.items():
        assert sorted(int(row["Fold"]) for row in rows) == [0, 1, 2]
        reported = aggregate_by_model[model]
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            assert close(statistics.mean(values), float(reported[f"{metric}_mean"]))
            assert close(
                statistics.stdev(values),
                float(reported[f"{metric}_sample_SD"]),
            )

    candidate_rows = read_csv("candidate_selection_summary.csv")
    assert len(candidate_rows) == 5
    assert {row["Method"] for row in candidate_rows} == {
        "adapter_score_only",
        "adapter_plus_energy",
        "adapter_plus_ensemble",
        "adapter_plus_anchor",
        "structires_adapter_rank",
    }

    with (RELEASE / "provenance.json").open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    assert provenance["candidate_selection"]["activity_labels_used_for_selection"] is False
    assert "generated candidate sequences" in provenance["excluded_from_git"]

    print("PASS: BIBE 2026 minimal release snapshot is internally consistent")


if __name__ == "__main__":
    main()
