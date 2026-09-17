#!/usr/bin/env python3
"""Build the protocol-matched BIBE Table I from frozen model predictions."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--iresfinder-predictions", type=Path, required=True)
    parser.add_argument("--deepires-template", required=True)
    parser.add_argument("--utrlm-template", required=True)
    parser.add_argument("--rnafm-template", required=True)
    parser.add_argument("--structires-template", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_template(template: str, fold: int) -> Path:
    return Path(template.format(fold=fold))


def load_dataset(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [
            name for name in archive.namelist()
            if name.endswith(".csv") and "__MACOSX" not in name
        ]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in dataset archive, found {members}")
        with archive.open(members[0]) as handle:
            return pd.read_csv(
                handle,
                usecols=["fold", "type", "idx", "ID", "Sequence", "IRES_class_600"],
            )


def metric_row(label: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "AUROC": float(roc_auc_score(label, probability)),
        "AUPR": float(average_precision_score(label, probability)),
        "F1": float(f1_score(label, prediction, zero_division=0)),
        "MCC": float(matthews_corrcoef(label, prediction)),
        "Accuracy": float(accuracy_score(label, prediction)),
    }


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    data = load_dataset(args.dataset)
    iresfinder = pd.read_csv(args.iresfinder_predictions)
    iresfinder["idx"] = iresfinder["id"].str.removeprefix("idx_").astype(int)
    iresfinder_map = iresfinder.set_index("idx")["probability"]

    model_templates = {
        "IRESfinder": None,
        "UTR-LM": args.utrlm_template,
        "DeepIRES": args.deepires_template,
        "RNA-FM": args.rnafm_template,
        "StructIRES": args.structires_template,
    }
    canonical_rows: list[dict[str, object]] = []
    aligned_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    input_audit: list[dict[str, object]] = []

    for fold in folds:
        frame = data.loc[
            data["fold"] == fold,
            ["type", "idx", "ID", "Sequence", "IRES_class_600"],
        ].copy()
        frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
        if frame.ID.duplicated().any() or len(frame) != 46774:
            raise ValueError(f"unexpected fold {fold} record set")
        label = frame.IRES_class_600.to_numpy(dtype=np.int64)
        upstream_train = np.flatnonzero(frame.type.to_numpy() == "train")
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=args.validation_fraction, random_state=args.seed
        )
        _, validation_local = next(splitter.split(upstream_train, label[upstream_train]))
        expected = upstream_train[validation_local]

        reference_path = format_template(args.rnafm_template, fold)
        reference = pd.read_csv(reference_path)
        observed = reference.sample_index.to_numpy(dtype=np.int64)
        if set(observed.tolist()) != set(expected.tolist()):
            raise ValueError(f"RNA-FM predictions do not match reconstructed fold {fold} validation IDs")
        validation = observed
        canonical = frame.iloc[validation].copy()
        canonical["sample_index"] = validation
        canonical["fold"] = fold
        canonical["label"] = canonical.IRES_class_600.astype(int)
        canonical["sequence_id"] = canonical.ID.astype(str)
        canonical["sequence_sha256"] = canonical.Sequence.astype(str).map(
            lambda value: hashlib.sha256(value.upper().replace("T", "U").encode("ascii")).hexdigest()
        )
        canonical_rows.extend(
            canonical[["fold", "sample_index", "idx", "sequence_id", "label", "sequence_sha256"]]
            .to_dict("records")
        )
        canonical_by_index = canonical.set_index("sample_index")

        for model, template in model_templates.items():
            if model == "IRESfinder":
                probability = canonical.idx.map(iresfinder_map)
                same_ids = not probability.isna().any()
                same_labels = True
                model_index = validation
                model_label = canonical.label.to_numpy(dtype=np.int64)
                model_probability = probability.to_numpy(dtype=np.float64)
                input_path = args.iresfinder_predictions
                training_protocol = "fixed released IRESfinder model"
            else:
                input_path = format_template(str(template), fold)
                prediction = pd.read_csv(input_path)
                model_index = prediction.sample_index.to_numpy(dtype=np.int64)
                model_label = prediction.label.to_numpy(dtype=np.int64)
                model_probability = prediction.probability.to_numpy(dtype=np.float64)
                same_ids = set(model_index.tolist()) == set(validation.tolist()) and len(model_index) == len(validation)
                if same_ids:
                    order = {int(index): offset for offset, index in enumerate(model_index)}
                    take = np.asarray([order[int(index)] for index in validation], dtype=np.int64)
                    model_index = model_index[take]
                    model_label = model_label[take]
                    model_probability = model_probability[take]
                same_labels = bool(
                    same_ids
                    and np.array_equal(model_label, canonical.label.to_numpy(dtype=np.int64))
                )
                training_protocol = {
                    "UTR-LM": "native UTR-LM training on canonical inner-train partition",
                    "DeepIRES": "native DeepIRES training on canonical inner-train partition",
                    "RNA-FM": "matched RNA-FM training on canonical inner-train partition",
                    "StructIRES": "matched StructIRES training on canonical inner-train partition",
                }[model]
            safe = bool(same_ids and same_labels)
            checks.append(
                {
                    "Model": model,
                    "Fold": fold,
                    "Same evaluation IDs?": "MATCH" if same_ids else "MISMATCH",
                    "Same labels?": "MATCH" if same_labels else "MISMATCH",
                    "Training protocol": training_protocol,
                    "Threshold protocol": f"fixed {args.threshold:g} for unified Table I",
                    "Safe for Table I?": "YES" if safe else "NO",
                }
            )
            input_audit.append(
                {"model": model, "fold": fold, "path": str(input_path.resolve()), "sha256": sha256(input_path)}
            )
            if not safe:
                continue
            values = metric_row(canonical.label.to_numpy(dtype=np.int64), model_probability, args.threshold)
            metric_rows.append(
                {
                    "Model": model,
                    "Fold": fold,
                    "N": len(canonical),
                    "Pos": int(canonical.label.sum()),
                    "Neg": int(len(canonical) - canonical.label.sum()),
                    **values,
                }
            )
            for sequence_id, lab, score in zip(canonical.sequence_id, model_label, model_probability):
                aligned_rows.append(
                    {
                        "Model": model,
                        "Fold": fold,
                        "sequence_id": sequence_id,
                        "label": int(lab),
                        "probability": float(score),
                    }
                )

    per_fold = pd.DataFrame(metric_rows)
    aggregate_rows = []
    paper_rows = []
    metrics = ("AUROC", "AUPR", "F1", "MCC", "Accuracy")
    for model in model_templates:
        subset = per_fold.loc[per_fold.Model == model]
        if len(subset) != len(folds):
            continue
        row: dict[str, object] = {"Model": model, "N folds": len(subset)}
        paper: dict[str, object] = {"Model": model}
        for metric in metrics:
            mean = float(subset[metric].mean())
            sd = float(subset[metric].std(ddof=1))
            row[f"{metric} mean"] = mean
            row[f"{metric} sample SD"] = sd
            paper[metric] = f"{mean:.4f} ± {sd:.4f}"
        aggregate_rows.append(row)
        paper_rows.append(paper)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "canonical_evaluation_manifest.csv": pd.DataFrame(canonical_rows),
        "per_fold_metrics.csv": per_fold,
        "aggregate_metrics.csv": pd.DataFrame(aggregate_rows),
        "protocol_check.csv": pd.DataFrame(checks),
        "paper_table.csv": pd.DataFrame(paper_rows),
    }
    output_audit = []
    for name, table in outputs.items():
        path = args.output_dir / name
        table.to_csv(path, index=False)
        output_audit.append({"path": str(path.resolve()), "sha256": sha256(path)})
    aligned_path = args.output_dir / "aligned_predictions.csv.gz"
    with gzip.open(aligned_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aligned_rows[0]))
        writer.writeheader()
        writer.writerows(aligned_rows)
    output_audit.append({"path": str(aligned_path.resolve()), "sha256": sha256(aligned_path)})
    manifest = {
        "schema_version": 1,
        "experiment": "protocol_matched_ires_recognition_table1",
        "folds": folds,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "evaluation_reference": "StructIRES canonical inner-validation split",
        "table_threshold": args.threshold,
        "threshold_note": "All threshold-dependent Table I metrics use fixed 0.5; AUROC and AUPR use continuous probabilities.",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "inputs": input_audit,
        "outputs": output_audit,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "models": len(paper_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
