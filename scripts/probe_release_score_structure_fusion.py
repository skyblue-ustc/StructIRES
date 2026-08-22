#!/usr/bin/env python3
"""Validation-clean calibrated RNA-FM versus RNA-FM+structure diagnostic.

This is an intentionally transparent complementarity check.  A fixed released
RNA-FM checkpoint scores every record in one official fold.  Logistic heads
are fitted only on a deterministic subset of the upstream training records:
one receives the released score alone and the other receives that score plus
the label-free 21-dimensional ViennaRNA cache.  Regularization and F1
thresholds are selected on the held-out validation subset, never the official
fold test.  It is a diagnostic baseline, not the proposed neural fusion.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rnafm-base", type=Path, required=True)
    parser.add_argument("--upstream-fm-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    parser.add_argument("--truncate-num", type=int, default=1024)
    parser.add_argument("--c-values", type=float, nargs="+", default=(.01, .1, 1., 10., 100.))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_adapter_module(path: Path):
    spec = importlib.util.spec_from_file_location("structires_profile_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load profile-adapter utility module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_model(x_train, y_train, x_validation, y_validation, c_values, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_train)
    curve, selected = [], None
    for c_value in c_values:
        model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(scaler.transform(x_train), y_train)
        probability = model.predict_proba(scaler.transform(x_validation))[:, 1]
        aupr = float(average_precision_score(y_validation, probability))
        curve.append({"C": float(c_value), "validation_aupr": aupr})
        if selected is None or aupr > selected[0]:
            selected = (aupr, float(c_value))
    assert selected is not None
    _, c_value = selected
    model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=seed)
    model.fit(scaler.transform(x_train), y_train)
    return model, scaler, selected, curve


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output_dir}")
    from sklearn.model_selection import StratifiedShuffleSplit

    script = Path(__file__).with_name("train_structires_release_profile_adapter.py")
    utility = load_adapter_module(script)
    frame = utility.load_fold_rows(args.dataset, args.fold)
    frame = frame.iloc[np.argsort(frame.ID.astype(str).to_numpy())].reset_index(drop=True)
    cache = np.load(args.feature_cache, allow_pickle=False)
    feature_by_id = {identifier: index for index, identifier in enumerate(cache["sequence_ids"].astype(str))}
    if set(frame.ID.astype(str)) != set(feature_by_id):
        raise ValueError("dataset and feature-cache IDs differ")
    feature = np.asarray(cache["features"], dtype=np.float64)[
        np.asarray([feature_by_id[identifier] for identifier in frame.ID.astype(str)], dtype=np.int64)
    ]
    labels = frame.label.to_numpy(dtype=np.int64)
    sequence = frame.Sequence.tolist()
    data_module, pretrained = utility.load_fm(args.upstream_fm_dir)
    backbone, alphabet = pretrained.rna_fm_t12(str(args.rnafm_base))
    model = utility.ReleasedRNAFM(backbone)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    device = torch.device(args.device)
    model = model.to(device).eval()
    all_index = np.arange(len(frame), dtype=np.int64)
    loader = utility.make_loader(data_module, alphabet, all_index, sequence, args.tokens_per_batch)
    ordered_index, batched_probability = utility.probabilities(
        model, loader, None, device, adapter=False, truncate_num=args.truncate_num
    )
    if sorted(ordered_index.tolist()) != all_index.tolist():
        raise RuntimeError("released predictor returned an invalid record index set")
    # FastaBatchedDataset length-sorts batches for efficiency.  Restore the
    # canonical ID order explicitly rather than treating this benign batching
    # optimization as a data-integrity failure.
    released_probability = np.empty(len(frame), dtype=np.float64)
    released_probability[ordered_index] = batched_probability
    released_logit = np.log(np.clip(released_probability, 1e-6, 1 - 1e-6) /
                            np.clip(1 - released_probability, 1e-6, 1 - 1e-6))
    upstream_train = np.flatnonzero(frame.type.to_numpy() == "train")
    test = np.flatnonzero(frame.type.to_numpy() == "test")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    local_train, local_validation = next(splitter.split(upstream_train, labels[upstream_train]))
    train, validation = upstream_train[local_train], upstream_train[local_validation]
    # The raw released checkpoint is a fixed reference.  The two fitted heads
    # have the identical grid/selection procedure; only structural features differ.
    raw_threshold = utility.best_f1_threshold(labels[validation], released_probability[validation])
    raw_metrics = utility.metrics(labels[test], released_probability[test], raw_threshold)
    score_x = released_logit[:, None]
    fusion_x = np.column_stack((released_logit, feature))
    score_model, score_scaler, score_selected, score_curve = select_model(
        score_x[train], labels[train], score_x[validation], labels[validation], args.c_values, args.seed
    )
    fusion_model, fusion_scaler, fusion_selected, fusion_curve = select_model(
        fusion_x[train], labels[train], fusion_x[validation], labels[validation], args.c_values, args.seed
    )
    score_validation = score_model.predict_proba(score_scaler.transform(score_x[validation]))[:, 1]
    fusion_validation = fusion_model.predict_proba(fusion_scaler.transform(fusion_x[validation]))[:, 1]
    score_probability = score_model.predict_proba(score_scaler.transform(score_x[test]))[:, 1]
    fusion_probability = fusion_model.predict_proba(fusion_scaler.transform(fusion_x[test]))[:, 1]
    score_metrics = utility.metrics(labels[test], score_probability, utility.best_f1_threshold(labels[validation], score_validation))
    fusion_metrics = utility.metrics(labels[test], fusion_probability, utility.best_f1_threshold(labels[validation], fusion_validation))
    args.output_dir.mkdir(parents=True)
    report = {"fold": args.fold, "released_checkpoint": raw_metrics,
              "score_only_validation_selected": score_metrics,
              "score_structure_validation_selected": fusion_metrics,
              "score_only_selected_C": score_selected[1], "score_only_best_validation_aupr": score_selected[0],
              "score_structure_selected_C": fusion_selected[1], "score_structure_best_validation_aupr": fusion_selected[0],
              "score_only_validation_curve": score_curve, "score_structure_validation_curve": fusion_curve}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with gzip.open(args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "label", "released_probability", "score_only_probability", "score_structure_probability"))
        writer.writeheader()
        for index, raw, score, fusion in zip(test, released_probability[test], score_probability, fusion_probability):
            writer.writerow({"id": frame.ID.iloc[int(index)], "label": int(labels[index]),
                             "released_probability": float(raw), "score_only_probability": float(score),
                             "score_structure_probability": float(fusion)})
    manifest = {"schema_version": 1, "experiment": "released_rnafm_score_structure_logistic_diagnostic",
                "fold": args.fold, "seed": args.seed, "validation_fraction": args.validation_fraction,
                "dataset_sha256": sha256(args.dataset), "feature_cache_sha256": sha256(args.feature_cache),
                "checkpoint_sha256": sha256(args.checkpoint), "rnafm_base_sha256": sha256(args.rnafm_base),
                "selection": "validation AUPR", "test_labels_used_for_selection": False,
                "feature_names": cache["feature_names"].astype(str).tolist()}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
