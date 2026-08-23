#!/usr/bin/env python3
"""Run the released IRESfinder mode-0 model from a modern Python environment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV


SELECTED_FEATURES = [
    0,
    50,
    1774,
    382,
    339,
    872,
    2069,
    197,
    250,
    119,
    98,
    13,
    591,
    1257,
    92,
    1117,
    822,
    258,
    1467,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iresfinder-root", type=Path, required=True)
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--source-commit", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    return [line.strip().lstrip(">") for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    args = parse_args()
    root = args.iresfinder_root.resolve()
    input_fasta = args.input_fasta.resolve()
    train_path = root / "module" / "train.data"
    handle_path = root / "module" / "handlemodel0or1.pl"
    feature_path = root / "module" / "calculate_features.pl"
    for path in (input_fasta, train_path, handle_path, feature_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="iresfinder-") as tmp_name:
        tmp = Path(tmp_name)
        sequences_path = tmp / "input.seq"
        ids_path = tmp / "ids.txt"
        features_path = tmp / "features.txt"
        subprocess.run(
            ["perl", str(handle_path), str(input_fasta), str(sequences_path), str(ids_path)],
            cwd=tmp,
            check=True,
        )
        subprocess.run(
            ["perl", str(feature_path), str(sequences_path), str(features_path)],
            cwd=tmp,
            check=True,
        )
        train = np.loadtxt(train_path)
        training_labels = train[:, 0]
        training_features = train[:, 1:]
        scaler = preprocessing.StandardScaler().fit(training_features)
        scaled_training = scaler.transform(training_features)
        retained = np.flatnonzero((scaled_training > 0).mean(axis=0) > 0.1)
        selected_training = scaled_training[:, retained][:, SELECTED_FEATURES]
        classifier = LogisticRegressionCV(cv=3).fit(selected_training, training_labels)

        prediction_features = np.loadtxt(features_path)
        if prediction_features.ndim == 1:
            prediction_features = prediction_features.reshape(1, -1)
        selected_prediction = scaler.transform(prediction_features)[:, retained][
            :, SELECTED_FEATURES
        ]
        probabilities = classifier.predict_proba(selected_prediction)[:, 1]
        predictions = classifier.predict(selected_prediction).astype(int)
        ids = read_ids(ids_path)

    if not (len(ids) == len(probabilities) == len(predictions)):
        raise ValueError("IRESfinder output row count mismatch")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prediction", "probability"])
        writer.writeheader()
        for sample_id, prediction, probability in zip(ids, predictions, probabilities):
            writer.writerow(
                {
                    "id": sample_id,
                    "prediction": "IRES" if prediction == 1 else "non-IRES",
                    "probability": float(probability),
                }
            )

    manifest = {
        "schema_version": 1,
        "experiment": "iresfinder_released_model_py3_adapter",
        "scope": "classification-only reproduction",
        "source_root": str(root),
        "source_commit": args.source_commit,
        "input_fasta": str(input_fasta),
        "input_fasta_sha256": sha256(input_fasta),
        "train_data_sha256": sha256(train_path),
        "compatibility_note": (
            "Original Perl feature extraction and feature indices; the Python 2.7 orchestration "
            "was ported to the installed scikit-learn API."
        ),
        "n": len(ids),
    }
    manifest_path = args.output_csv.with_suffix(args.output_csv.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
