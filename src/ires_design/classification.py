"""Shared binary-classification metrics for frozen IRES evaluations."""

from __future__ import annotations

from collections.abc import Sequence


def _arrays(labels: Sequence[int], probabilities: Sequence[float]):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - analysis dependency
        raise RuntimeError("classification metrics require numpy") from exc

    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("labels and probabilities must be equal-length one-dimensional arrays")
    if not len(y_true):
        raise ValueError("at least one prediction is required")
    if not set(y_true.tolist()) <= {0, 1}:
        raise ValueError("labels must be binary")
    if not np.isfinite(y_score).all() or ((y_score < 0.0) | (y_score > 1.0)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return np, y_true, y_score


def expected_calibration_error(
    labels: Sequence[int], probabilities: Sequence[float], bins: int = 10
) -> float:
    """Return fixed-width positive-class expected calibration error."""
    if bins < 1:
        raise ValueError("bins must be positive")
    np, y_true, y_score = _arrays(labels, probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        upper = y_score <= edges[index + 1] if index == bins - 1 else y_score < edges[index + 1]
        selected = (y_score >= edges[index]) & upper
        if selected.any():
            error += float(selected.mean()) * abs(
                float(y_true[selected].mean()) - float(y_score[selected].mean())
            )
    return error


def binary_classification_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5
) -> dict[str, float | int]:
    """Compute the project's complete frozen binary-classification metric set."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    np, y_true, y_score = _arrays(labels, probabilities)
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUROC and AUPRC require both classes")
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        recall_score,
        roc_auc_score,
    )

    predicted = (y_score >= threshold).astype(int)
    negatives = y_true == 0
    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, y_score)),
        "aupr": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float((predicted[negatives] == 0).mean()),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "ece10": expected_calibration_error(y_true, y_score, bins=10),
        "decision_threshold": float(threshold),
    }


def stratified_bootstrap_intervals(
    labels: Sequence[int],
    probabilities: Sequence[float],
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Return deterministic 95% stratified-bootstrap AUROC/AUPRC intervals."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    np, y_true, y_score = _arrays(labels, probabilities)
    class_indices = [np.flatnonzero(y_true == value) for value in (0, 1)]
    if any(not len(indices) for indices in class_indices):
        raise ValueError("stratified bootstrap requires both classes")
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    aucs = np.empty(replicates)
    auprs = np.empty(replicates)
    for index in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        aucs[index] = roc_auc_score(y_true[sampled], y_score[sampled])
        auprs[index] = average_precision_score(y_true[sampled], y_score[sampled])
    return {
        "auc_ci_low": float(np.quantile(aucs, 0.025)),
        "auc_ci_high": float(np.quantile(aucs, 0.975)),
        "aupr_ci_low": float(np.quantile(auprs, 0.025)),
        "aupr_ci_high": float(np.quantile(auprs, 0.975)),
        "bootstrap_replicates": int(replicates),
    }
