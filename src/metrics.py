"""Classification metrics computed from the confusion matrix."""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)``."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, precision, recall and F1 plus the raw confusion counts."""
    tp, fp, fn, tn = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision, "recall": recall, "f1": f1,
    }


def fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    """F-beta score. ``beta > 1`` weights recall more than precision."""
    denom = beta ** 2 * precision + recall
    return (1 + beta ** 2) * precision * recall / denom if denom else 0.0
