from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
    except ValueError:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        auc_value = 0.5
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc_value,
        "fpr": fpr,
        "tpr": tpr,
        "roc_curve_auc": auc(fpr, tpr),
        "confusion_matrix": cm,
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "balanced_accuracy",
) -> tuple[float, Dict]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, y_prob, threshold=best_threshold)
    best_score = float(best_metrics.get(metric, best_metrics["balanced_accuracy"]))

    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
        score = float(metrics.get(metric, metrics["balanced_accuracy"]))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    best_metrics["threshold"] = best_threshold
    best_metrics["threshold_metric"] = metric
    return best_threshold, best_metrics
