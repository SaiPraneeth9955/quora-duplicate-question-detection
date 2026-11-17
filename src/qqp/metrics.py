"""Threshold selection and evaluation metrics shared by every model."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return the score cut-off that maximises F1 on the given labelled set."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # The final precision/recall point has no associated threshold.
    return float(thresholds[np.argmax(f1[:-1])])


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Metrics for scores binarised at ``threshold``. ROC-AUC is threshold-free."""
    predictions = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
