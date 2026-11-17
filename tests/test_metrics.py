import numpy as np

from qqp.metrics import best_f1_threshold, classification_metrics


def test_threshold_separates_perfectly_separable_scores():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = best_f1_threshold(labels, scores)
    assert 0.3 < threshold <= 0.7
    metrics = classification_metrics(labels, scores, threshold)
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[3, 0], [0, 2]]


def test_threshold_maximises_f1_over_all_cutoffs():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    scores = labels * 0.3 + rng.random(500)
    chosen = classification_metrics(labels, scores, best_f1_threshold(labels, scores))["f1"]
    best = max(classification_metrics(labels, scores, t)["f1"] for t in np.unique(scores))
    assert chosen == best


def test_confusion_matrix_layout_is_tn_fp_fn_tp():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.1, 0.9, 0.1])
    assert classification_metrics(labels, scores, 0.5)["confusion_matrix"] == [[1, 1], [1, 1]]
