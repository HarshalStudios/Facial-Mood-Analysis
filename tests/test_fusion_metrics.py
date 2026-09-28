from __future__ import annotations

import numpy as np
import pytest

from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.fusion.metrics import compute_split_metrics, sanity_check_predictions


def test_compute_split_metrics_perfect_predictions():
    y_true = np.array([0, 1, 2, 3, 4, 5, 6])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6])

    metrics = compute_split_metrics("validation", y_true, y_pred)

    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.weighted_f1 == 1.0
    assert metrics.confusion_matrix_labels == list(CANONICAL_EMOTION_LABELS)
    assert np.array_equal(np.array(metrics.confusion_matrix), np.eye(7, dtype=int))


def test_compute_split_metrics_confusion_matrix_uses_canonical_order_not_alphabetical():
    # angry=0, neutral=6 in canonical order. If sklearn's default ordering
    # (alphabetical over observed labels) leaked in, this matrix would be
    # transposed/misaligned relative to CANONICAL_EMOTION_LABELS.
    y_true = np.array([0, 0, 6, 6])  # angry, angry, neutral, neutral
    y_pred = np.array([0, 6, 6, 6])  # one angry misclassified as neutral

    metrics = compute_split_metrics("validation", y_true, y_pred)
    cm = np.array(metrics.confusion_matrix)

    angry_idx = CANONICAL_EMOTION_LABELS.index("angry")
    neutral_idx = CANONICAL_EMOTION_LABELS.index("neutral")
    assert cm[angry_idx, angry_idx] == 1
    assert cm[angry_idx, neutral_idx] == 1
    assert cm[neutral_idx, neutral_idx] == 2


def test_compute_split_metrics_per_class_report_has_all_7_labels():
    y_true = np.array([0, 1, 2, 3, 4, 5, 6])
    y_pred = np.array([1, 1, 2, 3, 4, 5, 6])  # one miss

    metrics = compute_split_metrics("test", y_true, y_pred)

    for label in CANONICAL_EMOTION_LABELS:
        assert label in metrics.per_class_report
        assert "precision" in metrics.per_class_report[label]
        assert "recall" in metrics.per_class_report[label]
        assert "f1-score" in metrics.per_class_report[label]
        assert "support" in metrics.per_class_report[label]


def test_sanity_check_predictions_passes_for_well_formed_output():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_proba = np.array([[0.9, 0.01, 0.01, 0.01, 0.01, 0.03, 0.03], [0.02, 0.9, 0.01, 0.01, 0.02, 0.02, 0.02]])

    sanity_check_predictions(y_true, y_pred, y_proba)  # should not raise


def test_sanity_check_predictions_rejects_wrong_column_count():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_proba = np.zeros((2, 5))  # only 5 columns, not 7

    with pytest.raises(AssertionError):
        sanity_check_predictions(y_true, y_pred, y_proba)


def test_sanity_check_predictions_rejects_rows_not_summing_to_one():
    y_true = np.array([0])
    y_pred = np.array([0])
    y_proba = np.array([[0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])  # sums to 1.5

    with pytest.raises(AssertionError):
        sanity_check_predictions(y_true, y_pred, y_proba)


def test_sanity_check_predictions_rejects_count_mismatch():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1])  # wrong length
    y_proba = np.zeros((2, 7))

    with pytest.raises(AssertionError):
        sanity_check_predictions(y_true, y_pred, y_proba)
