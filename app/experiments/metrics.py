"""Reusable Phase 9 evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from app.emotion.validation import CANONICAL_EMOTION_LABELS


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CANONICAL_EMOTION_LABELS)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=CANONICAL_EMOTION_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "class_order": CANONICAL_EMOTION_LABELS,
        "classification_report": report,
    }
