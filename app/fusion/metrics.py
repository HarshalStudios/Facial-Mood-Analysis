"""
Classification metrics for the Phase 6 fusion model (spec sec. 17-19, 42).

Every metric here is computed with an explicit `labels=range(7)` /
`target_names=CANONICAL_EMOTION_LABELS` argument passed to scikit-learn, so
the canonical Angry/Disgust/Fear/Happy/Sad/Surprise/Neutral ordering is
preserved end-to-end and never silently replaced by scikit-learn's default
(alphabetical-on-the-labels-seen) ordering (spec sec. 18).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.schemas.fusion import SplitMetrics

_LABEL_IDS = list(range(len(CANONICAL_EMOTION_LABELS)))


def compute_split_metrics(split: str, y_true: np.ndarray, y_pred: np.ndarray) -> SplitMetrics:
    accuracy = float(accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=_LABEL_IDS, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=_LABEL_IDS, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=_LABEL_IDS)

    report = classification_report(
        y_true,
        y_pred,
        labels=_LABEL_IDS,
        target_names=CANONICAL_EMOTION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    return SplitMetrics(
        split=split,
        n_samples=int(len(y_true)),
        accuracy=accuracy,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        confusion_matrix=cm.tolist(),
        confusion_matrix_labels=list(CANONICAL_EMOTION_LABELS),
        per_class_report=report,
    )


def sanity_check_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, *, expected_classes: int = 7
) -> None:
    """
    Model sanity checks (Phase 6 spec sec. 42): prediction count matches
    sample count, probability matrix has one column per class, and every
    row of probabilities sums to ~1.
    """
    n = len(y_true)
    if len(y_pred) != n:
        raise AssertionError(f"prediction count ({len(y_pred)}) != sample count ({n})")
    if y_proba.shape != (n, expected_classes):
        raise AssertionError(f"probability matrix shape {y_proba.shape} != ({n}, {expected_classes})")
    row_sums = y_proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-3))[0]
        raise AssertionError(f"{len(bad)} probability row(s) do not sum to ~1 (example indices: {bad[:5].tolist()})")
