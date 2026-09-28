"""
Mock EmotionModel implementations (Phase 4).

Lets the handcrafted-feature and feature-assembly logic in
app.features be unit-tested without DeepFace, model weights, or
network access — see Phase 4 spec sec. 31. Never used in production;
selected via `emotion_model_backend="mock"` only in tests/dev.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.emotion.base import EmotionModel
from app.emotion.validation import CANONICAL_EMOTION_LABELS


class FixedEmotionModel(EmotionModel):
    """Always returns the same {label: probability} dict, regardless of input."""

    def __init__(self, probabilities: dict[str, float] | None = None) -> None:
        self._probabilities = probabilities or {label: 1.0 / 7.0 for label in CANONICAL_EMOTION_LABELS}

    def predict_probabilities(self, face_crop: np.ndarray) -> dict[str, float]:
        return dict(self._probabilities)


class FunctionEmotionModel(EmotionModel):
    """Delegates to a supplied callable(face_crop) -> dict, for tests that need input-dependent output."""

    def __init__(self, fn: Callable[[np.ndarray], dict[str, float]]) -> None:
        self._fn = fn

    def predict_probabilities(self, face_crop: np.ndarray) -> dict[str, float]:
        return self._fn(face_crop)
