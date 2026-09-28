"""
Canonical emotion label order + output validation (Phase 4).

Any EmotionModel implementation (DeepFace, a mock, or a future
alternative backend) returns a plain {label: probability} dict with no
guaranteed ordering. This module is the single place that maps that
dict onto the fixed order used everywhere else in the project
(app.schemas.feature_vector.FEATURE_NAMES[0:7]), and validates the
values before they become part of the 22D feature vector.
"""

from __future__ import annotations

import numpy as np

from app.core.exceptions import PredictionError

# Must match FEATURE_NAMES[0:7] in app.schemas.feature_vector, with the
# "emotion_" prefix stripped. Defined once here (not re-derived from
# FEATURE_NAMES) so this module has no import-time dependency loop and
# the mapping stays an explicit, readable list.
CANONICAL_EMOTION_LABELS: list[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

_PROB_EPSILON = 1e-6


def validate_and_order_emotion_probabilities(raw: dict[str, float]) -> np.ndarray:
    """
    Map an EmotionModel's {label: value} output onto CANONICAL_EMOTION_LABELS
    order and validate it.

    Raises PredictionError (not FeatureExtractionError — this is a
    failure of the emotion *model's* output, per app.core.exceptions) if:
    - any canonical label is missing
    - a value is non-numeric, NaN, or infinite
    - a value falls meaningfully outside [0, 1] (small floating-point
      overshoot right at the boundary is clipped, not treated as invalid,
      since some backends emit e.g. 1.0000000002 due to rounding)

    Extra keys beyond the 7 canonical labels are ignored, not an error —
    a provider may return additional metadata alongside the emotion dict.
    """
    if raw is None:
        raise PredictionError("Emotion provider returned None instead of a probability dict")

    normalized_keys = {str(k).strip().lower(): v for k, v in raw.items()}

    missing = [label for label in CANONICAL_EMOTION_LABELS if label not in normalized_keys]
    if missing:
        raise PredictionError(
            "Emotion provider output is missing required label(s)",
            details={"missing": missing, "received_keys": list(raw.keys())},
        )

    ordered_values: list[float] = []
    for label in CANONICAL_EMOTION_LABELS:
        value = normalized_keys[label]
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionError(
                f"Emotion probability for '{label}' is not numeric",
                details={"label": label, "value": normalized_keys[label]},
            ) from exc

        if not np.isfinite(value):
            raise PredictionError(
                f"Emotion probability for '{label}' is not finite",
                details={"label": label, "value": value},
            )

        if value < -_PROB_EPSILON or value > 1.0 + _PROB_EPSILON:
            raise PredictionError(
                f"Emotion probability for '{label}' is outside the expected [0, 1] range",
                details={"label": label, "value": value},
            )

        ordered_values.append(min(1.0, max(0.0, value)))

    return np.array(ordered_values, dtype=np.float64)
