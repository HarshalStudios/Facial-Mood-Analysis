import numpy as np
import pytest

from app.core.exceptions import PredictionError
from app.emotion.validation import (
    CANONICAL_EMOTION_LABELS,
    validate_and_order_emotion_probabilities,
)


def test_orders_values_to_canonical_label_order():
    raw = {
        "neutral": 0.1,
        "happy": 0.5,
        "angry": 0.05,
        "disgust": 0.05,
        "fear": 0.05,
        "sad": 0.1,
        "surprise": 0.15,
    }
    ordered = validate_and_order_emotion_probabilities(raw)
    assert list(CANONICAL_EMOTION_LABELS) == [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]
    assert ordered[CANONICAL_EMOTION_LABELS.index("happy")] == pytest.approx(0.5)
    assert ordered[CANONICAL_EMOTION_LABELS.index("neutral")] == pytest.approx(0.1)
    assert ordered.shape == (7,)


def test_is_case_and_whitespace_insensitive_on_keys():
    raw = {" Angry ": 0.2, "DISGUST": 0.1, "Fear": 0.1, "happy": 0.3, "Sad": 0.1, "surprise": 0.1, "NEUTRAL": 0.1}
    ordered = validate_and_order_emotion_probabilities(raw)
    assert ordered.shape == (7,)


def test_ignores_extra_keys():
    raw = {
        "angry": 0.1,
        "disgust": 0.1,
        "fear": 0.1,
        "happy": 0.4,
        "sad": 0.1,
        "surprise": 0.1,
        "neutral": 0.1,
        "region": {"x": 0, "y": 0},  # DeepFace also returns non-emotion keys
    }
    ordered = validate_and_order_emotion_probabilities(raw)
    assert ordered.shape == (7,)


def test_missing_label_raises_prediction_error():
    raw = {label: 1 / 6 for label in CANONICAL_EMOTION_LABELS if label != "surprise"}
    with pytest.raises(PredictionError):
        validate_and_order_emotion_probabilities(raw)


def test_none_input_raises_prediction_error():
    with pytest.raises(PredictionError):
        validate_and_order_emotion_probabilities(None)


def test_nan_value_raises_prediction_error():
    raw = {label: 1 / 7 for label in CANONICAL_EMOTION_LABELS}
    raw["happy"] = float("nan")
    with pytest.raises(PredictionError):
        validate_and_order_emotion_probabilities(raw)


def test_infinite_value_raises_prediction_error():
    raw = {label: 1 / 7 for label in CANONICAL_EMOTION_LABELS}
    raw["angry"] = float("inf")
    with pytest.raises(PredictionError):
        validate_and_order_emotion_probabilities(raw)


def test_grossly_out_of_range_value_raises_prediction_error():
    raw = {label: 1 / 7 for label in CANONICAL_EMOTION_LABELS}
    raw["sad"] = 42.0
    with pytest.raises(PredictionError):
        validate_and_order_emotion_probabilities(raw)


def test_tiny_floating_point_overshoot_is_clipped_not_rejected():
    raw = {label: 1 / 7 for label in CANONICAL_EMOTION_LABELS}
    raw["fear"] = 1.0000000005
    ordered = validate_and_order_emotion_probabilities(raw)
    assert ordered[CANONICAL_EMOTION_LABELS.index("fear")] == 1.0


def test_non_numeric_value_raises_prediction_error():
    raw = {label: 1 / 7 for label in CANONICAL_EMOTION_LABELS}
    raw["neutral"] = "not-a-number"
    with pytest.raises(PredictionError):
        validate_and_order_emotion_probabilities(raw)
