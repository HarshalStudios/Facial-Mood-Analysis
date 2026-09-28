"""
Tests for TemporalPredictionEngine (Phase 8 spec sec 40-47).

Mocks the Phase 7 `RealTimePredictor` entirely (a stub that returns
pre-built `PredictionResult`s in sequence) so these run without a
webcam, DeepFace, or a real trained model -- consistent with
`tests/test_predictor.py`'s own approach for Phase 7. A real webcam
smoke test is a separate manual step, not part of this suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.prediction.schemas import PredictionResult
from app.quality.base import QualityAnalyzer
from app.schemas.face import BoundingBox
from app.schemas.quality import QualityResult, QualityWarning
from app.temporal.engine import TemporalPredictionEngine
from app.temporal.moving_average import MovingAverageSmoother


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


def _fake_settings(**overrides) -> SimpleNamespace:
    base = dict(
        temporal_enabled=True,
        smoothing_method="moving_average",
        smoothing_window_size=5,
        ema_alpha=0.3,
        quality_enabled=True,
        quality_mode="advisory",
        max_missing_face_frames=3,
        face_continuity_iou_threshold=0.2,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _probabilities(dominant: str, dominant_value: float = 0.8) -> dict[str, float]:
    remaining = (1.0 - dominant_value) / (len(CANONICAL_EMOTION_LABELS) - 1)
    probs = {label: remaining for label in CANONICAL_EMOTION_LABELS}
    probs[dominant] = dominant_value
    return probs


def _make_frame(frame_id: int = 0):
    from app.schemas.frame import Frame

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(image=image, frame_id=frame_id)


def _valid_result(
    frame_id: int,
    emotion: str = "happy",
    confidence: float = 0.8,
    bbox: BoundingBox | None = None,
) -> PredictionResult:
    bbox = bbox or BoundingBox(x=10, y=10, width=100, height=100)
    return PredictionResult(
        frame_id=frame_id,
        face_detected=True,
        prediction_available=True,
        number_of_faces=1,
        primary_face_bbox=bbox,
        predicted_emotion=emotion,
        confidence=confidence,
        probabilities=_probabilities(emotion, confidence),
        feature_vector=None,
    )


def _no_face_result(frame_id: int) -> PredictionResult:
    return PredictionResult(frame_id=frame_id, face_detected=False, prediction_available=False)


def _unavailable_result(frame_id: int, bbox: BoundingBox | None = None, error_code: str = "model_inference_failed") -> PredictionResult:
    bbox = bbox or BoundingBox(x=10, y=10, width=100, height=100)
    return PredictionResult(
        frame_id=frame_id,
        face_detected=True,
        prediction_available=False,
        number_of_faces=1,
        primary_face_bbox=bbox,
        error_code=error_code,
        error_message="synthetic failure",
    )


class StubPredictor:
    """Returns pre-built PredictionResults in sequence, one per .predict() call."""

    def __init__(self, results: list[PredictionResult]) -> None:
        self._results = list(results)
        self._i = 0

    def predict(self, frame) -> PredictionResult:
        result = self._results[self._i]
        self._i = min(self._i + 1, len(self._results) - 1)
        return result


class FixedQualityAnalyzer(QualityAnalyzer):
    """Always returns the same QualityResult regardless of input."""

    def __init__(self, result: QualityResult) -> None:
        self._result = result

    def analyze(self, face_crop, bounding_box, frame_width, frame_height, feature_vector=None) -> QualityResult:
        return self._result


def _build_engine(
    results: list[PredictionResult],
    *,
    quality_result: QualityResult | None = None,
    settings: SimpleNamespace | None = None,
    smoother=None,
) -> TemporalPredictionEngine:
    quality_analyzer = FixedQualityAnalyzer(
        quality_result or QualityResult(is_acceptable=True, warnings=[], metrics={}, status="good")
    )
    return TemporalPredictionEngine(
        settings=settings or _fake_settings(),
        predictor=StubPredictor(results),
        quality_analyzer=quality_analyzer,
        smoother=smoother or MovingAverageSmoother(window_size=5),
    )


# ----------------------------------------------------------------------
# End-to-end mocked pipeline (spec sec 47)
# ----------------------------------------------------------------------


def test_end_to_end_mocked_pipeline_produces_raw_and_smoothed_output():
    results = [_valid_result(0, "happy", 0.8), _valid_result(1, "happy", 0.7)]
    engine = _build_engine(results)

    r1 = engine.process(_make_frame(0))
    assert r1.face_detected is True
    assert r1.prediction_available is True
    assert r1.raw_emotion == "happy"
    assert r1.smoothed_emotion == "happy"
    assert r1.history_length == 1

    r2 = engine.process(_make_frame(1))
    assert r2.history_length == 2
    assert r2.raw_confidence == pytest.approx(0.7)


# ----------------------------------------------------------------------
# No-face handling (spec sec 41)
# ----------------------------------------------------------------------


def test_no_face_does_not_hold_old_emotion_forever():
    results = [
        _valid_result(0, "happy", 0.9),
        _no_face_result(1),
        _no_face_result(2),
        _no_face_result(3),  # max_missing_face_frames=3 -> should trigger reset here
        _no_face_result(4),
    ]
    engine = _build_engine(results, settings=_fake_settings(max_missing_face_frames=3))

    engine.process(_make_frame(0))
    assert engine.history_length == 1

    r = engine.process(_make_frame(1))
    assert r.face_detected is False
    assert r.prediction_available is False

    engine.process(_make_frame(2))
    r3 = engine.process(_make_frame(3))
    assert engine.history_length == 0  # reset triggered by the 3rd consecutive no-face frame

    r4 = engine.process(_make_frame(4))
    assert r4.face_detected is False
    assert r4.history_length == 0


# ----------------------------------------------------------------------
# Reset (engine-level analogue of spec sec 40)
# ----------------------------------------------------------------------


def test_explicit_reset_clears_history_and_stability_stats():
    results = [_valid_result(0, "happy"), _valid_result(1, "sad")]
    engine = _build_engine(results)

    engine.process(_make_frame(0))
    engine.process(_make_frame(1))
    assert engine.history_length == 2
    assert engine.stability_stats["raw_transitions"] == 1

    engine.reset()
    assert engine.history_length == 0
    assert engine.stability_stats == {"raw_transitions": 0, "smoothed_transitions": 0}


# ----------------------------------------------------------------------
# Quality policy (spec sec 43)
# ----------------------------------------------------------------------


def test_advisory_quality_warning_keeps_prediction_available():
    results = [_valid_result(0, "happy", 0.8)]
    warning_quality = QualityResult(
        is_acceptable=True, warnings=[QualityWarning.LOW_CONTRAST], metrics={}, status="warning"
    )
    engine = _build_engine(results, quality_result=warning_quality, settings=_fake_settings(quality_mode="advisory"))

    result = engine.process(_make_frame(0))
    assert result.prediction_available is True
    assert result.quality.warnings == [QualityWarning.LOW_CONTRAST]
    assert engine.history_length == 1


def test_strict_severe_quality_failure_makes_prediction_unavailable():
    results = [_valid_result(0, "happy", 0.8)]
    severe_quality = QualityResult(
        is_acceptable=False, warnings=[QualityWarning.FACE_TOO_SMALL], metrics={}, status="reject"
    )
    engine = _build_engine(results, quality_result=severe_quality, settings=_fake_settings(quality_mode="strict"))

    result = engine.process(_make_frame(0))
    assert result.prediction_available is False
    # Transparency (spec sec 24-25): the raw model output is still surfaced.
    assert result.raw_emotion == "happy"
    # And it must never have entered temporal history.
    assert engine.history_length == 0


# ----------------------------------------------------------------------
# Invalid prediction never enters history (spec sec 23, 44)
# ----------------------------------------------------------------------


def test_model_inference_failure_is_not_added_to_history_and_holds_last_stable_state():
    results = [_valid_result(0, "happy", 0.9), _unavailable_result(1)]
    engine = _build_engine(results)

    r1 = engine.process(_make_frame(0))
    assert r1.smoothed_emotion == "happy"
    assert engine.history_length == 1

    r2 = engine.process(_make_frame(1))
    assert r2.prediction_available is False
    assert r2.raw_emotion is None
    # Holds at the last known stable state instead of going blank.
    assert r2.smoothed_emotion == "happy"
    assert engine.history_length == 1  # unchanged -- invalid frame never entered history


# ----------------------------------------------------------------------
# Raw vs smoothed distinguishable (spec sec 45)
# ----------------------------------------------------------------------


def test_raw_and_smoothed_can_disagree():
    results = [
        _valid_result(0, "happy", 0.9),
        _valid_result(1, "happy", 0.9),
        _valid_result(2, "surprise", 0.9),  # one isolated flicker
    ]
    engine = _build_engine(results, settings=_fake_settings(smoothing_window_size=5))
    engine.process(_make_frame(0))
    engine.process(_make_frame(1))
    r3 = engine.process(_make_frame(2))

    assert r3.raw_emotion == "surprise"
    # Two "happy" frames still outweigh one "surprise" frame in the average.
    assert r3.smoothed_emotion == "happy"


# ----------------------------------------------------------------------
# Multi-face continuity (spec sec 12, 46)
# ----------------------------------------------------------------------


def test_primary_face_jump_resets_temporal_history():
    face_a = BoundingBox(x=10, y=10, width=100, height=100)
    face_b = BoundingBox(x=500, y=500, width=100, height=100)
    results = [
        _valid_result(0, "happy", bbox=face_a),
        _valid_result(1, "happy", bbox=face_a),
        _valid_result(2, "sad", bbox=face_b),  # different subject suddenly becomes primary
    ]
    engine = _build_engine(results, settings=_fake_settings(face_continuity_iou_threshold=0.3))

    engine.process(_make_frame(0))
    engine.process(_make_frame(1))
    assert engine.history_length == 2

    r3 = engine.process(_make_frame(2))
    # History was reset then this frame's own prediction was added back in.
    assert engine.history_length == 1
    assert r3.raw_emotion == "sad"


# ----------------------------------------------------------------------
# NaN probabilities never enter history (spec sec 44)
# ----------------------------------------------------------------------


def test_nan_probabilities_never_enter_history():
    good = _valid_result(0, "happy", 0.9)
    bad = _valid_result(1, "sad", 0.8)
    bad.probabilities = {**bad.probabilities, "sad": float("nan")}
    engine = _build_engine([good, bad])

    engine.process(_make_frame(0))
    r2 = engine.process(_make_frame(1))

    assert r2.prediction_available is False
    assert engine.history_length == 1
    assert all(np.isfinite(v) for v in r2.smoothed_probabilities.values())
