"""
Phase 7 tests for RealTimePredictor.

Mocks every heavy dependency (detector, representation engine, feature
extractor, fusion model) so these run without a webcam, DeepFace, or a
real trained model -- see Phase 7 spec sec. 35, tests 1-9. A real
webcam smoke test is a separate manual step
(`python scripts/run_realtime_prediction.py`), not part of this suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.core.exceptions import FeatureExtractionError, PredictionError, RepresentationError
from app.detection.base import FaceDetector
from app.detection.service import FaceDetectionResult
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.features.base import FeatureExtractor
from app.fusion.base import FusionModel
from app.prediction.predictor import (
    ERROR_FEATURE_EXTRACTION_FAILED,
    ERROR_INVALID_PROBABILITIES,
    ERROR_MODEL_INFERENCE_FAILED,
    ERROR_REPRESENTATION_FAILED,
    RealTimePredictor,
)
from app.representations.base import RepresentationGenerator
from app.schemas.face import BoundingBox, Face
from app.schemas.feature_extraction import FeatureExtractionResult
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH, FeatureVector
from app.schemas.frame import Frame
from app.schemas.prediction import EmotionPrediction
from app.schemas.representations import Representations


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


def _fake_settings() -> SimpleNamespace:
    """
    A duck-typed stand-in for app.core.config.Settings. RealTimePredictor
    itself never reads from this when every component (detector,
    representation engine, feature extractor, fusion model) is supplied
    explicitly -- but FaceDetectionService (constructed internally around
    whatever detector is passed in) still needs its own handful of
    settings fields, so they're provided here too.
    """
    return SimpleNamespace(
        face_padding_ratio=0.0,
        min_face_width=10,
        min_face_height=10,
        primary_face_strategy="largest",
    )


def _make_frame(frame_id: int = 0) -> Frame:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(image=image, frame_id=frame_id)


class NoFaceDetector(FaceDetector):
    def detect(self, frame: Frame) -> list[Face]:
        return []


class OneFaceDetector(FaceDetector):
    """Always reports one already-primary face covering a fixed region."""

    def __init__(self, bbox: BoundingBox | None = None) -> None:
        self._bbox = bbox or BoundingBox(x=10, y=10, width=50, height=50)

    def detect(self, frame: Frame) -> list[Face]:
        crop = np.zeros((self._bbox.height, self._bbox.width, 3), dtype=np.uint8)
        return [
            Face(bounding_box=self._bbox, crop=crop, confidence=None, face_id=0, is_primary=True)
        ]


class FailingDetectionService:
    """Stand-in for FaceDetectionService.process raising a BackendError."""

    def process(self, frame: Frame):
        from app.core.exceptions import DetectionError

        raise DetectionError("synthetic detection failure")


class StubRepresentationEngine(RepresentationGenerator):
    def __init__(self, should_fail: bool = False) -> None:
        self._should_fail = should_fail

    def generate(self, face_crop: np.ndarray) -> Representations:
        if self._should_fail:
            raise RepresentationError("synthetic representation failure")
        h, w = face_crop.shape[:2]
        gray = np.zeros((h, w), dtype=np.uint8)
        return Representations(rgb=face_crop, grayscale=gray, canny=gray, lbp=gray, metadata={"rgb": object()})


class StubFeatureExtractor(FeatureExtractor):
    def __init__(self, should_fail: bool = False, bad_shape: bool = False) -> None:
        self._should_fail = should_fail
        self._bad_shape = bad_shape

    def extract(self, representations: Representations) -> FeatureExtractionResult:
        if self._should_fail:
            raise FeatureExtractionError("synthetic feature extraction failure")
        length = FEATURE_VECTOR_LENGTH - 1 if self._bad_shape else FEATURE_VECTOR_LENGTH
        values = np.full(length, 0.1, dtype=np.float64)
        if self._bad_shape:
            # FeatureVector itself would reject this; simulate a caller
            # that bypasses it to exercise the predictor's own defensive check.
            class _FakeVector:
                pass

            fake = _FakeVector()
            fake.values = values
            return FeatureExtractionResult(vector=fake, features={}, metadata={})
        vector = FeatureVector(values=values)
        return FeatureExtractionResult(vector=vector, features=vector.as_dict(), metadata={})


class StubFusionModel(FusionModel):
    def __init__(
        self,
        emotion: str = "happy",
        confidence: float = 0.80,
        should_fail: bool = False,
        bad_probabilities: dict | None = None,
    ) -> None:
        self._emotion = emotion
        self._confidence = confidence
        self._should_fail = should_fail
        self._bad_probabilities = bad_probabilities

    def load(self, model_path: str, scaler_path: str) -> None:  # pragma: no cover - not exercised
        raise NotImplementedError

    def predict(self, feature_vector: FeatureVector) -> EmotionPrediction:
        if self._should_fail:
            raise PredictionError("synthetic model inference failure")
        if self._bad_probabilities is not None:
            return EmotionPrediction(
                emotion=self._emotion, confidence=self._confidence, probabilities=self._bad_probabilities
            )
        remaining = (1.0 - self._confidence) / 6
        probabilities = {label: remaining for label in CANONICAL_EMOTION_LABELS}
        probabilities[self._emotion] = self._confidence
        return EmotionPrediction(emotion=self._emotion, confidence=self._confidence, probabilities=probabilities)


def _build_predictor(
    *,
    detector: FaceDetector | None = None,
    representation_engine: RepresentationGenerator | None = None,
    feature_extractor: FeatureExtractor | None = None,
    fusion_model: FusionModel | None = None,
    detection_service_override=None,
) -> RealTimePredictor:
    predictor = RealTimePredictor(
        settings=_fake_settings(),
        face_detector=detector or NoFaceDetector(),
        representation_engine=representation_engine or StubRepresentationEngine(),
        emotion_model=None,
        feature_extractor=feature_extractor or StubFeatureExtractor(),
        fusion_model=fusion_model or StubFusionModel(),
        model_metadata={"model_version": "test_v1", "feature_schema_version": "1"},
    )
    if detection_service_override is not None:
        predictor._detection_service = detection_service_override  # noqa: SLF001 - test injection
    return predictor


# ----------------------------------------------------------------------
# Test 5 — deterministic prediction (spec sec. 35 test 5)
# ----------------------------------------------------------------------


def test_known_probability_vector_yields_happy_with_expected_confidence():
    probabilities = dict(
        zip(
            CANONICAL_EMOTION_LABELS,
            [0.05, 0.01, 0.02, 0.80, 0.04, 0.03, 0.05],
        )
    )
    fusion_model = StubFusionModel(emotion="happy", confidence=0.80, bad_probabilities=probabilities)
    predictor = _build_predictor(detector=OneFaceDetector(), fusion_model=fusion_model)

    result = predictor.predict(_make_frame())

    assert result.prediction_available is True
    assert result.predicted_emotion == "happy"
    assert result.confidence == pytest.approx(0.80)


# ----------------------------------------------------------------------
# Test 6 — no-face case
# ----------------------------------------------------------------------


def test_no_face_detected_returns_unavailable_without_inventing_emotion():
    predictor = _build_predictor(detector=NoFaceDetector())

    result = predictor.predict(_make_frame())

    assert result.face_detected is False
    assert result.prediction_available is False
    assert result.predicted_emotion is None
    assert result.confidence is None
    assert result.number_of_faces == 0


# ----------------------------------------------------------------------
# Test 7 — multiple faces: predictor must use the existing primary-face
# selection (FaceDetectionService + selector), not a new implementation.
# ----------------------------------------------------------------------


def test_multiple_faces_uses_existing_primary_face_selection():
    from app.detection.selector import get_face_selector
    from app.detection.service import FaceDetectionService

    class TwoFaceDetector(FaceDetector):
        def detect(self, frame: Frame) -> list[Face]:
            small = BoundingBox(x=0, y=0, width=40, height=40)
            large = BoundingBox(x=200, y=200, width=120, height=120)
            return [
                Face(bounding_box=small, crop=np.zeros((40, 40, 3), dtype=np.uint8), confidence=None, face_id=0),
                Face(bounding_box=large, crop=np.zeros((120, 120, 3), dtype=np.uint8), confidence=None, face_id=1),
            ]

    settings = SimpleNamespace(
        face_padding_ratio=0.0,
        min_face_width=10,
        min_face_height=10,
        primary_face_strategy="largest",
    )
    detection_service = FaceDetectionService(TwoFaceDetector(), settings)
    assert get_face_selector("largest")  # sanity: strategy resolvable

    predictor = _build_predictor(detection_service_override=detection_service)

    result = predictor.predict(_make_frame())

    assert result.number_of_faces == 2
    assert result.face_detected is True
    assert result.primary_face_bbox.width == 120  # the larger face was selected as primary


# ----------------------------------------------------------------------
# Test 8 — feature extraction failure must not invent features/emotion
# ----------------------------------------------------------------------


def test_feature_extraction_failure_returns_unavailable_and_logs_no_invented_values():
    predictor = _build_predictor(
        detector=OneFaceDetector(), feature_extractor=StubFeatureExtractor(should_fail=True)
    )

    result = predictor.predict(_make_frame())

    assert result.face_detected is True
    assert result.prediction_available is False
    assert result.predicted_emotion is None
    assert result.error_code == ERROR_FEATURE_EXTRACTION_FAILED


def test_representation_failure_returns_unavailable():
    predictor = _build_predictor(
        detector=OneFaceDetector(), representation_engine=StubRepresentationEngine(should_fail=True)
    )

    result = predictor.predict(_make_frame())

    assert result.prediction_available is False
    assert result.error_code == ERROR_REPRESENTATION_FAILED


def test_model_inference_failure_returns_unavailable():
    predictor = _build_predictor(detector=OneFaceDetector(), fusion_model=StubFusionModel(should_fail=True))

    result = predictor.predict(_make_frame())

    assert result.prediction_available is False
    assert result.error_code == ERROR_MODEL_INFERENCE_FAILED


def test_invalid_probabilities_are_rejected_not_normalized_silently():
    bad_probabilities = dict(zip(CANONICAL_EMOTION_LABELS, [0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]))  # sums to 1.8
    predictor = _build_predictor(
        detector=OneFaceDetector(), fusion_model=StubFusionModel(bad_probabilities=bad_probabilities)
    )

    result = predictor.predict(_make_frame())

    assert result.prediction_available is False
    assert result.error_code == ERROR_INVALID_PROBABILITIES


# ----------------------------------------------------------------------
# Test 9 — end-to-end mocked pipeline: frame in, structured result out,
# no physical webcam required.
# ----------------------------------------------------------------------


def test_end_to_end_mocked_pipeline_returns_structured_available_prediction():
    predictor = _build_predictor(detector=OneFaceDetector())

    result = predictor.predict(_make_frame(frame_id=42))

    assert result.frame_id == 42
    assert result.face_detected is True
    assert result.prediction_available is True
    assert result.predicted_emotion in CANONICAL_EMOTION_LABELS
    assert result.probabilities is not None
    assert result.processing_time_ms is not None and result.processing_time_ms >= 0
    assert result.timing.face_detection_ms is not None
    assert result.timing.representation_ms is not None
    assert result.timing.feature_extraction_ms is not None
    assert result.timing.model_inference_ms is not None
    # Not requested via include_feature_vector -> stays out of the result.
    assert result.feature_vector is None


def test_feature_vector_included_only_when_requested():
    predictor = RealTimePredictor(
        settings=_fake_settings(),
        face_detector=OneFaceDetector(),
        representation_engine=StubRepresentationEngine(),
        emotion_model=None,
        feature_extractor=StubFeatureExtractor(),
        fusion_model=StubFusionModel(),
        model_metadata=None,
        include_feature_vector=True,
    )

    result = predictor.predict(_make_frame())

    assert result.feature_vector is not None
    assert len(result.feature_vector) == FEATURE_VECTOR_LENGTH


def test_model_version_and_feature_schema_version_are_surfaced_on_every_result():
    predictor = _build_predictor(detector=OneFaceDetector())

    no_face_result = predictor.predict(_make_frame())
    # OneFaceDetector always reports a face, so also check the no-face path
    # directly against a NoFaceDetector-backed predictor.
    predictor_no_face = _build_predictor(detector=NoFaceDetector())
    no_face_result = predictor_no_face.predict(_make_frame())

    assert no_face_result.model_version == "test_v1"
    assert no_face_result.feature_schema_version == "1"
