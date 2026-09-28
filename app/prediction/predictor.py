"""
RealTimePredictor (Phase 7).

Orchestrates the complete frame-level inference pipeline built in
Phases 1-6, reusing every existing component rather than
re-implementing any of it:

    Frame
      -> FaceDetectionService (Phase 2)      -> primary Face
      -> RepresentationEngine (Phase 3)       -> Representations
      -> StandardFeatureExtractor (Phase 4)   -> FeatureVector (22D)
      -> LogisticRegressionFusionModel (Phase 6) -> EmotionPrediction

`RealTimePredictor` does not own a webcam -- it only ever receives an
already-captured `Frame` (see `app.camera.base.CameraSource` for the
capture side, and `scripts/run_realtime_prediction.py` for a CLI that
wires the two together). This separation is what lets the same engine
be driven by a webcam loop today and, from Phase 10 onward, a REST/
WebSocket endpoint without any change here.

Scope (Phase 7 spec sec. 4): frame-level prediction only. No temporal
smoothing, no quality gating, no production API -- see
`app.temporal` / `app.quality` (Phase 8) and Phase 10 for those.

Error handling contract (spec sec. 23):
- No face in frame -> `prediction_available=False`, no error, never an
  invented emotion.
- A recoverable per-frame failure (bad crop, representation failure,
  feature-extraction failure, model-inference failure, invalid
  probabilities) -> logged and returned as an unavailable
  `PredictionResult` with a specific `error_code`. The loop must keep
  running.
- A fatal initialization failure (model artifact missing/incompatible,
  detector/model construction failure) -> raised out of `__init__` as
  the underlying `BackendError` (typically `ModelLoadError`); this is
  never caught here, since silently continuing with a broken predictor
  would be worse than refusing to start.
"""

from __future__ import annotations

import time

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    BackendError,
    FeatureExtractionError,
    PredictionError,
    RepresentationError,
)
from app.core.logging import get_logger
from app.detection.base import FaceDetector
from app.detection.opencv_detector import HaarCascadeFaceDetector
from app.detection.service import FaceDetectionService
from app.emotion.base import EmotionModel
from app.emotion.factory import get_emotion_model
from app.features.base import FeatureExtractor
from app.features.extractor import StandardFeatureExtractor
from app.fusion.base import FusionModel
from app.prediction.loader import load_fusion_model
from app.prediction.schemas import PredictionResult, TimingBreakdown
from app.representations.base import RepresentationGenerator
from app.representations.engine import RepresentationEngine
from app.schemas.face import BoundingBox
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH
from app.schemas.frame import Frame

logger = get_logger(__name__)

# Error codes surfaced on an unavailable PredictionResult (spec sec. 21/23).
# Kept as plain strings (not an enum) so a future API layer can pass them
# straight through without an extra translation step.
ERROR_DETECTION_FAILED = "detection_failed"
ERROR_INVALID_CROP = "invalid_face_crop"
ERROR_REPRESENTATION_FAILED = "representation_failed"
ERROR_FEATURE_EXTRACTION_FAILED = "feature_extraction_failed"
ERROR_INVALID_FEATURE_VECTOR = "invalid_feature_vector"
ERROR_MODEL_INFERENCE_FAILED = "model_inference_failed"
ERROR_INVALID_PROBABILITIES = "invalid_probabilities"

_PROB_SUM_TOLERANCE = 1e-3
_PROB_BOUND_TOLERANCE = 1e-6


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _validate_probabilities(probabilities: dict[str, float]) -> str | None:
    """
    Validate a fusion-model probability dict (spec sec. 18). Returns
    None if valid, otherwise a human-readable reason string.
    """
    if len(probabilities) != 7:
        return f"expected 7 class probabilities, got {len(probabilities)}"

    values = list(probabilities.values())

    for label, value in probabilities.items():
        if value is None:
            return f"probability for '{label}' is None"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return f"probability for '{label}' is not numeric: {value!r}"
        if value != value or value in (float("inf"), float("-inf")):  # NaN/Inf check
            return f"probability for '{label}' is not finite: {value}"
        if value < -_PROB_BOUND_TOLERANCE or value > 1.0 + _PROB_BOUND_TOLERANCE:
            return f"probability for '{label}' is outside [0, 1]: {value}"

    total = sum(float(v) for v in values)
    if abs(total - 1.0) > _PROB_SUM_TOLERANCE:
        return f"probabilities sum to {total}, expected ~1.0"

    return None


class RealTimePredictor:
    """
    Frame-level real-time emotion prediction engine.

    All heavy resources (fusion model, emotion model, face detector)
    are constructed once here and reused for every `predict(frame)`
    call (spec sec. 13/29) -- never reloaded per frame. Construction
    can raise a `BackendError` (typically `ModelLoadError`); that is a
    fatal initialization failure and is intentionally not caught here.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        face_detector: FaceDetector | None = None,
        representation_engine: RepresentationGenerator | None = None,
        emotion_model: EmotionModel | None = None,
        feature_extractor: FeatureExtractor | None = None,
        fusion_model: FusionModel | None = None,
        model_metadata: dict | None = None,
        include_feature_vector: bool = False,
    ) -> None:
        """
        Every dependency beyond `settings` is optional and defaults to
        the real Phase 1-6 implementation -- the optional parameters
        exist so tests (and Phase 9's ablation engine) can substitute
        mocks/alternatives without subclassing. Passing an explicit
        `fusion_model` skips `load_fusion_model()` entirely, so pass
        `model_metadata` alongside it if traceability fields are
        needed in that case.
        """
        self._settings = settings or get_settings()

        detector = face_detector or HaarCascadeFaceDetector(self._settings)
        self._detection_service = FaceDetectionService(detector, self._settings)

        self._representation_engine = representation_engine or RepresentationEngine(self._settings)

        if feature_extractor is not None:
            # An explicit extractor was supplied (tests, Phase 9 ablation) --
            # do not construct a default EmotionModel at all, since it would
            # go unused and, for the real "deepface" backend, would trigger
            # an expensive/heavy import for nothing.
            self._feature_extractor = feature_extractor
        else:
            emotion = emotion_model or get_emotion_model(self._settings)
            self._feature_extractor = StandardFeatureExtractor(emotion, self._settings)

        if fusion_model is not None:
            self._fusion_model = fusion_model
            self._model_metadata = model_metadata
        else:
            self._fusion_model, self._model_metadata = load_fusion_model(self._settings)

        self._include_feature_vector = include_feature_vector

        logger.info(
            "RealTimePredictor initialized (model_version=%s, feature_schema_version=%s)",
            self._model_version,
            self._feature_schema_version,
        )

    @property
    def _model_version(self) -> str | None:
        return (self._model_metadata or {}).get("model_version")

    @property
    def _feature_schema_version(self) -> str | None:
        return (self._model_metadata or {}).get("feature_schema_version")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, frame: Frame) -> PredictionResult:
        """Run the complete pipeline on one already-captured frame. Never raises for a single bad frame."""
        t_start = time.perf_counter()
        timing = TimingBreakdown()

        # --- Face detection --------------------------------------------------
        t0 = time.perf_counter()
        try:
            detection_result = self._detection_service.process(frame)
        except BackendError as exc:
            logger.error("Face detection failed for frame %s: %s", frame.frame_id, exc.message)
            timing.face_detection_ms = _elapsed_ms(t0)
            return self._unavailable_result(
                frame, timing, t_start,
                error_code=ERROR_DETECTION_FAILED, error_message=exc.message,
            )
        timing.face_detection_ms = _elapsed_ms(t0)

        faces = detection_result.faces
        number_of_faces = len(faces)
        primary = next((f for f in faces if f.is_primary), None)

        if primary is None:
            # No face detected. Never invent a prediction (spec sec. 8/22).
            logger.debug("No face detected in frame %s", frame.frame_id)
            timing.total_ms = _elapsed_ms(t_start)
            return PredictionResult(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                face_detected=False,
                prediction_available=False,
                number_of_faces=number_of_faces,
                timing=timing,
                processing_time_ms=timing.total_ms,
                model_version=self._model_version,
                feature_schema_version=self._feature_schema_version,
            )

        # --- Representations ---------------------------------------------------
        t0 = time.perf_counter()
        try:
            representations = self._representation_engine.generate(primary.crop)
        except RepresentationError as exc:
            logger.warning("Representation generation failed for frame %s: %s", frame.frame_id, exc.message)
            timing.representation_ms = _elapsed_ms(t0)
            return self._unavailable_result(
                frame, timing, t_start,
                error_code=ERROR_REPRESENTATION_FAILED, error_message=exc.message,
                face_detected=True, number_of_faces=number_of_faces, primary_face_bbox=primary.bounding_box,
            )
        timing.representation_ms = _elapsed_ms(t0)

        # --- Feature extraction --------------------------------------------------
        t0 = time.perf_counter()
        try:
            extraction_result = self._feature_extractor.extract(representations)
        except (FeatureExtractionError, PredictionError) as exc:
            logger.warning("Feature extraction failed for frame %s: %s", frame.frame_id, exc.message)
            timing.feature_extraction_ms = _elapsed_ms(t0)
            return self._unavailable_result(
                frame, timing, t_start,
                error_code=ERROR_FEATURE_EXTRACTION_FAILED, error_message=exc.message,
                face_detected=True, number_of_faces=number_of_faces, primary_face_bbox=primary.bounding_box,
            )
        timing.feature_extraction_ms = _elapsed_ms(t0)

        feature_vector = extraction_result.vector

        # Defense in depth: StandardFeatureExtractor / FeatureVector already
        # enforce shape + finiteness (spec sec. 11), but the predictor must
        # not blindly trust that and hand a bad vector to the model.
        if feature_vector.values.shape != (FEATURE_VECTOR_LENGTH,):
            message = (
                f"feature vector has shape {feature_vector.values.shape}, "
                f"expected ({FEATURE_VECTOR_LENGTH},)"
            )
            logger.error("Invalid feature vector for frame %s: %s", frame.frame_id, message)
            return self._unavailable_result(
                frame, timing, t_start,
                error_code=ERROR_INVALID_FEATURE_VECTOR, error_message=message,
                face_detected=True, number_of_faces=number_of_faces, primary_face_bbox=primary.bounding_box,
            )

        # --- Model inference ------------------------------------------------------
        t0 = time.perf_counter()
        try:
            prediction = self._fusion_model.predict(feature_vector)
        except PredictionError as exc:
            logger.warning("Fusion model inference failed for frame %s: %s", frame.frame_id, exc.message)
            timing.model_inference_ms = _elapsed_ms(t0)
            return self._unavailable_result(
                frame, timing, t_start,
                error_code=ERROR_MODEL_INFERENCE_FAILED, error_message=exc.message,
                face_detected=True, number_of_faces=number_of_faces, primary_face_bbox=primary.bounding_box,
            )
        timing.model_inference_ms = _elapsed_ms(t0)

        prob_error = _validate_probabilities(prediction.probabilities)
        if prob_error is not None:
            logger.error("Fusion model produced invalid probabilities for frame %s: %s", frame.frame_id, prob_error)
            return self._unavailable_result(
                frame, timing, t_start,
                error_code=ERROR_INVALID_PROBABILITIES, error_message=prob_error,
                face_detected=True, number_of_faces=number_of_faces, primary_face_bbox=primary.bounding_box,
            )

        timing.total_ms = _elapsed_ms(t_start)

        return PredictionResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            face_detected=True,
            prediction_available=True,
            number_of_faces=number_of_faces,
            primary_face_bbox=primary.bounding_box,
            predicted_emotion=prediction.emotion,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            feature_vector=feature_vector.values.tolist() if self._include_feature_vector else None,
            timing=timing,
            processing_time_ms=timing.total_ms,
            model_version=self._model_version,
            feature_schema_version=self._feature_schema_version,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unavailable_result(
        self,
        frame: Frame,
        timing: TimingBreakdown,
        t_start: float,
        *,
        error_code: str,
        error_message: str,
        face_detected: bool = False,
        number_of_faces: int = 0,
        primary_face_bbox: BoundingBox | None = None,
    ) -> PredictionResult:
        timing.total_ms = _elapsed_ms(t_start)
        return PredictionResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            face_detected=face_detected,
            prediction_available=False,
            number_of_faces=number_of_faces,
            primary_face_bbox=primary_face_bbox,
            timing=timing,
            processing_time_ms=timing.total_ms,
            error_code=error_code,
            error_message=error_message,
            model_version=self._model_version,
            feature_schema_version=self._feature_schema_version,
        )
