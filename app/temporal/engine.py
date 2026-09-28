"""
TemporalPredictionEngine (Phase 8).

Sits directly on top of Phase 7's `RealTimePredictor`, adding quality
control and temporal smoothing as a separate post-processing layer
(spec sec 4/35) -- it never re-implements detection/representation/
feature-extraction/model-inference, and never modifies the trained
fusion model or feeds predictions back into it.

Pipeline order (spec sec 22):

    Frame
      -> RealTimePredictor.predict()      (Phase 7, unchanged)
      -> QualityAnalyzer.analyze()        (this phase)
      -> validity decision                (spec sec 21/23)
      -> TemporalSmoother.update()         (only for valid predictions)
      -> TemporalPredictionResult

Reset behavior (spec sec 10-12): temporal history is cleared when the
face has been missing for `max_missing_face_frames` consecutive frames,
or when the primary face's bounding box jumps enough (low IoU with the
previous frame's box) to suggest a different subject.
"""

from __future__ import annotations

import numpy as np

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.prediction.predictor import RealTimePredictor
from app.prediction.schemas import PredictionResult
from app.quality.analyzer import StandardQualityAnalyzer
from app.quality.base import QualityAnalyzer
from app.schemas.face import BoundingBox
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH, FeatureVector
from app.schemas.frame import Frame
from app.schemas.quality import QualityResult
from app.schemas.temporal import TemporalPredictionResult
from app.temporal.base import TemporalSmoother
from app.temporal.continuity import is_same_subject
from app.temporal.factory import get_temporal_smoother

logger = get_logger(__name__)


class TemporalPredictionEngine:
    """
    Single-threaded, stateful wrapper (spec sec 34) -- construct one
    instance per camera/session; do not call `process()` concurrently
    on the same instance from multiple threads.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        predictor: RealTimePredictor | None = None,
        quality_analyzer: QualityAnalyzer | None = None,
        smoother: TemporalSmoother | None = None,
    ) -> None:
        self._settings = settings or get_settings()

        if predictor is not None:
            self._predictor = predictor
        else:
            # Constructed with include_feature_vector=True so the quality
            # analyzer can reuse the already-computed brightness/contrast/
            # sharpness (spec sec 15-17) instead of recomputing them.
            self._predictor = RealTimePredictor(self._settings, include_feature_vector=True)

        self._quality_analyzer = quality_analyzer or StandardQualityAnalyzer(self._settings)
        self._smoother = smoother or get_temporal_smoother(self._settings)

        self._missing_face_streak = 0
        self._previous_bbox: BoundingBox | None = None

        # Last emitted smoothed state, held so an invalid/skipped frame
        # (spec sec 23) can still report the current stable output
        # instead of going blank.
        self._last_smoothed_probabilities: dict[str, float] | None = None
        self._last_smoothed_emotion: str | None = None
        self._last_smoothed_confidence: float | None = None

        # Research/debug jitter counters (spec sec 28-29) -- measures
        # stability, never accuracy, and is never fabricated: it only
        # ever counts transitions actually observed by this instance.
        self._last_raw_emotion: str | None = None
        self._raw_transitions = 0
        self._smoothed_transitions = 0
        # Cursor used only for smoothed-transition counting -- kept
        # separate from `_last_smoothed_emotion` (which also holds the
        # displayed state across invalid frames) so a held-over repeat is
        # never miscounted as a transition.
        self._smoothed_transition_cursor: str | None = None

        logger.info(
            "TemporalPredictionEngine initialized (temporal_enabled=%s, smoothing_method=%s, "
            "window=%s, ema_alpha=%s, quality_enabled=%s, quality_mode=%s, "
            "max_missing_face_frames=%s, face_continuity_iou_threshold=%s)",
            self._settings.temporal_enabled,
            self._settings.smoothing_method,
            self._settings.smoothing_window_size,
            self._settings.ema_alpha,
            self._settings.quality_enabled,
            self._settings.quality_mode,
            self._settings.max_missing_face_frames,
            self._settings.face_continuity_iou_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Full reset: temporal history, missing-face streak, continuity, jitter counters (spec sec 33)."""
        self._smoother.reset()
        self._missing_face_streak = 0
        self._previous_bbox = None
        self._last_smoothed_probabilities = None
        self._last_smoothed_emotion = None
        self._last_smoothed_confidence = None
        self._last_raw_emotion = None
        self._raw_transitions = 0
        self._smoothed_transitions = 0
        self._smoothed_transition_cursor = None

    @property
    def stability_stats(self) -> dict[str, int]:
        """Actual, measured label-transition counts (spec sec 28-29) -- not a claim about accuracy."""
        return {
            "raw_transitions": self._raw_transitions,
            "smoothed_transitions": self._smoothed_transitions,
        }

    @property
    def history_length(self) -> int:
        return self._smoother.history_length

    def process(self, frame: Frame) -> TemporalPredictionResult:
        """Run quality + temporal post-processing on top of one Phase 7 `predict()` call."""
        result = self._predictor.predict(frame)

        if not result.face_detected:
            return self._handle_missing_face(result)

        self._missing_face_streak = 0
        return self._handle_face_present(frame, result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_missing_face(self, result: PredictionResult) -> TemporalPredictionResult:
        self._missing_face_streak += 1

        if self._missing_face_streak >= self._settings.max_missing_face_frames and (
            self._smoother.history_length > 0 or self._previous_bbox is not None
        ):
            logger.info(
                "No face for %d consecutive frame(s) (>= max_missing_face_frames=%d); "
                "resetting temporal history so a stale emotion is not held indefinitely.",
                self._missing_face_streak,
                self._settings.max_missing_face_frames,
            )
            self._reset_history_only()

        return TemporalPredictionResult(
            frame_id=result.frame_id,
            timestamp=result.timestamp,
            face_detected=False,
            prediction_available=False,
            history_length=self._smoother.history_length,
            quality=None,
            error_code=result.error_code,
            error_message=result.error_message,
        )

    def _handle_face_present(self, frame: Frame, result: PredictionResult) -> TemporalPredictionResult:
        current_bbox = result.primary_face_bbox

        # Face continuity (spec sec 12): a big jump suggests a different
        # subject took over as primary face -- reset rather than blend
        # unrelated faces' predictions together.
        if current_bbox is not None and not is_same_subject(
            self._previous_bbox, current_bbox, self._settings.face_continuity_iou_threshold
        ):
            logger.info("Primary face bounding box jumped (different subject); resetting temporal history.")
            self._reset_history_only()
        self._previous_bbox = current_bbox

        quality_result = self._analyze_quality(frame, current_bbox, result)

        is_valid = (
            result.prediction_available
            and self._probabilities_are_finite(result.probabilities)
            and (quality_result is None or quality_result.is_acceptable)
        )

        raw_emotion = result.predicted_emotion
        raw_confidence = result.confidence
        raw_probabilities = result.probabilities

        if is_valid and raw_probabilities is not None:
            smoothed_probabilities = self._smoother.update(raw_probabilities)
            smoothed_emotion = max(smoothed_probabilities, key=smoothed_probabilities.get)
            smoothed_confidence = smoothed_probabilities[smoothed_emotion]

            self._last_smoothed_probabilities = smoothed_probabilities
            self._last_smoothed_emotion = smoothed_emotion
            self._last_smoothed_confidence = smoothed_confidence

            if self._last_raw_emotion is not None and self._last_raw_emotion != raw_emotion:
                self._raw_transitions += 1
            self._last_raw_emotion = raw_emotion
        else:
            # Invalid frame (no prediction, or a severe quality failure in
            # strict mode): never enters temporal history (spec sec 22-23).
            # Hold at the last known stable output rather than going blank.
            smoothed_probabilities = self._last_smoothed_probabilities
            smoothed_emotion = self._last_smoothed_emotion
            smoothed_confidence = self._last_smoothed_confidence

        if smoothed_emotion is not None and self._smoothed_transition_tracker_needs_update(smoothed_emotion):
            self._smoothed_transitions += 1

        return TemporalPredictionResult(
            frame_id=result.frame_id,
            timestamp=result.timestamp,
            face_detected=True,
            prediction_available=is_valid,
            raw_emotion=raw_emotion,
            raw_confidence=raw_confidence,
            raw_probabilities=raw_probabilities,
            smoothed_emotion=smoothed_emotion,
            smoothed_confidence=smoothed_confidence,
            smoothed_probabilities=smoothed_probabilities,
            history_length=self._smoother.history_length,
            quality=quality_result,
            error_code=result.error_code,
            error_message=result.error_message,
        )

    @staticmethod
    def _probabilities_are_finite(probabilities: dict[str, float] | None) -> bool:
        """Defensive guard (spec sec 44): NaN/inf must never reach the smoother state."""
        if not probabilities:
            return False
        return all(np.isfinite(v) for v in probabilities.values())

    def _smoothed_transition_tracker_needs_update(self, smoothed_emotion: str) -> bool:
        """Tracks smoothed-label transitions (spec sec 28-29); see `_smoothed_transition_cursor` docstring above."""
        if self._smoothed_transition_cursor is None:
            self._smoothed_transition_cursor = smoothed_emotion
            return False
        changed = self._smoothed_transition_cursor != smoothed_emotion
        self._smoothed_transition_cursor = smoothed_emotion
        return changed

    def _analyze_quality(
        self,
        frame: Frame,
        bbox: BoundingBox | None,
        result: PredictionResult,
    ) -> QualityResult | None:
        if not self._settings.quality_enabled or bbox is None:
            return None

        frame_width, frame_height = frame.resolution
        face_crop = frame.image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]

        feature_vector = None
        if result.feature_vector is not None and len(result.feature_vector) == FEATURE_VECTOR_LENGTH:
            feature_vector = FeatureVector(values=np.asarray(result.feature_vector, dtype=np.float64))

        return self._quality_analyzer.analyze(
            face_crop=face_crop,
            bounding_box=bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            feature_vector=feature_vector,
        )

    def _reset_history_only(self) -> None:
        """Reset temporal/continuity state without touching the jitter counters (spec sec 33)."""
        self._smoother.reset()
        self._previous_bbox = None
        self._last_smoothed_probabilities = None
        self._last_smoothed_emotion = None
        self._last_smoothed_confidence = None
        self._smoothed_transition_cursor = None
