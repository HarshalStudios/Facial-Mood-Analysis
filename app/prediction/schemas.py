"""
Data contract for Phase 7 frame-level prediction results.

`PredictionResult` is the single structured object `RealTimePredictor.predict()`
returns. It is a plain dataclass (no OpenCV/numpy image data embedded)
so it is safe to log, hand to a CLI display layer, or -- from Phase 10
onward -- serialize into an API/WebSocket response without further
filtering. `to_api_dict()` sketches that eventual serialized shape (see
Phase 7 spec sec. 22/42) but Phase 7 itself does not expose any API.

`feature_vector` is only populated when the predictor is explicitly
constructed with `include_feature_vector=True` (default False) -- per
spec sec. 21, a result destined for transmission should not carry the
raw 22D array unless something actually needs it (e.g. a debug CLI or a
future research/logging consumer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.schemas.face import BoundingBox


@dataclass
class TimingBreakdown:
    """Per-stage latency, populated as far as the pipeline actually got (spec sec. 28)."""

    face_detection_ms: float | None = None
    representation_ms: float | None = None
    feature_extraction_ms: float | None = None
    model_inference_ms: float | None = None
    total_ms: float | None = None

    def to_json(self) -> dict:
        return {
            "face_detection_ms": self.face_detection_ms,
            "representation_ms": self.representation_ms,
            "feature_extraction_ms": self.feature_extraction_ms,
            "model_inference_ms": self.model_inference_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class PredictionResult:
    """Structured, frame-level result of `RealTimePredictor.predict()`."""

    frame_id: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    face_detected: bool = False
    prediction_available: bool = False

    number_of_faces: int = 0
    primary_face_bbox: BoundingBox | None = None

    predicted_emotion: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None

    feature_vector: list[float] | None = None

    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    processing_time_ms: float | None = None

    error_code: str | None = None
    error_message: str | None = None

    model_version: str | None = None
    feature_schema_version: str | None = None

    def to_api_dict(self) -> dict:
        """
        A JSON-serializable shape close to what a future API (Phase 10)
        would expose (spec sec. 22/42). Never includes `feature_vector`
        or internal timing -- those stay for internal/debug consumers.
        """
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "face_detected": self.face_detected,
            "prediction_available": self.prediction_available,
            "number_of_faces": self.number_of_faces,
            "emotion": self.predicted_emotion,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "processing_time_ms": self.processing_time_ms,
            "error_code": self.error_code,
        }
