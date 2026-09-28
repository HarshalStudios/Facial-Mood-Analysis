"""
Data contract for Phase 8 temporal + quality prediction results.

`TemporalPredictionResult` is what `app.temporal.engine.TemporalPredictionEngine.process()`
returns for every frame. Mirrors the shape of Phase 7's `PredictionResult`
(same `frame_id`/`timestamp`/`error_code` conventions) but adds the
raw-vs-smoothed split (spec sec 25-26) and the quality report, and
deliberately never drops the raw output even when a smoothed one is
also present -- Phase 9 evaluation needs both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.schemas.quality import QualityResult


@dataclass
class TemporalPredictionResult:
    frame_id: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    face_detected: bool = False
    # Final, post-quality-policy availability of a *usable* prediction
    # (spec sec 21/24) -- False on no-face, on an unavailable Phase 7
    # result, or on a severe quality failure in "strict" quality mode.
    prediction_available: bool = False

    raw_emotion: str | None = None
    raw_confidence: float | None = None
    raw_probabilities: dict[str, float] | None = None

    smoothed_emotion: str | None = None
    smoothed_confidence: float | None = None
    smoothed_probabilities: dict[str, float] | None = None

    # Number of valid predictions currently held in the temporal window
    # (spec sec 27) -- useful for understanding startup/reset behavior.
    history_length: int = 0

    quality: QualityResult | None = None

    error_code: str | None = None
    error_message: str | None = None

    def to_api_dict(self) -> dict:
        """JSON-serializable shape, analogous to `PredictionResult.to_api_dict()`."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "face_detected": self.face_detected,
            "prediction_available": self.prediction_available,
            "raw": {
                "emotion": self.raw_emotion,
                "confidence": self.raw_confidence,
                "probabilities": self.raw_probabilities,
            },
            "stable": {
                "emotion": self.smoothed_emotion,
                "confidence": self.smoothed_confidence,
                "probabilities": self.smoothed_probabilities,
            },
            "history_length": self.history_length,
            "quality": self.quality.to_json() if self.quality is not None else None,
            "error_code": self.error_code,
        }
