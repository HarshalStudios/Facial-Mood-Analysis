"""Output contracts for prediction and quality results (used from Phase 7 onward)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmotionPrediction:
    emotion: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class QualityReport:
    brightness: float
    contrast: float
    sharpness: float
    status: str  # e.g. "good" | "poor"
