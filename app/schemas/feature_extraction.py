"""
Result object returned by FeatureExtractor.extract() (Phase 4).

Wraps the canonical FeatureVector with named-value and metadata views so
callers that need individual metrics (debugging, frontend, research
reporting) don't have to re-derive them from the raw array, while the
model-facing ordering still comes from the single FeatureVector /
FEATURE_NAMES contract in app.schemas.feature_vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.feature_vector import FeatureVector


@dataclass
class FeatureExtractionResult:
    vector: FeatureVector
    features: dict[str, float]
    metadata: dict = field(default_factory=dict)
