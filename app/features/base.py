"""
FeatureExtractor interface.

Implemented in Phase 4. Must always return a result built around the
fixed FeatureVector ordering defined in app.schemas.feature_vector.

Phase 1 originally sketched this returning a bare FeatureVector. Phase
4 (its first real implementation) widens the return type to
FeatureExtractionResult, which carries the same FeatureVector plus a
named-value dict and metadata (Phase 4 spec sec. 27/39) — useful for
debugging, the future frontend, and research reporting. This is safe to
change now because no concrete implementation existed yet and no saved
model depends on this Python-level return type (only on FeatureVector's
internal (22,) ordering, which is unchanged).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.feature_extraction import FeatureExtractionResult
from app.schemas.representations import Representations


class FeatureExtractor(ABC):
    """Abstract base class for building the 22D feature vector."""

    @abstractmethod
    def extract(self, representations: Representations) -> FeatureExtractionResult:
        """Turn a Representations bundle into a validated FeatureExtractionResult."""
        raise NotImplementedError
