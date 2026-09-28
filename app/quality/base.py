"""
QualityAnalyzer interface (Phase 8).

Evaluates whether an input face crop/frame is good enough to trust a
prediction (brightness, contrast, sharpness, face size) before the
result is surfaced to the user. See `app.quality.analyzer` for the
concrete implementation and `app.schemas.quality.QualityResult` for
the return contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.schemas.face import BoundingBox
from app.schemas.feature_vector import FeatureVector
from app.schemas.quality import QualityResult


class QualityAnalyzer(ABC):
    """Abstract base class for input-quality validation."""

    @abstractmethod
    def analyze(
        self,
        face_crop: np.ndarray,
        bounding_box: BoundingBox,
        frame_width: int,
        frame_height: int,
        feature_vector: FeatureVector | None = None,
    ) -> QualityResult:
        """
        Return a quality report for the given face crop.

        `feature_vector`, when available, lets the analyzer reuse the
        already-computed brightness/contrast/sharpness features (spec
        sec 15-17) instead of recomputing them from `face_crop`.
        """
        raise NotImplementedError
