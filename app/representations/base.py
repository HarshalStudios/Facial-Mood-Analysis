"""
RepresentationGenerator interface.

Implemented in Phase 3 (grayscale, CLAHE, Canny, Sobel, LBP). Downstream
consumers (FeatureExtractor) depend only on the Representations contract
this produces, not on how each representation is computed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.schemas.representations import Representations


class RepresentationGenerator(ABC):
    """Abstract base class for turning a face crop into multiple representations."""

    @abstractmethod
    def generate(self, face_crop: np.ndarray) -> Representations:
        """Given a BGR face crop, return the full set of representations."""
        raise NotImplementedError
