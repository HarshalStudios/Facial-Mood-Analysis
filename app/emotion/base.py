"""
EmotionModel interface.

Implemented starting Phase 4/6, currently planned around DeepFace.
Produces the 7 emotion probabilities that become part of the 22D
feature vector — this is NOT the final fusion model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmotionModel(ABC):
    """Abstract base class for the deep-learning emotion probability model."""

    @abstractmethod
    def predict_probabilities(self, face_crop: np.ndarray) -> dict[str, float]:
        """Return a mapping of emotion label -> probability (7 classes)."""
        raise NotImplementedError
