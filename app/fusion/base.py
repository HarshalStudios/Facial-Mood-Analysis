"""
FusionModel interface.

Implemented in Phase 6 (Logistic Regression trained on the 22D feature
vector). Kept abstract so the ablation engine (Phase 9) can swap in
alternative fusion strategies (heuristic fusion, other classifiers)
without touching the prediction engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.feature_vector import FeatureVector
from app.schemas.prediction import EmotionPrediction


class FusionModel(ABC):
    """Abstract base class for the feature-level fusion classifier."""

    @abstractmethod
    def predict(self, feature_vector: FeatureVector) -> EmotionPrediction:
        """Run inference on a single feature vector."""
        raise NotImplementedError

    @abstractmethod
    def load(self, model_path: str, scaler_path: str) -> None:
        """Load a trained model + scaler from disk."""
        raise NotImplementedError
