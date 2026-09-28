"""
EmotionModel factory (Phase 4).

Selects a concrete EmotionModel by `settings.emotion_model_backend`, so
callers (the real-time pipeline in Phase 7, dataset generation in
Phase 5) depend on this one function rather than importing a specific
provider class directly.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.emotion.base import EmotionModel
from app.emotion.mock_provider import FixedEmotionModel

_BACKENDS = {"deepface", "fer2013_cnn", "fer2013_linear", "mock"}


def get_emotion_model(settings: Settings | None = None) -> EmotionModel:
    settings = settings or get_settings()
    backend = settings.emotion_model_backend

    if backend == "deepface":
        from app.emotion.deepface_provider import DeepFaceEmotionModel

        return DeepFaceEmotionModel(settings)
    if backend == "fer2013_cnn":
        from app.emotion.fer2013_cnn_provider import FER2013CNNEmotionModel
        model_path = getattr(settings, "fer2013_emotion_model_path", None)
        if not model_path:
            raise ValueError("FMA_FER2013_EMOTION_MODEL_PATH is required for backend 'fer2013_cnn'")
        return FER2013CNNEmotionModel(model_path)
    if backend == "fer2013_linear":
        from app.emotion.fer2013_linear_provider import FER2013LinearEmotionModel
        model_path = getattr(settings, "fer2013_linear_emotion_model_path", None)
        if not model_path:
            raise ValueError("FMA_FER2013_LINEAR_EMOTION_MODEL_PATH is required for backend 'fer2013_linear'")
        return FER2013LinearEmotionModel(model_path)
    if backend == "mock":
        return FixedEmotionModel()

    raise ValueError(f"Unknown emotion_model_backend '{backend}'. Valid options: {sorted(_BACKENDS)}")
