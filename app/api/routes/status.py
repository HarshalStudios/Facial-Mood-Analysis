"""Backend readiness and runtime status endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import model_status
from app.core.config import get_settings
from app.schemas.feature_vector import FEATURE_NAMES, FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_VERSION
from app.emotion.validation import CANONICAL_EMOTION_LABELS

router = APIRouter(tags=["status"])


@router.get("/status")
def status() -> dict:
    settings = get_settings()
    artifacts = model_status()
    return {
        "status": "ready" if artifacts["ready"] else "not_ready",
        "environment": settings.environment,
        "feature_schema": {
            "version": FEATURE_VECTOR_VERSION,
            "count": FEATURE_VECTOR_LENGTH,
            "names": FEATURE_NAMES,
        },
        "classes": CANONICAL_EMOTION_LABELS,
        "model": artifacts,
        "temporal": {
            "enabled": settings.temporal_enabled,
            "method": settings.smoothing_method,
            "window_size": settings.smoothing_window_size,
            "ema_alpha": settings.ema_alpha,
        },
    }
