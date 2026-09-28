"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import model_status
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }


@router.get("/ready")
def ready() -> dict:
    """Readiness reflects whether required inference artifacts exist."""
    settings = get_settings()
    artifacts = model_status()
    return {
        "status": "ready" if artifacts["ready"] else "not_ready",
        "app": settings.app_name,
        "model": artifacts,
    }
