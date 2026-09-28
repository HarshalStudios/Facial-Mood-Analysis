"""FastAPI dependency providers and process-local inference runtime."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock

from app.core.config import Settings, get_settings
from app.core.exceptions import BackendError
from app.prediction.loader import load_fusion_model
from app.prediction.predictor import RealTimePredictor
from app.temporal.engine import TemporalPredictionEngine


def settings_dependency() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def get_predictor() -> RealTimePredictor:
    """Create the heavy frame-level predictor once per API process."""
    return RealTimePredictor(get_settings())


@lru_cache(maxsize=1)
def get_temporal_engine() -> TemporalPredictionEngine:
    """Create one stateful temporal engine for a single interactive session."""
    return TemporalPredictionEngine(get_settings())


_runtime_lock = Lock()


def model_status() -> dict:
    """Return artifact availability without loading heavyweight models."""
    settings = get_settings()
    model_path = settings.fusion_model_path
    scaler_path = settings.fusion_scaler_path
    metadata_path = model_path.parent / "fusion_model_metadata.json"
    return {
        "model_artifact": str(model_path),
        "model_exists": model_path.exists(),
        "scaler_artifact": str(scaler_path),
        "scaler_exists": scaler_path.exists(),
        "metadata_artifact": str(metadata_path),
        "metadata_exists": metadata_path.exists(),
        "ready": model_path.exists() and scaler_path.exists(),
    }


def reset_runtime() -> None:
    """Clear cached predictors/temporal state, primarily for tests/admin restart."""
    with _runtime_lock:
        get_temporal_engine.cache_clear()
        get_predictor.cache_clear()
