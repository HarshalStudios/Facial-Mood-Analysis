"""HTTP inference endpoints.

The API accepts an image body directly (JPEG/PNG/WebP/etc.) so browser
clients can POST a Blob without requiring multipart/form-data or an
additional multipart parser dependency.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_predictor, get_temporal_engine
from app.core.exceptions import BackendError
from app.core.logging import get_logger
from app.prediction.predictor import RealTimePredictor
from app.prediction.schemas import PredictionResult
from app.schemas.frame import Frame
from app.temporal.engine import TemporalPredictionEngine

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


class Base64ImageRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image bytes. A data: URL prefix is also accepted.")
    frame_id: int | None = Field(default=None, ge=0)


def _decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="Request body contains no image bytes.")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Body is not a supported/valid image.")
    return image


def _make_frame(image: np.ndarray, frame_id: int | None = None) -> Frame:
    return Frame(
        image=image,
        frame_id=0 if frame_id is None else frame_id,
        timestamp=datetime.now(timezone.utc),
    )


def _prediction_payload(result: PredictionResult) -> dict:
    payload = {
        "frame_id": result.frame_id,
        "timestamp": result.timestamp.isoformat(),
        "face_detected": result.face_detected,
        "prediction_available": result.prediction_available,
        "number_of_faces": result.number_of_faces,
        "predicted_emotion": result.predicted_emotion,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "processing_time_ms": result.processing_time_ms,
        "model_version": result.model_version,
        "feature_schema_version": result.feature_schema_version,
    }
    if result.primary_face_bbox is not None:
        b = result.primary_face_bbox
        payload["primary_face_bbox"] = {
            "x": b.x, "y": b.y, "width": b.width, "height": b.height
        }
    else:
        payload["primary_face_bbox"] = None
    return payload


@router.post("/image")
async def predict_image(
    request: Request,
    predictor: RealTimePredictor = Depends(get_predictor),
) -> dict:
    """Predict one image frame using the trained Phase 6 model."""
    try:
        image = _decode_image(await request.body())
        result = predictor.predict(_make_frame(image))
        return _prediction_payload(result)
    except BackendError as exc:
        logger.error("Prediction request failed: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        ) from exc


@router.post("/base64")
def predict_base64(
    body: Base64ImageRequest,
    predictor: RealTimePredictor = Depends(get_predictor),
) -> dict:
    """Predict one base64-encoded image, convenient for JSON-only clients."""
    value = body.image
    if "," in value and value.startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from exc

    image = _decode_image(raw)
    try:
        result = predictor.predict(_make_frame(image, body.frame_id))
        return _prediction_payload(result)
    except BackendError as exc:
        logger.error("Prediction request failed: %s", exc.message)
        raise HTTPException(status_code=503, detail=exc.message) from exc


@router.post("/temporal/image")
async def predict_temporal_image(
    request: Request,
    temporal_engine: TemporalPredictionEngine = Depends(get_temporal_engine),
) -> dict:
    """Run Phase 7 prediction plus Phase 8 temporal/quality processing.

    The temporal engine is intentionally process-local and stateful. It is
    appropriate for one interactive stream; production multi-user deployment
    should create one engine per session rather than sharing state globally.
    """
    try:
        image = _decode_image(await request.body())
        result = temporal_engine.process(_make_frame(image))
        return {
            "frame_id": result.frame_id,
            "timestamp": result.timestamp.isoformat(),
            "face_detected": result.face_detected,
            "prediction_available": result.prediction_available,
            "raw_emotion": result.raw_emotion,
            "raw_confidence": result.raw_confidence,
            "raw_probabilities": result.raw_probabilities,
            "smoothed_emotion": result.smoothed_emotion,
            "smoothed_confidence": result.smoothed_confidence,
            "smoothed_probabilities": result.smoothed_probabilities,
            "history_length": result.history_length,
            "quality": result.quality.__dict__ if result.quality else None,
            "error_code": result.error_code,
            "error_message": result.error_message,
        }
    except BackendError as exc:
        logger.error("Temporal prediction request failed: %s", exc.message)
        raise HTTPException(status_code=503, detail=exc.message) from exc


@router.post("/temporal/reset")
def reset_temporal(
    temporal_engine: TemporalPredictionEngine = Depends(get_temporal_engine),
) -> dict:
    temporal_engine.reset()
    return {"status": "ok", "message": "Temporal prediction state reset."}
