import base64
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_predictor
from app.prediction.schemas import PredictionResult, TimingBreakdown
from app.schemas.face import BoundingBox
from datetime import datetime, timezone


class FakePredictor:
    def predict(self, frame):
        return PredictionResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            face_detected=True,
            prediction_available=True,
            number_of_faces=1,
            primary_face_bbox=BoundingBox(x=5, y=5, width=20, height=20),
            predicted_emotion="happy",
            confidence=0.8,
            probabilities={
                "angry": 0.02, "disgust": 0.01, "fear": 0.03,
                "happy": 0.80, "sad": 0.04, "surprise": 0.05, "neutral": 0.05
            },
            timing=TimingBreakdown(total_ms=2.0),
            processing_time_ms=2.0,
            model_version="test",
            feature_schema_version="1.0",
        )


def _jpeg():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:45, 20:45] = 255
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_health_and_status():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/status").status_code == 200
    assert client.get("/ready").status_code == 200


def test_predict_image_with_injected_predictor():
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    try:
        client = TestClient(app)
        response = client.post("/predict/image", content=_jpeg(), headers={"content-type": "image/jpeg"})
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_emotion"] == "happy"
        assert body["confidence"] == 0.8
        assert body["primary_face_bbox"]["width"] == 20
    finally:
        app.dependency_overrides.clear()


def test_predict_base64_with_injected_predictor():
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    try:
        client = TestClient(app)
        encoded = base64.b64encode(_jpeg()).decode()
        response = client.post("/predict/base64", json={"image": encoded, "frame_id": 12})
        assert response.status_code == 200
        assert response.json()["frame_id"] == 12
    finally:
        app.dependency_overrides.clear()


def test_predict_rejects_invalid_image():
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    try:
        client = TestClient(app)
        response = client.post("/predict/image", content=b"not an image")
        assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()
