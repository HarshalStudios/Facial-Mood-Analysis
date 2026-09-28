# Backend API Contract

Base URL during local development: `http://localhost:8000`

## GET /health

Liveness only. Does not require model artifacts.

## GET /ready

Reports whether the Phase 6 model and scaler files exist.

## GET /status

Returns the active 22D feature schema, canonical classes, model artifact state,
and temporal configuration.

## POST /predict/image

Request body: raw image bytes.

Recommended header:

```text
Content-Type: image/jpeg
```

Supported formats are those understood by OpenCV `imdecode`.

Response shape:

```json
{
  "frame_id": 0,
  "timestamp": "...",
  "face_detected": true,
  "prediction_available": true,
  "number_of_faces": 1,
  "primary_face_bbox": {
    "x": 10,
    "y": 20,
    "width": 150,
    "height": 150
  },
  "predicted_emotion": "happy",
  "confidence": 0.0,
  "probabilities": {
    "angry": 0.0,
    "disgust": 0.0,
    "fear": 0.0,
    "happy": 0.0,
    "sad": 0.0,
    "surprise": 0.0,
    "neutral": 0.0
  },
  "error_code": null,
  "error_message": null,
  "processing_time_ms": 0.0,
  "model_version": "...",
  "feature_schema_version": "..."
}
```

The numerical values above are schema examples only, not project results.

## POST /predict/base64

JSON:

```json
{
  "image": "<base64 bytes>",
  "frame_id": 123
}
```

A `data:image/jpeg;base64,...` prefix is also accepted.

## POST /predict/temporal/image

Same image-body convention as `/predict/image`, but adds Phase 8:

- raw emotion/probabilities
- smoothed emotion/probabilities
- quality result
- history length

The current implementation keeps temporal state per API process. A production
multi-user deployment should create a separate temporal engine per user/session.

## POST /predict/temporal/reset

Clears the process-local temporal history.

## Error behavior

- `400`: malformed or unsupported image payload.
- `503`: model/artifact/inference backend unavailable.
- No face: HTTP 200 with `prediction_available=false`; no emotion is invented.

## Frontend integration

The frontend can capture a browser camera frame to a JPEG/WebP Blob and POST
that Blob directly to `/predict/image`. It does not need to understand the
internal OpenCV, DeepFace, feature extraction, or Logistic Regression pipeline.
