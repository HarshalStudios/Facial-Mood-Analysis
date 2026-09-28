"""
DeepFace-backed EmotionModel implementation (Phase 4).

Responsible only for: given a BGR face crop, return a
{emotion_label: probability} dict. Everything else (label ordering,
validation, becoming part of the 22D feature vector) lives in
app.emotion.validation / app.features — this class stays a thin,
swappable provider behind the EmotionModel interface (Phase 1).

IMPORTANT — not exercised in this sandbox: this environment has no
network access, so the `deepface` package (and the TensorFlow/Keras
stack it depends on) could not be installed or verified here. The
integration test in tests/test_deepface_integration.py documents this
and skips itself rather than fabricating a passing result. Verify this
class on a machine with `pip install deepface` and a real webcam frame
before relying on it.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelLoadError, PredictionError
from app.core.logging import get_logger
from app.emotion.base import EmotionModel

logger = get_logger(__name__)


class DeepFaceEmotionModel(EmotionModel):
    """
    EmotionModel implementation backed by DeepFace's "emotion" action.

    Model loading (DeepFace's internal Keras model weights) happens
    lazily on first use and is cached on this instance — construct one
    DeepFaceEmotionModel at application/process startup (Phase 7) and
    reuse it for every frame; do not construct a new one per frame.
    DeepFace's own module-level model cache would technically survive
    repeated construction too, but re-running its detector-backend setup
    and import machinery per frame is still wasted work, so this class
    does its own one-time lazy init behind a lock for thread-safety.

    Color-space contract: this project stores face crops as BGR, uint8
    (Phase 2). DeepFace's `analyze()` expects RGB-ordered arrays when
    given a numpy array directly. This class converts BGR -> RGB
    explicitly at this boundary and nowhere else.

    Probability scale: DeepFace's "emotion" action returns a dict of
    PERCENTAGES per class that sum to ~100 (e.g. {"happy": 87.3, ...}),
    not 0-1 probabilities. This class divides by 100 before returning,
    so predict_probabilities() always yields values in [0, 1] as the
    EmotionModel interface and app.emotion.validation expect.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._deepface_module = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._deepface_module is not None:
            return
        with self._lock:
            if self._deepface_module is not None:
                return
            try:
                from deepface import DeepFace  # imported lazily: heavy, optional dep
            except ImportError as exc:
                raise ModelLoadError(
                    "The 'deepface' package is not installed. Install it "
                    "(see requirements.txt) to use DeepFaceEmotionModel; "
                    "for tests, use app.emotion.mock_provider instead.",
                    details={"import_error": str(exc)},
                ) from exc
            self._deepface_module = DeepFace
            logger.info("DeepFace module loaded (emotion model weights load lazily on first analyze() call).")

    def predict_probabilities(self, face_crop: np.ndarray) -> dict[str, float]:
        """
        `face_crop` is a BGR, uint8, already-cropped face (i.e.
        Representations.rgb from Phase 3, which — despite the field name
        — is still BGR; see that module's docstring). Converted to RGB
        here before calling DeepFace.
        """
        self._ensure_loaded()

        if face_crop is None or face_crop.size == 0:
            raise PredictionError("face_crop is None or empty; cannot run emotion inference")

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        try:
            # detector_backend="skip": Phase 2 already detected/cropped
            # the face, so DeepFace must not re-run its own detector
            # (which could fail or crop differently) — it should treat
            # `rgb` as an already-aligned face image.
            result = self._deepface_module.analyze(
                rgb,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
        except Exception as exc:  # DeepFace raises assorted exception types
            raise PredictionError(
                "DeepFace emotion inference failed", details={"error": str(exc)}
            ) from exc

        if isinstance(result, list):
            if not result:
                raise PredictionError("DeepFace returned an empty result list")
            result = result[0]

        raw_scores = result.get("emotion")
        if not raw_scores:
            raise PredictionError(
                "DeepFace result did not contain an 'emotion' field", details={"keys": list(result.keys())}
            )

        # DeepFace scores are percentages (sum ~= 100); normalize to [0, 1].
        return {str(label).lower(): float(score) / 100.0 for label, score in raw_scores.items()}
