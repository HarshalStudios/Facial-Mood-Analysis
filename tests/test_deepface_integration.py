"""
INTEGRATION TEST — not a unit test.

Exercises the real DeepFaceEmotionModel against an actual DeepFace
model (requires the `deepface` package, its TensorFlow/Keras
dependency stack, and — on first run — network access to download
model weights). Skips itself when `deepface` is not importable rather
than fabricating a pass/fail.

As of this Phase 4 implementation, this test has NOT been run
successfully in the development sandbox: that environment has no
network access, so `deepface` could not be installed there. Run this
on a machine with `pip install deepface` and a real face photo before
trusting DeepFaceEmotionModel in production.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import Settings
from app.emotion.deepface_provider import DeepFaceEmotionModel
from app.emotion.validation import CANONICAL_EMOTION_LABELS, validate_and_order_emotion_probabilities

deepface = pytest.importorskip(
    "deepface", reason="deepface is not installed; this integration test requires it (see module docstring)."
)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def _synthetic_face_like_image(size: int = 224) -> np.ndarray:
    # Not a real face — DeepFace with detector_backend="skip" will run
    # its emotion model on whatever it's given regardless, so this is
    # sufficient to check the plumbing (call succeeds, returns 7 valid
    # probabilities), not to check emotion-recognition *accuracy*.
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_deepface_returns_seven_valid_probabilities(settings):
    model = DeepFaceEmotionModel(settings)
    raw = model.predict_probabilities(_synthetic_face_like_image())
    ordered = validate_and_order_emotion_probabilities(raw)
    assert ordered.shape == (7,)
    assert set(k.lower() for k in raw.keys()) >= set(CANONICAL_EMOTION_LABELS)


def test_deepface_model_is_reused_across_calls(settings):
    model = DeepFaceEmotionModel(settings)
    model.predict_probabilities(_synthetic_face_like_image())
    first_module = model._deepface_module
    model.predict_probabilities(_synthetic_face_like_image())
    assert model._deepface_module is first_module
