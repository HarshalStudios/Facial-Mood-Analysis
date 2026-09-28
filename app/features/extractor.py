"""
StandardFeatureExtractor (Phase 4).

The single canonical implementation of FeatureExtractor. Must be reused
unchanged for both real-time inference (Phase 7) and FER2013 dataset
generation (Phase 5) — see Phase 4 spec sec. 36 ("training/inference
consistency"). Do not create a second, parallel implementation.

Pipeline:

    Representations (Phase 3) + EmotionModel (this phase)
        -> 7 deep emotion probabilities   (app.emotion.validation)
        -> 10 LBP histogram bins          (app.features.lbp_histogram)
        -> edge density                   (app.features.edge_density)
        -> gradient energy, brightness,
           contrast, sharpness            (app.features.image_quality)
        -> concatenate in FEATURE_NAMES order
        -> validate -> FeatureVector -> FeatureExtractionResult
"""

from __future__ import annotations

import numpy as np

from app.core.config import Settings, get_settings
from app.core.exceptions import FeatureExtractionError
from app.core.logging import get_logger
from app.emotion.base import EmotionModel
from app.emotion.validation import validate_and_order_emotion_probabilities
from app.features.base import FeatureExtractor
from app.features.edge_density import compute_edge_density
from app.features.image_quality import (
    compute_brightness,
    compute_contrast,
    compute_gradient_energy,
    compute_sharpness,
)
from app.features.lbp_histogram import compute_lbp_histogram
from app.schemas.feature_extraction import FeatureExtractionResult
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH, FeatureVector
from app.schemas.representations import Representations

logger = get_logger(__name__)


class StandardFeatureExtractor(FeatureExtractor):
    def __init__(self, emotion_model: EmotionModel, settings: Settings | None = None) -> None:
        self._emotion_model = emotion_model
        self._settings = settings or get_settings()

    def extract(self, representations: Representations) -> FeatureExtractionResult:
        settings = self._settings

        self._require(representations.grayscale, "grayscale")
        self._require(representations.canny, "canny")
        self._require(representations.lbp, "lbp")
        # Note: representations.sobel is NOT required here — gradient_energy
        # recomputes Sobel from grayscale directly (see
        # app/features/image_quality.py docstring for why), so this
        # feature stays available even when enable_sobel=False.

        emotion_raw = self._emotion_model.predict_probabilities(representations.rgb)
        emotion_values = validate_and_order_emotion_probabilities(emotion_raw)

        lbp_hist = compute_lbp_histogram(representations.lbp, settings)

        edge_density = compute_edge_density(representations.canny)
        gradient_energy = compute_gradient_energy(representations.grayscale, settings)
        brightness = compute_brightness(representations.grayscale)
        contrast = compute_contrast(representations.grayscale)
        sharpness = compute_sharpness(representations.grayscale)

        values = np.concatenate(
            [
                emotion_values,
                lbp_hist,
                np.array(
                    [edge_density, gradient_energy, brightness, contrast, sharpness],
                    dtype=np.float64,
                ),
            ]
        )

        if values.shape != (FEATURE_VECTOR_LENGTH,):
            raise FeatureExtractionError(
                "Assembled feature vector has the wrong shape",
                details={"shape": values.shape, "expected": (FEATURE_VECTOR_LENGTH,)},
            )
        if not np.all(np.isfinite(values)):
            bad = [i for i, v in enumerate(values) if not np.isfinite(v)]
            raise FeatureExtractionError(
                "Assembled feature vector contains NaN/Inf value(s)", details={"bad_indices": bad}
            )

        try:
            vector = FeatureVector(values=values)
        except ValueError as exc:
            raise FeatureExtractionError(str(exc)) from exc

        metadata = {
            "emotion_provider": type(self._emotion_model).__name__,
            "lbp_params": {
                "P": settings.lbp_n_points,
                "R": settings.lbp_radius,
                "method": settings.lbp_method,
                "num_bins": settings.lbp_num_bins,
            },
            "gradient_energy_sobel_kernel_size": settings.sobel_kernel_size,
            "representations_used": [
                name for name in ("rgb", "grayscale", "canny", "lbp") if getattr(representations, name) is not None
            ],
        }

        return FeatureExtractionResult(vector=vector, features=vector.as_dict(), metadata=metadata)

    @staticmethod
    def _require(value, name: str) -> None:
        if value is None:
            raise FeatureExtractionError(
                f"Representations.{name} is required for feature extraction but is None "
                f"(was it disabled via config for an ablation run?)",
                details={"missing_representation": name},
            )
