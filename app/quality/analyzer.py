"""
StandardQualityAnalyzer (Phase 8 spec sec 13-21).

An engineering-heuristic quality gate, deliberately kept simple (spec
sec 19): a face-size ratio, plus the brightness/contrast/sharpness
already computed as part of the 22D feature vector (Phase 4), each
checked against a configurable threshold. It never invents a
"scientifically validated" confidence score, and it never silently
turns a quality problem into a rejected prediction -- see
`app.schemas.quality.SEVERE_WARNINGS` for exactly which observations
are even capable of doing that, and only in "strict" `quality_mode`.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.quality.base import QualityAnalyzer
from app.schemas.face import BoundingBox
from app.schemas.feature_vector import FeatureVector
from app.schemas.quality import (
    SEVERE_WARNINGS,
    VALID_QUALITY_MODES,
    QualityResult,
    QualityWarning,
)

logger = get_logger(__name__)

# Simple, documented heuristic: lose a fixed amount of "score" per
# warning present. Not a scientifically validated confidence measure
# (spec sec 19) -- purely for human-readable debug/demo display.
_SCORE_PENALTY_PER_WARNING = 0.25


def _brightness_contrast_sharpness_from_crop(face_crop: np.ndarray) -> tuple[float, float, float]:
    """
    Fallback computation used only when no `FeatureVector` is supplied
    (e.g. a standalone/unit-test call to the analyzer). When a feature
    vector *is* available, its already-computed values are reused
    instead (spec sec 15-17) rather than recomputing a second,
    independent set of numbers.
    """
    if face_crop.ndim == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_crop
    gray_f = gray.astype(np.float64)
    brightness = float(gray_f.mean())
    contrast = float(gray_f.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, contrast, sharpness


class StandardQualityAnalyzer(QualityAnalyzer):
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if self._settings.quality_mode not in VALID_QUALITY_MODES:
            raise ConfigurationError(
                f"Unknown quality_mode '{self._settings.quality_mode}', "
                f"expected one of {sorted(VALID_QUALITY_MODES)}"
            )

    def analyze(
        self,
        face_crop: np.ndarray,
        bounding_box: BoundingBox,
        frame_width: int,
        frame_height: int,
        feature_vector: FeatureVector | None = None,
    ) -> QualityResult:
        s = self._settings
        warnings: list[str] = []

        if feature_vector is not None:
            brightness = feature_vector["brightness"]
            contrast = feature_vector["contrast"]
            sharpness = feature_vector["sharpness"]
        else:
            brightness, contrast, sharpness = _brightness_contrast_sharpness_from_crop(face_crop)

        frame_area = max(frame_width * frame_height, 1)
        face_area = max(bounding_box.width * bounding_box.height, 0)
        face_area_ratio = face_area / frame_area

        if face_area_ratio < s.min_face_area_ratio:
            warnings.append(QualityWarning.FACE_TOO_SMALL)
        if brightness < s.min_brightness:
            warnings.append(QualityWarning.LOW_BRIGHTNESS)
        if brightness > s.max_brightness:
            warnings.append(QualityWarning.HIGH_BRIGHTNESS)
        if sharpness < s.min_sharpness:
            warnings.append(QualityWarning.LOW_SHARPNESS)
        if contrast < s.min_contrast:
            warnings.append(QualityWarning.LOW_CONTRAST)

        metrics = {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "sharpness": float(sharpness),
            "face_area_ratio": float(face_area_ratio),
        }

        quality_score = max(0.0, 1.0 - _SCORE_PENALTY_PER_WARNING * len(warnings))

        has_severe = any(w in SEVERE_WARNINGS for w in warnings)
        if s.quality_mode == "strict":
            is_acceptable = not has_severe
        else:  # "advisory" (default): informational only, never rejects on its own (spec sec 21)
            is_acceptable = True

        if not warnings:
            status = "good"
        elif is_acceptable:
            status = "warning"
        else:
            status = "reject"

        if warnings:
            logger.debug(
                "Quality warnings (ratio=%.4f brightness=%.1f contrast=%.1f sharpness=%.1f "
                "mode=%s): %s -> acceptable=%s",
                face_area_ratio, brightness, contrast, sharpness, s.quality_mode, warnings, is_acceptable,
            )

        return QualityResult(
            is_acceptable=is_acceptable,
            warnings=warnings,
            metrics=metrics,
            quality_score=quality_score,
            status=status,
        )
