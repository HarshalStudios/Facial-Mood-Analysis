"""
Tests for StandardQualityAnalyzer (Phase 8 spec sec 13-21, 42-43).

Uses a duck-typed `SimpleNamespace` stand-in for `Settings` (same
pattern as `tests/test_predictor.py`) and constructs `FeatureVector`
instances directly so brightness/contrast/sharpness are controlled
exactly, rather than depending on real image content.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.core.exceptions import ConfigurationError
from app.quality.analyzer import StandardQualityAnalyzer
from app.schemas.face import BoundingBox
from app.schemas.feature_vector import FEATURE_NAMES, FEATURE_VECTOR_LENGTH, FeatureVector
from app.schemas.quality import QualityWarning


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        quality_mode="advisory",
        min_face_area_ratio=0.05,
        min_brightness=40.0,
        max_brightness=220.0,
        min_sharpness=60.0,
        min_contrast=20.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _feature_vector(brightness: float, contrast: float, sharpness: float) -> FeatureVector:
    values = np.full(FEATURE_VECTOR_LENGTH, 0.1, dtype=np.float64)
    values[FEATURE_NAMES.index("brightness")] = brightness
    values[FEATURE_NAMES.index("contrast")] = contrast
    values[FEATURE_NAMES.index("sharpness")] = sharpness
    return FeatureVector(values=values)


def _bbox(w: int, h: int) -> BoundingBox:
    return BoundingBox(x=0, y=0, width=w, height=h)


def _crop(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_rejects_unknown_quality_mode_at_construction():
    with pytest.raises(ConfigurationError):
        StandardQualityAnalyzer(settings=_settings(quality_mode="not_a_real_mode"))


def test_good_quality_produces_no_warnings():
    analyzer = StandardQualityAnalyzer(settings=_settings())
    fv = _feature_vector(brightness=128.0, contrast=50.0, sharpness=150.0)
    result = analyzer.analyze(
        face_crop=_crop(200, 200),
        bounding_box=_bbox(200, 200),
        frame_width=400,
        frame_height=400,
        feature_vector=fv,
    )
    assert result.warnings == []
    assert result.is_acceptable is True
    assert result.status == "good"
    assert result.metrics["brightness"] == pytest.approx(128.0)


def test_low_brightness_flagged():
    analyzer = StandardQualityAnalyzer(settings=_settings())
    fv = _feature_vector(brightness=10.0, contrast=50.0, sharpness=150.0)
    result = analyzer.analyze(_crop(200, 200), _bbox(200, 200), 400, 400, feature_vector=fv)
    assert QualityWarning.LOW_BRIGHTNESS in result.warnings


def test_high_brightness_flagged():
    analyzer = StandardQualityAnalyzer(settings=_settings())
    fv = _feature_vector(brightness=250.0, contrast=50.0, sharpness=150.0)
    result = analyzer.analyze(_crop(200, 200), _bbox(200, 200), 400, 400, feature_vector=fv)
    assert QualityWarning.HIGH_BRIGHTNESS in result.warnings


def test_low_sharpness_flagged():
    analyzer = StandardQualityAnalyzer(settings=_settings())
    fv = _feature_vector(brightness=128.0, contrast=50.0, sharpness=5.0)
    result = analyzer.analyze(_crop(200, 200), _bbox(200, 200), 400, 400, feature_vector=fv)
    assert QualityWarning.LOW_SHARPNESS in result.warnings


def test_low_contrast_flagged():
    analyzer = StandardQualityAnalyzer(settings=_settings())
    fv = _feature_vector(brightness=128.0, contrast=2.0, sharpness=150.0)
    result = analyzer.analyze(_crop(200, 200), _bbox(200, 200), 400, 400, feature_vector=fv)
    assert QualityWarning.LOW_CONTRAST in result.warnings


def test_small_face_flagged():
    analyzer = StandardQualityAnalyzer(settings=_settings())
    fv = _feature_vector(brightness=128.0, contrast=50.0, sharpness=150.0)
    # face area ratio = (20*20)/(400*400) = 0.0025 << 0.05 threshold
    result = analyzer.analyze(_crop(20, 20), _bbox(20, 20), 400, 400, feature_vector=fv)
    assert QualityWarning.FACE_TOO_SMALL in result.warnings


def test_advisory_mode_never_rejects_even_with_severe_warning():
    """Spec sec 21: advisory mode allows evaluation without discarding data."""
    analyzer = StandardQualityAnalyzer(settings=_settings(quality_mode="advisory"))
    fv = _feature_vector(brightness=128.0, contrast=50.0, sharpness=150.0)
    result = analyzer.analyze(_crop(5, 5), _bbox(5, 5), 400, 400, feature_vector=fv)
    assert QualityWarning.FACE_TOO_SMALL in result.warnings
    assert result.is_acceptable is True
    assert result.status == "warning"


def test_strict_mode_rejects_on_severe_warning():
    """Spec sec 43: strict mode + severe quality failure -> prediction unavailable."""
    analyzer = StandardQualityAnalyzer(settings=_settings(quality_mode="strict"))
    fv = _feature_vector(brightness=128.0, contrast=50.0, sharpness=150.0)
    result = analyzer.analyze(_crop(5, 5), _bbox(5, 5), 400, 400, feature_vector=fv)
    assert result.is_acceptable is False
    assert result.status == "reject"


def test_strict_mode_does_not_reject_on_advisory_only_warnings():
    """Low contrast alone must never reject, in either mode (spec sec 17/20)."""
    analyzer = StandardQualityAnalyzer(settings=_settings(quality_mode="strict"))
    fv = _feature_vector(brightness=128.0, contrast=2.0, sharpness=150.0)
    result = analyzer.analyze(_crop(200, 200), _bbox(200, 200), 400, 400, feature_vector=fv)
    assert QualityWarning.LOW_CONTRAST in result.warnings
    assert result.is_acceptable is True
    assert result.status == "warning"


def test_falls_back_to_computing_metrics_when_no_feature_vector_given():
    analyzer = StandardQualityAnalyzer(settings=_settings())
    crop = np.full((200, 200, 3), 128, dtype=np.uint8)
    result = analyzer.analyze(crop, _bbox(200, 200), 400, 400, feature_vector=None)
    assert "brightness" in result.metrics
    assert "sharpness" in result.metrics
    # A flat, uniform crop has zero contrast and zero Laplacian variance.
    assert QualityWarning.LOW_CONTRAST in result.warnings
    assert QualityWarning.LOW_SHARPNESS in result.warnings
