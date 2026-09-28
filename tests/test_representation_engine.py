"""
Unit tests for RepresentationEngine (Phase 3).

Uses a small deterministic synthetic image, not a webcam or a real
face — these tests only validate that each image-processing operation
runs, preserves dimensions, and is deterministic. They make no claim
about emotion-recognition accuracy (see scripts/test_representations.py
for the manual, real-webcam integration check).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import RepresentationError
from app.representations.engine import RepresentationEngine


def make_test_image(size: int = 96) -> np.ndarray:
    """
    Deterministic synthetic BGR uint8 image: a horizontal gradient with
    a couple of simple shapes drawn on it, so edge/gradient/texture
    representations all have something non-trivial to compute on.
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)

    gradient = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    for c in range(3):
        image[:, :, c] = gradient

    # A filled rectangle and a filled circle give Canny/Sobel real edges.
    image[20:40, 20:60, :] = 220
    yy, xx = np.ogrid[:size, :size]
    circle_mask = (xx - 70) ** 2 + (yy - 70) ** 2 <= 15**2
    image[circle_mask] = (30, 30, 30)

    return image


@pytest.fixture
def settings() -> Settings:
    # Explicit Settings instance (not get_settings()) so tests don't
    # depend on a real .env file and stay isolated from each other.
    return Settings(_env_file=None)


@pytest.fixture
def engine(settings: Settings) -> RepresentationEngine:
    return RepresentationEngine(settings=settings)


@pytest.fixture
def face_crop() -> np.ndarray:
    return make_test_image()


# ---------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------


def test_grayscale_dimensions_and_dtype(engine, face_crop):
    result = engine.generate(face_crop)
    h, w = face_crop.shape[:2]
    assert result.grayscale.shape == (h, w)
    assert result.grayscale.dtype == np.uint8


def test_grayscale_is_deterministic(engine, face_crop):
    first = engine.generate(face_crop).grayscale
    second = engine.generate(face_crop).grayscale
    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------


def test_clahe_output_exists_and_preserves_dimensions(engine, face_crop):
    result = engine.generate(face_crop)
    assert result.clahe is not None
    assert result.clahe.shape == result.grayscale.shape
    assert result.clahe.dtype == np.uint8


def test_clahe_output_is_valid_intensity_image(engine, face_crop):
    result = engine.generate(face_crop)
    assert result.clahe.min() >= 0
    assert result.clahe.max() <= 255


# ---------------------------------------------------------------------
# Canny
# ---------------------------------------------------------------------


def test_canny_preserves_dimensions(engine, face_crop):
    result = engine.generate(face_crop)
    assert result.canny.shape == result.grayscale.shape


def test_canny_is_binary_edge_map(engine, face_crop):
    result = engine.generate(face_crop)
    unique_values = set(np.unique(result.canny).tolist())
    assert unique_values <= {0, 255}


def test_canny_thresholds_are_configurable(settings, face_crop):
    settings.canny_low_threshold = 10
    settings.canny_high_threshold = 30
    permissive = RepresentationEngine(settings=settings).generate(face_crop).canny

    settings.canny_low_threshold = 200
    settings.canny_high_threshold = 250
    strict = RepresentationEngine(settings=settings).generate(face_crop).canny

    # Lower thresholds should never detect fewer edge pixels than
    # much stricter ones on the same image.
    assert int(np.count_nonzero(permissive)) >= int(np.count_nonzero(strict))


# ---------------------------------------------------------------------
# Sobel
# ---------------------------------------------------------------------


def test_sobel_preserves_dimensions(engine, face_crop):
    result = engine.generate(face_crop)
    assert result.sobel.shape == result.grayscale.shape
    assert result.sobel.dtype == np.uint8


def test_sobel_produces_nonzero_gradient_on_edges(engine, face_crop):
    result = engine.generate(face_crop)
    # The synthetic image has a hard-edged rectangle; Sobel magnitude
    # should be non-trivial somewhere.
    assert result.sobel.max() > 0


def test_sobel_configuration_is_applied(settings, face_crop):
    settings.sobel_kernel_size = 3
    k3 = RepresentationEngine(settings=settings).generate(face_crop).sobel

    settings.sobel_kernel_size = 5
    k5 = RepresentationEngine(settings=settings).generate(face_crop).sobel

    # Different kernel sizes should not coincidentally produce an
    # identical gradient image.
    assert not np.array_equal(k3, k5)


# ---------------------------------------------------------------------
# LBP
# ---------------------------------------------------------------------


def test_lbp_preserves_dimensions(engine, face_crop):
    result = engine.generate(face_crop)
    assert result.lbp.shape == result.grayscale.shape


def test_lbp_is_deterministic(engine, face_crop):
    first = engine.generate(face_crop).lbp
    second = engine.generate(face_crop).lbp
    np.testing.assert_array_equal(first, second)


def test_lbp_respects_configured_parameters(settings, face_crop):
    settings.lbp_n_points = 8
    settings.lbp_radius = 1
    p8 = RepresentationEngine(settings=settings).generate(face_crop).lbp

    settings.lbp_n_points = 16
    settings.lbp_radius = 2
    p16 = RepresentationEngine(settings=settings).generate(face_crop).lbp

    assert not np.array_equal(p8, p16)


# ---------------------------------------------------------------------
# Full engine / RepresentationSet
# ---------------------------------------------------------------------


def test_full_engine_produces_all_six_representations(engine, face_crop):
    result = engine.generate(face_crop)
    assert result.rgb is not None
    assert result.grayscale is not None
    assert result.clahe is not None
    assert result.canny is not None
    assert result.sobel is not None
    assert result.lbp is not None


def test_all_representations_share_face_crop_dimensions(engine, face_crop):
    result = engine.generate(face_crop)
    h, w = face_crop.shape[:2]
    for name in ("grayscale", "clahe", "canny", "sobel", "lbp"):
        arr = getattr(result, name)
        assert arr.shape[:2] == (h, w), f"{name} changed dimensions"


def test_rgb_representation_is_unmodified_copy(engine, face_crop):
    result = engine.generate(face_crop)
    np.testing.assert_array_equal(result.rgb, face_crop)
    # Must be a copy, not the same underlying buffer.
    assert result.rgb is not face_crop


def test_metadata_recorded_for_every_enabled_representation(engine, face_crop):
    result = engine.generate(face_crop)
    assert set(result.enabled_names()) == {"rgb", "grayscale", "clahe", "canny", "sobel", "lbp"}
    for name, meta in result.metadata.items():
        assert meta.name == name
        assert meta.width == face_crop.shape[1]
        assert meta.height == face_crop.shape[0]


def test_disabling_a_representation_omits_it(settings, face_crop):
    settings.enable_lbp = False
    settings.enable_sobel = False
    result = RepresentationEngine(settings=settings).generate(face_crop)

    assert result.lbp is None
    assert result.sobel is None
    assert "lbp" not in result.enabled_names()
    assert "sobel" not in result.enabled_names()
    # Others remain unaffected.
    assert result.grayscale is not None
    assert result.clahe is not None
    assert result.canny is not None


def test_disabling_all_derived_representations_skips_grayscale_conversion(settings, face_crop):
    for flag in ("enable_grayscale", "enable_clahe", "enable_canny", "enable_sobel", "enable_lbp"):
        setattr(settings, flag, False)
    result = RepresentationEngine(settings=settings).generate(face_crop)

    assert result.grayscale is None
    assert result.enabled_names() == ["rgb"]


# ---------------------------------------------------------------------
# Invalid input handling
# ---------------------------------------------------------------------


def test_rejects_none_input(engine):
    with pytest.raises(RepresentationError):
        engine.generate(None)


def test_rejects_empty_array(engine):
    with pytest.raises(RepresentationError):
        engine.generate(np.zeros((0, 0, 3), dtype=np.uint8))


def test_rejects_wrong_channel_count(engine):
    with pytest.raises(RepresentationError):
        engine.generate(np.zeros((50, 50, 4), dtype=np.uint8))


def test_rejects_grayscale_shaped_input(engine):
    with pytest.raises(RepresentationError):
        engine.generate(np.zeros((50, 50), dtype=np.uint8))


def test_rejects_wrong_dtype(engine):
    with pytest.raises(RepresentationError):
        engine.generate(np.zeros((50, 50, 3), dtype=np.float32))


def test_rejects_non_ndarray_input(engine):
    with pytest.raises(RepresentationError):
        engine.generate([[1, 2, 3]])
