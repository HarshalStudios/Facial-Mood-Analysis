import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import FeatureExtractionError
from app.features.image_quality import (
    compute_brightness,
    compute_contrast,
    compute_gradient_energy,
    compute_sharpness,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


# ---------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------


def test_black_image_has_zero_brightness():
    assert compute_brightness(np.zeros((20, 20), dtype=np.uint8)) == 0.0


def test_white_image_has_max_brightness():
    assert compute_brightness(np.full((20, 20), 255, dtype=np.uint8)) == pytest.approx(255.0)


def test_brightness_none_raises():
    with pytest.raises(FeatureExtractionError):
        compute_brightness(None)


# ---------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------


def test_constant_image_has_zero_contrast():
    assert compute_contrast(np.full((20, 20), 128, dtype=np.uint8)) == 0.0


def test_varying_image_has_positive_contrast():
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[:10, :] = 255
    assert compute_contrast(gray) > 0.0


# ---------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------


def test_constant_image_has_zero_sharpness():
    assert compute_sharpness(np.full((20, 20), 128, dtype=np.uint8)) == pytest.approx(0.0, abs=1e-9)


def test_structured_edges_have_higher_sharpness_than_blank():
    blank = np.full((40, 40), 128, dtype=np.uint8)
    checker = np.zeros((40, 40), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    assert compute_sharpness(checker) > compute_sharpness(blank)


# ---------------------------------------------------------------------
# Gradient energy
# ---------------------------------------------------------------------


def test_constant_image_has_zero_gradient_energy(settings):
    gray = np.full((20, 20), 128, dtype=np.uint8)
    assert compute_gradient_energy(gray, settings) == pytest.approx(0.0, abs=1e-9)


def test_edge_image_has_positive_gradient_energy(settings):
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[:, 10:] = 255
    assert compute_gradient_energy(gray, settings) > 0.0


def test_gradient_energy_is_deterministic(settings):
    gray = np.random.default_rng(1).integers(0, 255, size=(30, 30), dtype=np.uint8)
    first = compute_gradient_energy(gray, settings)
    second = compute_gradient_energy(gray, settings)
    assert first == second


def test_gradient_energy_scales_with_kernel_size(settings):
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[:, 10:] = 255

    settings.sobel_kernel_size = 3
    energy_k3 = compute_gradient_energy(gray, settings)

    settings.sobel_kernel_size = 5
    energy_k5 = compute_gradient_energy(gray, settings)

    assert energy_k3 != energy_k5


def test_gradient_energy_none_raises(settings):
    with pytest.raises(FeatureExtractionError):
        compute_gradient_energy(None, settings)
