import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import FeatureExtractionError
from app.features.lbp_histogram import compute_lbp_histogram


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def test_histogram_has_expected_bin_count(settings):
    lbp = np.random.default_rng(0).integers(0, 10, size=(50, 50)).astype(np.uint8)
    hist = compute_lbp_histogram(lbp, settings)
    assert hist.shape == (10,)


def test_histogram_is_normalized(settings):
    lbp = np.random.default_rng(0).integers(0, 10, size=(50, 50)).astype(np.uint8)
    hist = compute_lbp_histogram(lbp, settings)
    assert hist.sum() == pytest.approx(1.0)
    assert (hist >= 0).all()


def test_histogram_is_deterministic(settings):
    lbp = np.random.default_rng(0).integers(0, 10, size=(50, 50)).astype(np.uint8)
    first = compute_lbp_histogram(lbp, settings)
    second = compute_lbp_histogram(lbp, settings)
    np.testing.assert_array_equal(first, second)


def test_uniform_codes_map_one_to_one_into_bins(settings):
    # One pixel per code 0..9, otherwise empty -> each of the 10 bins gets exactly one count.
    lbp = np.arange(10, dtype=np.uint8).reshape(1, 10)
    hist = compute_lbp_histogram(lbp, settings)
    np.testing.assert_allclose(hist, np.full(10, 0.1))


def test_none_lbp_raises(settings):
    with pytest.raises(FeatureExtractionError):
        compute_lbp_histogram(None, settings)


def test_non_uniform_method_raises(settings):
    settings.lbp_method = "default"
    lbp = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(FeatureExtractionError):
        compute_lbp_histogram(lbp, settings)


def test_mismatched_num_bins_raises_instead_of_truncating(settings):
    settings.lbp_n_points = 16  # expects 18 bins, but config still says 10
    lbp = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(FeatureExtractionError):
        compute_lbp_histogram(lbp, settings)
