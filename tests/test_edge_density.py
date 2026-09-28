import numpy as np
import pytest

from app.core.exceptions import FeatureExtractionError
from app.features.edge_density import compute_edge_density


def test_blank_canny_gives_zero_density():
    canny = np.zeros((50, 50), dtype=np.uint8)
    assert compute_edge_density(canny) == 0.0


def test_all_edges_gives_density_one():
    canny = np.full((50, 50), 255, dtype=np.uint8)
    assert compute_edge_density(canny) == 1.0


def test_half_edges_gives_density_half():
    canny = np.zeros((10, 10), dtype=np.uint8)
    canny[:5, :] = 255
    assert compute_edge_density(canny) == pytest.approx(0.5)


def test_none_raises():
    with pytest.raises(FeatureExtractionError):
        compute_edge_density(None)


def test_empty_raises():
    with pytest.raises(FeatureExtractionError):
        compute_edge_density(np.zeros((0, 0), dtype=np.uint8))
