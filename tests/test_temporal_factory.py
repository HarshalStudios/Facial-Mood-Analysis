"""
Tests for `get_temporal_smoother()` (Phase 8 spec sec 31, 50-51).

Uses a duck-typed `SimpleNamespace` stand-in for `Settings` (same
pattern as `tests/test_predictor.py`) since the factory only reads
plain attributes off whatever settings object it is given.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.exceptions import ConfigurationError
from app.temporal.ema import EMASmoother
from app.temporal.factory import get_temporal_smoother
from app.temporal.moving_average import MovingAverageSmoother
from app.temporal.no_smoothing import NoOpSmoother


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        temporal_enabled=True,
        smoothing_method="moving_average",
        smoothing_window_size=5,
        ema_alpha=0.3,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_moving_average_is_default():
    smoother = get_temporal_smoother(_settings())
    assert isinstance(smoother, MovingAverageSmoother)
    assert smoother.window_size == 5


def test_ema_method_selected():
    smoother = get_temporal_smoother(_settings(smoothing_method="ema", ema_alpha=0.4))
    assert isinstance(smoother, EMASmoother)
    assert smoother.alpha == pytest.approx(0.4)


def test_none_method_yields_passthrough():
    smoother = get_temporal_smoother(_settings(smoothing_method="none"))
    assert isinstance(smoother, NoOpSmoother)


def test_temporal_disabled_always_yields_passthrough_regardless_of_method():
    """Spec sec 51: temporal_enabled=False must always win, whatever smoothing_method says."""
    smoother = get_temporal_smoother(_settings(temporal_enabled=False, smoothing_method="ema"))
    assert isinstance(smoother, NoOpSmoother)


def test_unknown_method_raises_configuration_error():
    with pytest.raises(ConfigurationError):
        get_temporal_smoother(_settings(smoothing_method="not_a_real_method"))
