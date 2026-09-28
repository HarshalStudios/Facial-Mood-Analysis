"""
Build a `TemporalSmoother` from configuration (Phase 8 spec sec 31, 50).

`Settings.temporal_enabled=False` always wins and yields a passthrough
`NoOpSmoother`, regardless of `smoothing_method` -- this is what lets
Phase 9 ablation flip smoothing fully off without touching any other
config value (spec sec 51).
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.temporal.base import TemporalSmoother
from app.temporal.ema import EMASmoother
from app.temporal.moving_average import MovingAverageSmoother
from app.temporal.no_smoothing import NoOpSmoother

VALID_SMOOTHING_METHODS = frozenset({"moving_average", "ema", "none"})


def get_temporal_smoother(settings: Settings) -> TemporalSmoother:
    if not settings.temporal_enabled:
        return NoOpSmoother()

    method = settings.smoothing_method
    if method == "moving_average":
        return MovingAverageSmoother(window_size=settings.smoothing_window_size)
    if method == "ema":
        return EMASmoother(alpha=settings.ema_alpha)
    if method == "none":
        return NoOpSmoother()

    raise ConfigurationError(
        f"Unknown smoothing_method '{method}', expected one of {sorted(VALID_SMOOTHING_METHODS)}"
    )
