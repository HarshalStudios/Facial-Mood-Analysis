"""
EMASmoother (Phase 8 spec sec 8) -- optional exponential moving average.

    S_t = alpha * P_t + (1 - alpha) * S_(t-1)

`alpha` is a configurable starting value (`Settings.ema_alpha`), never
claimed to be optimal (spec sec 8/31/32) -- Phase 9's ablation engine
is where alternative values get evaluated against held-out data.
"""

from __future__ import annotations

from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.temporal.base import TemporalSmoother


class EMASmoother(TemporalSmoother):
    def __init__(self, alpha: float = 0.3) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._state: dict[str, float] | None = None
        self._count = 0

    def update(self, probabilities: dict[str, float]) -> dict[str, float]:
        if self._state is None:
            # First observation: nothing to blend with yet, so the
            # smoothed state starts exactly at the raw vector.
            self._state = {label: probabilities.get(label, 0.0) for label in CANONICAL_EMOTION_LABELS}
        else:
            self._state = {
                label: self._alpha * probabilities.get(label, 0.0) + (1.0 - self._alpha) * self._state[label]
                for label in CANONICAL_EMOTION_LABELS
            }
        self._count += 1
        return dict(self._state)

    def reset(self) -> None:
        self._state = None
        self._count = 0

    @property
    def history_length(self) -> int:
        return self._count

    @property
    def alpha(self) -> float:
        return self._alpha
