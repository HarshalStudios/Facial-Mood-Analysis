"""
MovingAverageSmoother (Phase 8 spec sec 6-7, 9) -- the primary/default
temporal smoothing method.

Averages the 7-class probability vector over a bounded, configurable
window. Uses a `deque(maxlen=window_size)` so there is no artificial
startup delay (spec sec 9): frame 1 is smoothed over just itself,
frame 2 over the last 2, and so on until the window fills.
"""

from __future__ import annotations

from collections import deque

from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.temporal.base import TemporalSmoother


class MovingAverageSmoother(TemporalSmoother):
    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._history: deque[dict[str, float]] = deque(maxlen=window_size)

    def update(self, probabilities: dict[str, float]) -> dict[str, float]:
        self._history.append(dict(probabilities))
        n = len(self._history)

        smoothed = {label: 0.0 for label in CANONICAL_EMOTION_LABELS}
        for entry in self._history:
            for label in CANONICAL_EMOTION_LABELS:
                smoothed[label] += entry.get(label, 0.0)
        for label in smoothed:
            smoothed[label] /= n

        return smoothed

    def reset(self) -> None:
        self._history.clear()

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def window_size(self) -> int:
        return self._window_size
