"""
NoOpSmoother (Phase 8 spec sec 50/51: "Mode A: No smoothing").

A passthrough `TemporalSmoother` used when `temporal_enabled=False` or
`smoothing_method="none"`, so Phase 9's ablation engine can compare
raw Phase 7 output against smoothed output through the exact same
`TemporalPredictionEngine` code path -- no separate branch needed
elsewhere for "smoothing disabled".
"""

from __future__ import annotations

from app.temporal.base import TemporalSmoother


class NoOpSmoother(TemporalSmoother):
    def __init__(self) -> None:
        self._last: dict[str, float] | None = None
        self._count = 0

    def update(self, probabilities: dict[str, float]) -> dict[str, float]:
        self._last = dict(probabilities)
        self._count += 1
        return dict(self._last)

    def reset(self) -> None:
        self._last = None
        self._count = 0

    @property
    def history_length(self) -> int:
        return self._count
