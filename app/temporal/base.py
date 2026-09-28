"""
TemporalSmoother interface (Phase 8 spec sec 5-9).

Smooths a stream of 7-class *probability* dicts into a stable
probability dict (spec sec 7: "smooth probabilities, not labels" --
the stable label/confidence are derived via argmax only after
averaging, by the caller). Implementations keep a bounded internal
history (spec sec 5) and must be independently resettable (spec sec
33). Single-threaded by design (spec sec 34) -- construct one instance
per camera/session, never share across threads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TemporalSmoother(ABC):
    """Abstract base class for smoothing a sequence of probability vectors."""

    @abstractmethod
    def update(self, probabilities: dict[str, float]) -> dict[str, float]:
        """
        Feed in the latest *valid* raw probability vector (only valid
        predictions may ever reach this call -- spec sec 22-23) and
        return the current smoothed probability vector.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal history (e.g. on face lost / new session)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def history_length(self) -> int:
        """Number of valid predictions currently held in the window (spec sec 27)."""
        raise NotImplementedError
