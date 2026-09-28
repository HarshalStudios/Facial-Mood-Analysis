"""Tests for NoOpSmoother (Phase 8 spec sec 50/51: 'Mode A: No smoothing')."""

from __future__ import annotations

import pytest

from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.temporal.no_smoothing import NoOpSmoother


def _one_hot(label: str) -> dict[str, float]:
    return {l: (1.0 if l == label else 0.0) for l in CANONICAL_EMOTION_LABELS}


def test_passthrough_returns_input_unchanged():
    smoother = NoOpSmoother()
    probabilities = {"happy": 0.7, "neutral": 0.2, "sad": 0.1}
    result = smoother.update(probabilities)
    assert result == probabilities
    assert result is not probabilities  # defensive copy, not aliasing caller state


def test_history_length_counts_updates_not_a_bounded_window():
    smoother = NoOpSmoother()
    smoother.update(_one_hot("happy"))
    smoother.update(_one_hot("sad"))
    assert smoother.history_length == 2


def test_reset_clears_state():
    smoother = NoOpSmoother()
    smoother.update(_one_hot("happy"))
    smoother.reset()
    assert smoother.history_length == 0
