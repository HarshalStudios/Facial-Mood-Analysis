"""Tests for EMASmoother (Phase 8 spec sec 8, 39)."""

from __future__ import annotations

import pytest

from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.temporal.ema import EMASmoother


def _one_hot(label: str) -> dict[str, float]:
    return {l: (1.0 if l == label else 0.0) for l in CANONICAL_EMOTION_LABELS}


def test_alpha_must_be_in_valid_range():
    with pytest.raises(ValueError):
        EMASmoother(alpha=0.0)
    with pytest.raises(ValueError):
        EMASmoother(alpha=1.5)


def test_first_observation_seeds_state_directly():
    smoother = EMASmoother(alpha=0.3)
    result = smoother.update(_one_hot("happy"))
    assert result["happy"] == pytest.approx(1.0)
    assert smoother.history_length == 1


def test_deterministic_ema_update_matches_formula():
    """
    Spec sec 39: verify S_t = alpha * P_t + (1 - alpha) * S_(t-1) with a
    deterministic example, alpha = 0.5.

    S_1 = P_1 = happy:1.0
    S_2 = 0.5 * P_2(sad:1.0) + 0.5 * S_1(happy:1.0) -> happy:0.5, sad:0.5
    """
    smoother = EMASmoother(alpha=0.5)
    smoother.update(_one_hot("happy"))
    result = smoother.update(_one_hot("sad"))

    assert result["happy"] == pytest.approx(0.5)
    assert result["sad"] == pytest.approx(0.5)
    for label in CANONICAL_EMOTION_LABELS:
        if label not in ("happy", "sad"):
            assert result[label] == pytest.approx(0.0)
    assert smoother.history_length == 2


def test_reset_clears_state():
    smoother = EMASmoother(alpha=0.5)
    smoother.update(_one_hot("happy"))
    smoother.reset()
    assert smoother.history_length == 0

    result = smoother.update(_one_hot("sad"))
    assert result["sad"] == pytest.approx(1.0)  # behaves like a fresh instance
