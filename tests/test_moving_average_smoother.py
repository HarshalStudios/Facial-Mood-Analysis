"""
Tests for MovingAverageSmoother (Phase 8 spec sec 6-7, 33, 38, 40).

Verifies the actual smoothed *probability vector*, not just the final
label (spec sec 38: "Do not merely test the final label").
"""

from __future__ import annotations

import pytest

from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.temporal.moving_average import MovingAverageSmoother


def _one_hot(label: str) -> dict[str, float]:
    return {l: (1.0 if l == label else 0.0) for l in CANONICAL_EMOTION_LABELS}


def test_window_size_must_be_positive():
    with pytest.raises(ValueError):
        MovingAverageSmoother(window_size=0)


def test_starts_empty():
    smoother = MovingAverageSmoother(window_size=5)
    assert smoother.history_length == 0


def test_no_startup_delay_single_frame_is_smoothed_over_itself():
    """Spec sec 9: frame 1 should not wait for a full window before producing output."""
    smoother = MovingAverageSmoother(window_size=5)
    result = smoother.update(_one_hot("happy"))
    assert smoother.history_length == 1
    assert result["happy"] == pytest.approx(1.0)
    assert result["sad"] == pytest.approx(0.0)


def test_deterministic_three_frame_average_matches_spec_example():
    """
    Spec sec 38 example:
        P1 = [1,0,0,0,0,0,0]  (angry)
        P2 = [0,1,0,0,0,0,0]  (disgust)
        P3 = [0,1,0,0,0,0,0]  (disgust)
    Expected average: angry=1/3, disgust=2/3, everything else 0.
    """
    smoother = MovingAverageSmoother(window_size=5)
    smoother.update(_one_hot("angry"))
    smoother.update(_one_hot("disgust"))
    result = smoother.update(_one_hot("disgust"))

    assert result["angry"] == pytest.approx(1 / 3)
    assert result["disgust"] == pytest.approx(2 / 3)
    for label in CANONICAL_EMOTION_LABELS:
        if label not in ("angry", "disgust"):
            assert result[label] == pytest.approx(0.0)
    assert smoother.history_length == 3


def test_window_is_bounded_oldest_entries_drop_out():
    smoother = MovingAverageSmoother(window_size=2)
    smoother.update(_one_hot("angry"))
    result = smoother.update(_one_hot("happy"))
    assert smoother.history_length == 2
    assert result["angry"] == pytest.approx(0.5)
    assert result["happy"] == pytest.approx(0.5)

    # A third update should push "angry" out of the bounded window entirely.
    result = smoother.update(_one_hot("sad"))
    assert smoother.history_length == 2
    assert result["angry"] == pytest.approx(0.0)
    assert result["happy"] == pytest.approx(0.5)
    assert result["sad"] == pytest.approx(0.5)


def test_reset_clears_all_history():
    """Spec sec 40: after reset(), history_length == 0 and no old probability information remains."""
    smoother = MovingAverageSmoother(window_size=5)
    smoother.update(_one_hot("happy"))
    smoother.update(_one_hot("happy"))
    assert smoother.history_length == 2

    smoother.reset()
    assert smoother.history_length == 0

    # The next update should behave exactly like a fresh instance -- no
    # leftover influence from before reset().
    result = smoother.update(_one_hot("sad"))
    assert result["sad"] == pytest.approx(1.0)
    assert result["happy"] == pytest.approx(0.0)


def test_raw_vs_smoothed_alternating_sequence():
    """
    Spec sec 45: an alternating raw sequence should be smoothed
    according to the actual moving-average math, not forced toward one
    particular emotion.
    """
    smoother = MovingAverageSmoother(window_size=5)
    sequence = ["happy", "neutral", "happy", "neutral", "happy"]
    last = None
    for label in sequence:
        last = smoother.update(_one_hot(label))

    # 3 "happy" + 2 "neutral" over 5 frames.
    assert last["happy"] == pytest.approx(3 / 5)
    assert last["neutral"] == pytest.approx(2 / 5)
    assert max(last, key=last.get) == "happy"
