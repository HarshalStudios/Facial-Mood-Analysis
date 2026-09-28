import numpy as np

from app.detection.selector import (
    CenterMostFaceSelector,
    HighestConfidenceFaceSelector,
    LargestFaceSelector,
    get_face_selector,
)
from app.schemas.face import BoundingBox, Face
from app.schemas.frame import Frame


def _face(x, y, w, h, confidence=None) -> Face:
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    return Face(bounding_box=BoundingBox(x, y, w, h), crop=crop, confidence=confidence)


def test_largest_selector_picks_biggest_area():
    small = _face(0, 0, 50, 50)
    big = _face(100, 100, 200, 200)
    selected = LargestFaceSelector().select([small, big])
    assert selected is big


def test_largest_selector_empty_list_returns_none():
    assert LargestFaceSelector().select([]) is None


def test_highest_confidence_selector_picks_highest_score():
    low = _face(0, 0, 50, 50, confidence=0.4)
    high = _face(0, 0, 50, 50, confidence=0.9)
    selected = HighestConfidenceFaceSelector().select([low, high])
    assert selected is high


def test_highest_confidence_selector_falls_back_to_largest_when_no_confidence():
    small = _face(0, 0, 50, 50, confidence=None)
    big = _face(0, 0, 200, 200, confidence=None)
    selected = HighestConfidenceFaceSelector().select([small, big])
    assert selected is big


def test_center_most_selector_picks_closest_to_frame_center():
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=0)
    off_center = _face(0, 0, 50, 50)
    centered = _face(295, 215, 50, 50)  # center ~ (320, 240)
    selected = CenterMostFaceSelector().select([off_center, centered], frame)
    assert selected is centered


def test_get_face_selector_returns_configured_strategy():
    assert isinstance(get_face_selector("largest"), LargestFaceSelector)
    assert isinstance(get_face_selector("highest_confidence"), HighestConfidenceFaceSelector)
    assert isinstance(get_face_selector("center_most"), CenterMostFaceSelector)


def test_get_face_selector_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        get_face_selector("not_a_real_strategy")
