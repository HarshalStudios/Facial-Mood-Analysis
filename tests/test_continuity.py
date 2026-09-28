"""Tests for the face-continuity IoU helper (Phase 8 spec sec 12, 46)."""

from __future__ import annotations

import pytest

from app.schemas.face import BoundingBox
from app.temporal.continuity import bbox_iou, is_same_subject


def test_identical_boxes_have_iou_one():
    box = BoundingBox(x=10, y=10, width=50, height=50)
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero():
    a = BoundingBox(x=0, y=0, width=50, height=50)
    b = BoundingBox(x=500, y=500, width=50, height=50)
    assert bbox_iou(a, b) == pytest.approx(0.0)


def test_partial_overlap_is_between_zero_and_one():
    a = BoundingBox(x=0, y=0, width=100, height=100)
    b = BoundingBox(x=50, y=50, width=100, height=100)
    iou = bbox_iou(a, b)
    assert 0.0 < iou < 1.0
    # Known value: intersection = 50x50 = 2500, union = 10000+10000-2500 = 17500
    assert iou == pytest.approx(2500 / 17500)


def test_no_previous_face_is_always_same_subject():
    """Spec sec 12: nothing to compare against yet -- must not force an immediate reset."""
    current = BoundingBox(x=0, y=0, width=50, height=50)
    assert is_same_subject(None, current, iou_threshold=0.5) is True


def test_large_jump_is_flagged_as_different_subject():
    """
    Spec sec 46 (multi-face continuity test): Face A stays put across
    frames, then Face B suddenly becomes primary at a distant location.
    """
    face_a_frame1 = BoundingBox(x=10, y=10, width=80, height=80)
    face_a_frame2 = BoundingBox(x=15, y=12, width=80, height=80)
    face_b = BoundingBox(x=400, y=400, width=80, height=80)

    assert is_same_subject(face_a_frame1, face_a_frame2, iou_threshold=0.3) is True
    assert is_same_subject(face_a_frame2, face_b, iou_threshold=0.3) is False


def test_threshold_boundary_is_inclusive():
    a = BoundingBox(x=0, y=0, width=100, height=100)
    b = BoundingBox(x=0, y=0, width=100, height=100)
    iou = bbox_iou(a, b)
    assert is_same_subject(a, b, iou_threshold=iou) is True
