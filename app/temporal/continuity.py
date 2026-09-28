"""
Primary-face continuity check (Phase 8 spec sec 12).

Deliberately NOT a face-tracking or face-recognition system -- just a
bounding-box IoU test used to decide whether the temporal history
should reset because the primary face jumped to a plausibly different
subject between frames.
"""

from __future__ import annotations

from app.schemas.face import BoundingBox


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two bounding boxes, in [0, 1]."""
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height

    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0

    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def is_same_subject(previous: BoundingBox | None, current: BoundingBox, iou_threshold: float) -> bool:
    """
    True if `current` is plausibly the same subject as `previous`.

    With no previous face on record (first frame, or history was just
    reset), there is nothing to compare against, so this returns True
    rather than forcing an immediate reset.
    """
    if previous is None:
        return True
    return bbox_iou(previous, current) >= iou_threshold
