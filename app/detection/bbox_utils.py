"""
Bounding-box normalization.

Raw detector output can have negative coordinates, boxes that spill
outside the frame, or degenerate (zero/negative) width and height. This
module clamps a raw box to valid frame coordinates and rejects boxes
that are still degenerate or too small after clamping, so nothing
downstream (cropping, feature extraction) ever sees an invalid box.
"""

from __future__ import annotations

from app.schemas.face import BoundingBox


def clamp_bbox(
    bbox: BoundingBox,
    frame_width: int,
    frame_height: int,
) -> BoundingBox | None:
    """
    Clamp a bounding box to lie fully within [0, frame_width) x [0, frame_height).

    Returns None if the box is degenerate even after clamping (e.g. it
    falls entirely outside the frame, or width/height collapses to <= 0).
    """
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(frame_width, bbox.x + bbox.width)
    y2 = min(frame_height, bbox.y + bbox.height)

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return None

    return BoundingBox(x=x1, y=y1, width=width, height=height)


def is_face_large_enough(
    bbox: BoundingBox,
    min_width: int,
    min_height: int,
) -> bool:
    """Return True if a (already-clamped) bbox meets the minimum face size."""
    return bbox.width >= min_width and bbox.height >= min_height


def apply_padding(
    bbox: BoundingBox,
    padding_ratio: float,
    frame_width: int,
    frame_height: int,
) -> BoundingBox:
    """
    Expand a bbox by `padding_ratio` on each side (e.g. 0.10 = +10% per
    side), then re-clamp to the frame. padding_ratio of 0 is a no-op.
    """
    if padding_ratio <= 0:
        return bbox

    pad_x = int(bbox.width * padding_ratio)
    pad_y = int(bbox.height * padding_ratio)

    expanded = BoundingBox(
        x=bbox.x - pad_x,
        y=bbox.y - pad_y,
        width=bbox.width + 2 * pad_x,
        height=bbox.height + 2 * pad_y,
    )

    clamped = clamp_bbox(expanded, frame_width, frame_height)
    # Padding expands from an already-valid box, so this should never be
    # None, but fall back to the original box defensively.
    return clamped if clamped is not None else bbox
