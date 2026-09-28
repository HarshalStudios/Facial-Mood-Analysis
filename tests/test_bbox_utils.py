from app.detection.bbox_utils import apply_padding, clamp_bbox, is_face_large_enough
from app.schemas.face import BoundingBox


def test_clamp_negative_coordinates():
    bbox = BoundingBox(x=-5, y=20, width=200, height=180)
    clamped = clamp_bbox(bbox, frame_width=640, frame_height=480)
    assert clamped is not None
    assert clamped.x == 0
    assert clamped.y == 20


def test_clamp_box_exceeding_frame_bounds():
    bbox = BoundingBox(x=600, y=450, width=200, height=200)
    clamped = clamp_bbox(bbox, frame_width=640, frame_height=480)
    assert clamped is not None
    assert clamped.x + clamped.width <= 640
    assert clamped.y + clamped.height <= 480


def test_clamp_rejects_box_entirely_outside_frame():
    bbox = BoundingBox(x=700, y=500, width=50, height=50)
    clamped = clamp_bbox(bbox, frame_width=640, frame_height=480)
    assert clamped is None


def test_clamp_rejects_zero_width_or_height():
    assert clamp_bbox(BoundingBox(0, 0, 0, 100), 640, 480) is None
    assert clamp_bbox(BoundingBox(0, 0, 100, 0), 640, 480) is None


def test_is_face_large_enough():
    assert is_face_large_enough(BoundingBox(0, 0, 100, 100), min_width=40, min_height=40)
    assert not is_face_large_enough(BoundingBox(0, 0, 10, 10), min_width=40, min_height=40)


def test_padding_expands_box_and_reclamps():
    bbox = BoundingBox(x=100, y=100, width=100, height=100)
    padded = apply_padding(bbox, padding_ratio=0.10, frame_width=640, frame_height=480)
    assert padded.width == 120
    assert padded.height == 120
    assert padded.x == 90
    assert padded.y == 90


def test_padding_noop_when_ratio_is_zero():
    bbox = BoundingBox(x=100, y=100, width=100, height=100)
    padded = apply_padding(bbox, padding_ratio=0.0, frame_width=640, frame_height=480)
    assert padded == bbox


def test_padding_clamps_at_frame_edge():
    bbox = BoundingBox(x=0, y=0, width=100, height=100)
    padded = apply_padding(bbox, padding_ratio=0.5, frame_width=640, frame_height=480)
    assert padded.x == 0
    assert padded.y == 0
