import numpy as np

from app.core.config import Settings
from app.detection.base import FaceDetector
from app.detection.service import FaceDetectionService
from app.schemas.face import BoundingBox, Face
from app.schemas.frame import Frame


class _FakeDetector(FaceDetector):
    """Returns a fixed list of raw (un-clamped) detections for testing."""

    def __init__(self, raw_faces: list[Face]) -> None:
        self._raw_faces = raw_faces

    def detect(self, frame: Frame) -> list[Face]:
        return self._raw_faces


def _frame() -> Frame:
    return Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=0)


def _raw_face(x, y, w, h) -> Face:
    return Face(bounding_box=BoundingBox(x, y, w, h), crop=np.zeros((1, 1, 3), dtype=np.uint8), confidence=None)


def test_service_marks_exactly_one_primary_face():
    settings = Settings(_env_file=None, min_face_width=10, min_face_height=10, face_padding_ratio=0.0)
    detector = _FakeDetector([_raw_face(10, 10, 50, 50), _raw_face(300, 300, 150, 150)])
    service = FaceDetectionService(detector, settings=settings)

    result = service.process(_frame())

    assert len(result.faces) == 2
    primaries = [f for f in result.faces if f.is_primary]
    assert len(primaries) == 1
    assert primaries[0].bounding_box.width == 150  # largest wins by default


def test_service_rejects_bbox_outside_frame():
    settings = Settings(_env_file=None, min_face_width=10, min_face_height=10)
    detector = _FakeDetector([_raw_face(700, 500, 50, 50)])  # fully outside 640x480
    service = FaceDetectionService(detector, settings=settings)

    result = service.process(_frame())

    assert result.faces == []
    assert result.rejected_count == 1


def test_service_rejects_faces_below_minimum_size():
    settings = Settings(_env_file=None, min_face_width=100, min_face_height=100)
    detector = _FakeDetector([_raw_face(10, 10, 20, 20)])
    service = FaceDetectionService(detector, settings=settings)

    result = service.process(_frame())

    assert result.faces == []
    assert result.rejected_count == 1


def test_service_no_faces_returns_empty_result_without_crashing():
    settings = Settings(_env_file=None)
    detector = _FakeDetector([])
    service = FaceDetectionService(detector, settings=settings)

    result = service.process(_frame())

    assert result.faces == []
    assert result.rejected_count == 0
