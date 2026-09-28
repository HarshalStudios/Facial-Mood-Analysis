import numpy as np

from app.core.config import Settings
from app.detection.opencv_detector import HaarCascadeFaceDetector
from app.schemas.frame import Frame


def test_haar_detector_loads_cascade_successfully():
    detector = HaarCascadeFaceDetector(settings=Settings(_env_file=None))
    assert detector._cascade.empty() is False


def test_haar_detector_returns_empty_list_on_blank_frame():
    detector = HaarCascadeFaceDetector(settings=Settings(_env_file=None))
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=0)

    faces = detector.detect(frame)

    assert faces == []


def test_haar_detector_reports_confidence_as_none_by_design():
    # Documented behavior: Haar cascades don't provide calibrated confidence,
    # so this backend must never invent one.
    detector = HaarCascadeFaceDetector(settings=Settings(_env_file=None))
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), frame_id=0)
    faces = detector.detect(frame)
    assert all(f.confidence is None for f in faces)
