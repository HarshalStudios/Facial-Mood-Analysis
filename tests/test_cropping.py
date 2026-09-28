import numpy as np
import pytest

from app.detection.cropping import crop_face
from app.schemas.face import BoundingBox


def test_crop_face_returns_expected_region():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:150, 100:150] = 255  # white square

    bbox = BoundingBox(x=100, y=100, width=50, height=50)
    crop = crop_face(image, bbox)

    assert crop.shape == (50, 50, 3)
    assert (crop == 255).all()


def test_crop_face_raises_on_empty_result():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bad_bbox = BoundingBox(x=640, y=480, width=0, height=0)
    with pytest.raises(ValueError):
        crop_face(image, bad_bbox)
