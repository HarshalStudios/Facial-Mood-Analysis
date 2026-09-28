from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.camera.opencv_camera import OpenCVCameraSource
from app.core.config import Settings
from app.core.exceptions import CameraError


def _settings() -> Settings:
    return Settings(_env_file=None)


@patch("app.camera.opencv_camera.cv2.VideoCapture")
def test_open_raises_camera_error_when_capture_fails(mock_capture_cls):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = False
    mock_capture_cls.return_value = mock_capture

    camera = OpenCVCameraSource(settings=_settings())
    with pytest.raises(CameraError):
        camera.open()


@patch("app.camera.opencv_camera.cv2.VideoCapture")
def test_open_succeeds_and_configures_capture(mock_capture_cls):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture_cls.return_value = mock_capture

    camera = OpenCVCameraSource(settings=_settings())
    camera.open()

    assert mock_capture.set.called


@patch("app.camera.opencv_camera.cv2.VideoCapture")
def test_read_returns_frame_with_incrementing_ids(mock_capture_cls):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_capture.read.return_value = (True, fake_image)
    mock_capture_cls.return_value = mock_capture

    camera = OpenCVCameraSource(settings=_settings())
    camera.open()

    frame0 = camera.read()
    frame1 = camera.read()

    assert frame0.frame_id == 0
    assert frame1.frame_id == 1
    assert frame0.resolution == (640, 480)


@patch("app.camera.opencv_camera.cv2.VideoCapture")
def test_read_returns_none_on_failed_frame(mock_capture_cls):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.return_value = (False, None)
    mock_capture_cls.return_value = mock_capture

    camera = OpenCVCameraSource(settings=_settings())
    camera.open()

    assert camera.read() is None


def test_read_before_open_raises_camera_error():
    camera = OpenCVCameraSource(settings=_settings())
    with pytest.raises(CameraError):
        camera.read()


@patch("app.camera.opencv_camera.cv2.VideoCapture")
def test_close_releases_capture(mock_capture_cls):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture_cls.return_value = mock_capture

    camera = OpenCVCameraSource(settings=_settings())
    camera.open()
    camera.close()

    mock_capture.release.assert_called_once()


@patch("app.camera.opencv_camera.cv2.VideoCapture")
def test_context_manager_opens_and_closes(mock_capture_cls):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture_cls.return_value = mock_capture

    with OpenCVCameraSource(settings=_settings()) as camera:
        assert camera._capture is not None

    mock_capture.release.assert_called_once()
