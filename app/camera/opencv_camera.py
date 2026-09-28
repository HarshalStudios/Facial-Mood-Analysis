"""
OpenCV-backed webcam camera source.

Color space contract: frames are stored internally as OpenCV's native
BGR, uint8, shape (H, W, 3) — matching app.schemas.frame.Frame's
documented contract. Any module that needs RGB (e.g. a future DeepFace
call, which expects RGB) must convert explicitly at its own boundary;
this class never silently converts color spaces.
"""

from __future__ import annotations

from datetime import datetime, timezone

import cv2

from app.camera.base import CameraSource
from app.core.config import Settings, get_settings
from app.core.exceptions import CameraError
from app.core.logging import get_logger
from app.schemas.frame import Frame

logger = get_logger(__name__)


class OpenCVCameraSource(CameraSource):
    """CameraSource implementation backed by cv2.VideoCapture."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0

    def open(self) -> None:
        settings = self._settings
        capture = cv2.VideoCapture(settings.camera_index)

        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Camera index {settings.camera_index} cannot be opened.",
                details={"camera_index": settings.camera_index},
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
        if settings.camera_target_fps:
            capture.set(cv2.CAP_PROP_FPS, settings.camera_target_fps)

        self._capture = capture
        self._frame_id = 0
        logger.info(
            "Camera %s opened (%sx%s @ target %s fps)",
            settings.camera_index,
            settings.camera_width,
            settings.camera_height,
            settings.camera_target_fps,
        )

    def read(self) -> Frame | None:
        if self._capture is None:
            raise CameraError("read() called before open(); camera is not initialized.")

        ok, image = self._capture.read()
        if not ok or image is None:
            logger.warning("Frame read failed (camera_index=%s)", self._settings.camera_index)
            return None

        frame = Frame(image=image, frame_id=self._frame_id, timestamp=datetime.now(timezone.utc))
        self._frame_id += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            logger.info("Camera %s released", self._settings.camera_index)
            self._capture = None

    def __enter__(self) -> "OpenCVCameraSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
