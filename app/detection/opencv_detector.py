"""
OpenCV Haar Cascade face detector.

Chosen for Phase 2 because it:
- ships with OpenCV (no extra model download / extra dependency beyond
  opencv-python, which the project already needs for image processing
  in later phases)
- is fast enough for real-time webcam use on CPU
- handles frontal/near-frontal faces well, which matches the project's
  webcam-in-front-of-user use case

IMPORTANT — confidence: cv2.CascadeClassifier.detectMultiScale does not
produce a calibrated detection confidence/probability. Per the project
requirement not to invent confidence values, this detector reports
confidence=None for every detection. Primary-face selection therefore
defaults to the "largest" strategy (see app.detection.selector) rather
than "highest_confidence" for this backend.

This can be swapped for a detector that does provide real confidence
(e.g. an OpenCV DNN face detector or MediaPipe) later by implementing
FaceDetector again — nothing outside this module needs to change.
"""

from __future__ import annotations

import cv2

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelLoadError
from app.core.logging import get_logger
from app.detection.base import FaceDetector
from app.schemas.face import BoundingBox, Face
from app.schemas.frame import Frame

logger = get_logger(__name__)


class HaarCascadeFaceDetector(FaceDetector):
    """FaceDetector implementation backed by OpenCV's frontal-face Haar cascade."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise ModelLoadError(
                "Failed to load Haar cascade for face detection",
                details={"cascade_path": cascade_path},
            )
        logger.info("HaarCascadeFaceDetector loaded from %s", cascade_path)

    def detect(self, frame: Frame) -> list[Face]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._settings.haar_scale_factor,
            minNeighbors=self._settings.haar_min_neighbors,
            minSize=(self._settings.min_face_width, self._settings.min_face_height),
        )

        faces: list[Face] = []
        for i, (x, y, w, h) in enumerate(detections):
            bbox = BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h))
            crop = frame.image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
            faces.append(
                Face(
                    bounding_box=bbox,
                    crop=crop,
                    confidence=None,
                    face_id=i,
                    is_primary=False,
                )
            )

        return faces
