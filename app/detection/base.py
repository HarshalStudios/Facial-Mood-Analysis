"""
FaceDetector interface.

Implemented starting Phase 2. Defined now so no other module ever
depends on a concrete detector (OpenCV Haar cascade, MediaPipe, MTCNN,
etc.) — only on this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.face import Face
from app.schemas.frame import Frame


class FaceDetector(ABC):
    """Abstract base class for all face detector implementations."""

    @abstractmethod
    def detect(self, frame: Frame) -> list[Face]:
        """Detect faces in a frame. Returns an empty list if none found."""
        raise NotImplementedError
