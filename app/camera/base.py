"""
CameraSource interface.

Implemented in Phase 2. Abstracted so a live webcam, a video file, or a
test fixture (fake frame generator) can all be swapped in without
touching any downstream module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.frame import Frame


class CameraSource(ABC):
    """Abstract base class for anything that produces a stream of Frames."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the camera/video resource."""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Frame | None:
        """Read the next frame, or None if the stream has ended."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release the camera/video resource."""
        raise NotImplementedError
