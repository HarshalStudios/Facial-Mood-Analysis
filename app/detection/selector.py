"""
Primary face selection.

When multiple faces are detected, exactly one (at most) needs to be
marked as the "primary" face that the rest of the real-time pipeline
tracks. The strategy is intentionally pluggable — swapping strategies
must not require touching the detector or the camera module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.face import Face
from app.schemas.frame import Frame


class FaceSelector(ABC):
    """Abstract base class for primary-face selection strategies."""

    @abstractmethod
    def select(self, faces: list[Face], frame: Frame | None = None) -> Face | None:
        """Return the primary face, or None if `faces` is empty."""
        raise NotImplementedError


class LargestFaceSelector(FaceSelector):
    """Default strategy: pick the face with the largest bounding-box area."""

    def select(self, faces: list[Face], frame: Frame | None = None) -> Face | None:
        if not faces:
            return None
        return max(faces, key=lambda f: f.bounding_box.width * f.bounding_box.height)


class HighestConfidenceFaceSelector(FaceSelector):
    """
    Pick the face with the highest detector confidence.

    Only meaningful for detector backends that actually provide a
    confidence score; falls back to LargestFaceSelector when every
    candidate has confidence=None.
    """

    def select(self, faces: list[Face], frame: Frame | None = None) -> Face | None:
        if not faces:
            return None
        scored = [f for f in faces if f.confidence is not None]
        if not scored:
            return LargestFaceSelector().select(faces, frame)
        return max(scored, key=lambda f: f.confidence)


class CenterMostFaceSelector(FaceSelector):
    """Pick the face whose bounding-box center is closest to the frame center."""

    def select(self, faces: list[Face], frame: Frame | None = None) -> Face | None:
        if not faces:
            return None
        if frame is None:
            return LargestFaceSelector().select(faces, frame)

        fw, fh = frame.resolution
        cx, cy = fw / 2, fh / 2

        def dist(face: Face) -> float:
            b = face.bounding_box
            face_cx = b.x + b.width / 2
            face_cy = b.y + b.height / 2
            return (face_cx - cx) ** 2 + (face_cy - cy) ** 2

        return min(faces, key=dist)


_STRATEGIES: dict[str, type[FaceSelector]] = {
    "largest": LargestFaceSelector,
    "highest_confidence": HighestConfidenceFaceSelector,
    "center_most": CenterMostFaceSelector,
}


def get_face_selector(strategy: str) -> FaceSelector:
    """Look up a FaceSelector by name (see Settings.primary_face_strategy)."""
    try:
        return _STRATEGIES[strategy]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown primary_face_strategy '{strategy}'. "
            f"Valid options: {list(_STRATEGIES)}"
        ) from exc
