"""
Face detection service.

Orchestrates the Phase 2 pipeline on a single frame:

    Frame -> FaceDetector.detect() -> clamp bbox -> pad -> size check
          -> crop -> FaceSelector.select() marks is_primary

This is the boundary Phase 3 (representations) will call into — it
should only ever need a `Frame` in and a `list[Face]` (with at most one
`is_primary=True`) out.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.detection.base import FaceDetector
from app.detection.bbox_utils import apply_padding, clamp_bbox, is_face_large_enough
from app.detection.cropping import crop_face
from app.detection.selector import get_face_selector
from app.schemas.face import Face
from app.schemas.frame import Frame

logger = get_logger(__name__)


@dataclass
class FaceDetectionResult:
    faces: list[Face]
    rejected_count: int  # detections dropped for being degenerate/too small


class FaceDetectionService:
    """Combines a FaceDetector with bbox validation, cropping, and primary selection."""

    def __init__(self, detector: FaceDetector, settings: Settings | None = None) -> None:
        self._detector = detector
        self._settings = settings or get_settings()
        self._selector = get_face_selector(self._settings.primary_face_strategy)

    def process(self, frame: Frame) -> FaceDetectionResult:
        settings = self._settings
        frame_w, frame_h = frame.resolution

        raw_faces = self._detector.detect(frame)

        valid_faces: list[Face] = []
        rejected_count = 0

        for raw in raw_faces:
            clamped = clamp_bbox(raw.bounding_box, frame_w, frame_h)
            if clamped is None:
                rejected_count += 1
                logger.debug("Rejected face: bbox degenerate after clamping: %s", raw.bounding_box)
                continue

            padded = apply_padding(clamped, settings.face_padding_ratio, frame_w, frame_h)

            if not is_face_large_enough(padded, settings.min_face_width, settings.min_face_height):
                rejected_count += 1
                logger.debug("Rejected face: too small (%sx%s)", padded.width, padded.height)
                continue

            crop = crop_face(frame.image, padded)

            valid_faces.append(
                Face(
                    bounding_box=padded,
                    crop=crop,
                    confidence=raw.confidence,
                    face_id=raw.face_id,
                    is_primary=False,
                )
            )

        primary = self._selector.select(valid_faces, frame)
        if primary is not None:
            primary.is_primary = True

        return FaceDetectionResult(faces=valid_faces, rejected_count=rejected_count)
