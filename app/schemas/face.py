"""Data contract for a detected face (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Face:
    """A single detected face within a Frame."""

    bounding_box: BoundingBox
    crop: np.ndarray  # BGR face crop
    confidence: float | None  # None when the detector backend doesn't provide one
    face_id: int = 0
    is_primary: bool = False
    landmarks: list[tuple[int, int]] | None = None  # populated in a future phase
