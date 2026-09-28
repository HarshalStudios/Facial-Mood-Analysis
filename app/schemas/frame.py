"""Data contract for a single captured camera frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


@dataclass
class Frame:
    """A single raw frame captured from the camera (Phase 2)."""

    image: np.ndarray  # BGR, shape (H, W, 3)
    frame_id: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (width, height)."""
        h, w = self.image.shape[:2]
        return w, h
