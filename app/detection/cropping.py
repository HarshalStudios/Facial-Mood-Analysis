"""
Face cropping.

Crops a validated (already clamped) bounding box out of a frame. Does
NOT resize the crop — different downstream consumers (real-time
pipeline vs. FER2013 dataset processing) may need different target
sizes, and that decision is deliberately deferred to those later
phases rather than baked in here.
"""

from __future__ import annotations

import numpy as np

from app.schemas.face import BoundingBox


def crop_face(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """
    Crop `bbox` out of `image`. `bbox` must already be clamped to the
    frame (see bbox_utils.clamp_bbox) — this function does not
    re-validate bounds itself.
    """
    x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
    crop = image[y : y + h, x : x + w]

    if crop.size == 0:
        raise ValueError(f"Resulting crop is empty for bbox={bbox}")

    return crop
