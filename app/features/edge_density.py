"""
Edge density feature (Phase 4, feature index 17).

edge_density = (number of edge pixels) / (total number of pixels) in
the Phase 3 Canny representation, which is already a binary {0, 255}
edge map. No additional thresholding is applied — the Canny
representation's own configured thresholds (Phase 3 config) are the
only thresholding involved.
"""

from __future__ import annotations

import numpy as np

from app.core.exceptions import FeatureExtractionError


def compute_edge_density(canny_image: np.ndarray) -> float:
    if canny_image is None:
        raise FeatureExtractionError("canny_image is None; Canny representation must be enabled for Phase 4")
    if canny_image.size == 0:
        raise FeatureExtractionError("canny_image is empty")

    return float(np.count_nonzero(canny_image)) / float(canny_image.size)
