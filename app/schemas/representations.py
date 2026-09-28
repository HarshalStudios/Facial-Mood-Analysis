"""
Data contract for the multi-representation output of a single face
(Phase 3). RepresentationGenerator implementations must populate this
structure; downstream consumers (FeatureExtractor) must not depend on
how each field was produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RepresentationMeta:
    """
    Descriptive metadata for a single generated representation.

    Populated by RepresentationEngine alongside the array itself so
    downstream consumers (the Phase 4 FeatureExtractor, and eventually
    the frontend) can introspect a representation without re-deriving
    its properties from the raw array.
    """

    name: str
    source: str  # what it was derived from, e.g. "face_crop" or "grayscale"
    width: int
    height: int
    dtype: str
    color_space: str  # "BGR" | "GRAY" | "BINARY_EDGE" | "GRADIENT_MAGNITUDE" | "LBP_CODE"
    params: dict = field(default_factory=dict)


@dataclass
class Representations:
    rgb: np.ndarray
    grayscale: np.ndarray | None = None
    clahe: np.ndarray | None = None
    canny: np.ndarray | None = None
    sobel: np.ndarray | None = None
    lbp: np.ndarray | None = None
    optical_flow: np.ndarray | None = None  # future extension, not Phase 1-3

    # Keyed by representation name ("rgb", "grayscale", "clahe", "canny",
    # "sobel", "lbp"). Populated by RepresentationEngine for every
    # representation it actually generates; absent entries mean that
    # representation was disabled via config, not that it failed.
    metadata: dict[str, RepresentationMeta] = field(default_factory=dict)

    def enabled_names(self) -> list[str]:
        """Names of representations that were actually generated."""
        return list(self.metadata.keys())
