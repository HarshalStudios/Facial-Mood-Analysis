"""
LBP histogram feature (Phase 4, feature indices 7-16).

Converts the Phase 3 LBP *representation* (a texture-code image) into
the 10 normalized histogram bins used in the feature vector. See
app.representations.engine for how the representation itself is
generated (skimage.feature.local_binary_pattern).

BINNING STRATEGY (must exactly match Phase 3's LBP config):

For method="uniform" with P sample points, skimage's uniform LBP codes
are integers in [0, P+1] inclusive: codes 0..P-1... actually 0..P are
the P+1 "uniform" patterns (by number of 0->1 transitions... by number
of set bits along the uniform patterns) and code P+1 collects every
"non-uniform" pattern. That is exactly P+2 distinct code values.

With the project's default P=8, that is exactly 10 distinct codes —
which is exactly `lbp_num_bins=10`. So for the default configuration,
every possible LBP code maps to its own bin: no information is
discarded, no truncation happens. This function enforces that the
config is actually in this compatible state (method="uniform" and
lbp_num_bins == lbp_n_points + 2) and raises FeatureExtractionError
instead of silently truncating if someone changes P without updating
lbp_num_bins to match, per Phase 4 spec sec. 14.

Other LBP methods (e.g. "default", which produces codes in [0, 255])
are not supported by this function — mapping 256 raw codes onto 10
bins would require an arbitrary binning scheme this project does not
want to invent silently.
"""

from __future__ import annotations

import numpy as np

from app.core.config import Settings
from app.core.exceptions import FeatureExtractionError


def compute_lbp_histogram(lbp_image: np.ndarray, settings: Settings) -> np.ndarray:
    if lbp_image is None:
        raise FeatureExtractionError("lbp_image is None; LBP representation must be enabled for Phase 4")

    method = settings.lbp_method
    p = settings.lbp_n_points
    num_bins = settings.lbp_num_bins

    if method != "uniform":
        raise FeatureExtractionError(
            "LBP histogram binning is only defined for method='uniform'; "
            "other methods would require an undocumented, arbitrary binning "
            "scheme to fit 10 bins.",
            details={"configured_method": method},
        )

    expected_bins = p + 2
    if num_bins != expected_bins:
        raise FeatureExtractionError(
            "lbp_num_bins does not match lbp_n_points for uniform LBP "
            "(expected lbp_n_points + 2 distinct codes). Change the config "
            "to match rather than truncating LBP information.",
            details={"lbp_n_points": p, "configured_num_bins": num_bins, "expected_num_bins": expected_bins},
        )

    histogram, _ = np.histogram(lbp_image, bins=expected_bins, range=(0, expected_bins))

    total = int(histogram.sum())
    if total == 0:
        # Should not happen for a real non-empty image, but a zero-size
        # or fully-masked-out input shouldn't raise a ZeroDivisionError.
        return np.zeros(expected_bins, dtype=np.float64)

    return histogram.astype(np.float64) / float(total)
