"""
The 22-dimensional feature vector contract.

THIS ORDERING IS FIXED. It must never change once later phases start
training and saving models against it — the fusion model (Phase 6) and
scaler are trained on this exact column order, so changing it silently
would invalidate every saved model artifact.

If the ordering ever must change, bump FEATURE_VECTOR_VERSION and treat
it as a breaking change to any saved model.

Layout (indices 0-21):

    0   emotion_angry
    1   emotion_disgust
    2   emotion_fear
    3   emotion_happy
    4   emotion_sad
    5   emotion_surprise
    6   emotion_neutral
    7   lbp_bin_0
    8   lbp_bin_1
    9   lbp_bin_2
    10  lbp_bin_3
    11  lbp_bin_4
    12  lbp_bin_5
    13  lbp_bin_6
    14  lbp_bin_7
    15  lbp_bin_8
    16  lbp_bin_9
    17  edge_density
    18  gradient_energy
    19  brightness
    20  contrast
    21  sharpness
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_VECTOR_VERSION = 1
FEATURE_VECTOR_LENGTH = 22

FEATURE_NAMES: list[str] = [
    "emotion_angry",
    "emotion_disgust",
    "emotion_fear",
    "emotion_happy",
    "emotion_sad",
    "emotion_surprise",
    "emotion_neutral",
    "lbp_bin_0",
    "lbp_bin_1",
    "lbp_bin_2",
    "lbp_bin_3",
    "lbp_bin_4",
    "lbp_bin_5",
    "lbp_bin_6",
    "lbp_bin_7",
    "lbp_bin_8",
    "lbp_bin_9",
    "edge_density",
    "gradient_energy",
    "brightness",
    "contrast",
    "sharpness",
]

assert len(FEATURE_NAMES) == FEATURE_VECTOR_LENGTH

# Authoritative index lookup (Phase 4 sec. 6) — every module that needs a
# feature's position looks it up here rather than hard-coding a number.
FEATURE_INDICES: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
FEATURE_COUNT: int = FEATURE_VECTOR_LENGTH


@dataclass(frozen=True)
class FeatureMeta:
    """Research/debug metadata for one of the 22 features (Phase 4 sec. 7)."""

    name: str
    index: int
    source: str
    description: str
    expected_range: str
    normalized: bool


def _emotion_meta(name: str, index: int) -> FeatureMeta:
    label = name.removeprefix("emotion_")
    return FeatureMeta(
        name=name,
        index=index,
        source="Deep emotion provider (DeepFace)",
        description=f"Predicted probability of the '{label}' emotion class",
        expected_range="0-1 (DeepFace percentages divided by 100)",
        normalized=True,
    )


def _lbp_meta(name: str, index: int, bin_index: int) -> FeatureMeta:
    return FeatureMeta(
        name=name,
        index=index,
        source="LBP representation (Phase 3)",
        description=f"Normalized uniform-LBP histogram bin {bin_index}",
        expected_range="0-1 (sums to ~1 across all 10 LBP bins)",
        normalized=True,
    )


FEATURE_METADATA: dict[str, FeatureMeta] = {
    **{name: _emotion_meta(name, i) for i, name in enumerate(FEATURE_NAMES[0:7])},
    **{name: _lbp_meta(name, 7 + i, i) for i, name in enumerate(FEATURE_NAMES[7:17])},
    "edge_density": FeatureMeta(
        name="edge_density",
        index=17,
        source="Canny representation (Phase 3)",
        description="Fraction of pixels that are edge pixels: count_nonzero(canny) / canny.size",
        expected_range="0-1",
        normalized=True,
    ),
    "gradient_energy": FeatureMeta(
        name="gradient_energy",
        index=18,
        source="Grayscale representation (Sobel recomputed internally, see FeatureExtractionError docs)",
        description=(
            "Mean squared Sobel gradient magnitude computed on grayscale "
            "intensity normalized to [0, 1] before differentiation"
        ),
        expected_range="Typically small positive float (0-~2 for natural face images), not bounded to [0,1]",
        normalized=False,
    ),
    "brightness": FeatureMeta(
        name="brightness",
        index=19,
        source="Grayscale representation (Phase 3)",
        description="Mean grayscale pixel intensity",
        expected_range="0-255 (8-bit grayscale)",
        normalized=False,
    ),
    "contrast": FeatureMeta(
        name="contrast",
        index=20,
        source="Grayscale representation (Phase 3)",
        description="Standard deviation of grayscale pixel intensity",
        expected_range=">= 0, typically 0-128 for 8-bit grayscale",
        normalized=False,
    ),
    "sharpness": FeatureMeta(
        name="sharpness",
        index=21,
        source="Grayscale representation (Phase 3), Laplacian",
        description="Variance of the Laplacian of grayscale intensity (blur/sharpness indicator)",
        expected_range=">= 0, unbounded above",
        normalized=False,
    ),
}

assert set(FEATURE_METADATA.keys()) == set(FEATURE_NAMES)


@dataclass
class FeatureVector:
    """A validated 22-dimensional feature vector with named access."""

    values: np.ndarray  # shape (22,), dtype float32/float64

    def __post_init__(self) -> None:
        if self.values.shape != (FEATURE_VECTOR_LENGTH,):
            raise ValueError(
                f"FeatureVector must have shape ({FEATURE_VECTOR_LENGTH},), "
                f"got {self.values.shape}"
            )
        if not np.all(np.isfinite(self.values)):
            raise ValueError("FeatureVector contains NaN or infinite value(s)")

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values.tolist()))

    def __getitem__(self, name: str) -> float:
        return float(self.values[FEATURE_NAMES.index(name)])
