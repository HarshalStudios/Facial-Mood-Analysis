"""
Gradient energy, brightness, contrast, sharpness features (Phase 4,
feature indices 18-21). All derived from the Phase 3 grayscale
representation.

GRADIENT ENERGY — deliberate deviation from a literal reading of the
Phase 4 spec, documented per its own "if you change it, explain why"
rule: the spec's sec. 17 example source is the Phase 3 Sobel
*representation*, but that representation is `cv2.convertScaleAbs`'d to
uint8 for display/storage (see app/representations/engine.py), which
clips and rescales magnitude — a lossy transform that would make
"gradient energy" depend on that clipping instead of the image's actual
gradient content, and would also make this feature unavailable whenever
`enable_sobel=False` in a Phase 9 ablation run. Instead, gradient_energy
recomputes Sobel directly from the grayscale representation, in
float64, on intensity normalized to [0, 1] first (so the value doesn't
trivially scale with 8-bit range or image size):

    gray_f = grayscale.astype(float64) / 255.0
    Gx = Sobel_x(gray_f), Gy = Sobel_y(gray_f)
    gradient_energy = mean(Gx^2 + Gy^2)

using the same `sobel_kernel_size` config value as Phase 3, so the two
stay parameter-consistent even though they're computed independently.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.config import Settings
from app.core.exceptions import FeatureExtractionError


def _require_grayscale(grayscale: np.ndarray) -> None:
    if grayscale is None:
        raise FeatureExtractionError(
            "grayscale is None; Grayscale representation must be enabled for Phase 4"
        )
    if grayscale.size == 0:
        raise FeatureExtractionError("grayscale is empty")


def compute_gradient_energy(grayscale: np.ndarray, settings: Settings) -> float:
    _require_grayscale(grayscale)

    gray_f = grayscale.astype(np.float64) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=settings.sobel_kernel_size)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=settings.sobel_kernel_size)
    magnitude_squared = gx**2 + gy**2
    return float(np.mean(magnitude_squared))


def compute_brightness(grayscale: np.ndarray) -> float:
    _require_grayscale(grayscale)
    return float(np.mean(grayscale.astype(np.float64)))


def compute_contrast(grayscale: np.ndarray) -> float:
    _require_grayscale(grayscale)
    return float(np.std(grayscale.astype(np.float64)))


def compute_sharpness(grayscale: np.ndarray) -> float:
    _require_grayscale(grayscale)
    laplacian = cv2.Laplacian(grayscale, cv2.CV_64F)
    return float(np.var(laplacian))
