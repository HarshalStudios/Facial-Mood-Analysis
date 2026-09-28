"""
RepresentationEngine (Phase 3).

Turns one validated face crop (BGR, uint8, from the Phase 2
FaceDetectionService) into a set of complementary DIP representations:
RGB, Grayscale, CLAHE, Canny, Sobel, LBP.

This module knows nothing about emotion, features, FER2013, fusion,
temporal smoothing, or the API layer — its only job is image-processing
transforms, so it can be swapped, extended, or ablated (Phase 9)
without touching anything downstream.

Color-space contract (see README "Color space contract" and Phase 2):
frames/crops are OpenCV-native BGR, uint8. This engine never silently
assumes RGB. The "rgb" representation is therefore, honestly, still
BGR-ordered data — it is simply the unmodified face crop — and its
metadata says so explicitly. Any consumer that needs true RGB channel
order (e.g. DeepFace in a later phase) must convert at its own
boundary, exactly as the Phase 2 README already requires.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from app.core.config import Settings, get_settings
from app.core.exceptions import RepresentationError
from app.core.logging import get_logger
from app.representations.base import RepresentationGenerator
from app.schemas.representations import RepresentationMeta, Representations

logger = get_logger(__name__)

_EXPECTED_CHANNELS = 3


class RepresentationEngine(RepresentationGenerator):
    """Generates the configured set of representations from one face crop."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, face_crop: np.ndarray) -> Representations:
        """
        Given a BGR face crop, return the enabled representations.

        Raises RepresentationError for any input that isn't a valid
        BGR uint8 image (None, empty, wrong ndim/channels, bad dtype).
        Never lets a raw cv2/skimage exception propagate uncaught.
        """
        self._validate_input(face_crop)
        settings = self._settings

        rgb = face_crop.copy()
        height, width = rgb.shape[:2]

        metadata: dict[str, RepresentationMeta] = {
            "rgb": RepresentationMeta(
                name="rgb",
                source="face_crop",
                width=width,
                height=height,
                dtype=str(rgb.dtype),
                color_space="BGR",
                params={},
            )
        }

        needs_gray = any(
            (
                settings.enable_grayscale,
                settings.enable_clahe,
                settings.enable_canny,
                settings.enable_sobel,
                settings.enable_lbp,
            )
        )

        gray: np.ndarray | None = None
        if needs_gray:
            try:
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            except cv2.error as exc:
                raise RepresentationError(
                    "Failed to compute grayscale conversion", details={"cv2_error": str(exc)}
                ) from exc

        result = Representations(rgb=rgb, metadata=metadata)

        if settings.enable_grayscale:
            result.grayscale = gray
            metadata["grayscale"] = RepresentationMeta(
                name="grayscale",
                source="rgb",
                width=width,
                height=height,
                dtype=str(gray.dtype),
                color_space="GRAY",
                params={},
            )

        if settings.enable_clahe:
            clahe_img, params = self._make_clahe(gray, settings)
            result.clahe = clahe_img
            metadata["clahe"] = RepresentationMeta(
                name="clahe",
                source="grayscale",
                width=width,
                height=height,
                dtype=str(clahe_img.dtype),
                color_space="GRAY",
                params=params,
            )

        if settings.enable_canny:
            canny_img, params = self._make_canny(gray, settings)
            result.canny = canny_img
            metadata["canny"] = RepresentationMeta(
                name="canny",
                source="grayscale",
                width=width,
                height=height,
                dtype=str(canny_img.dtype),
                color_space="BINARY_EDGE",
                params=params,
            )

        if settings.enable_sobel:
            sobel_img, params = self._make_sobel(gray, settings)
            result.sobel = sobel_img
            metadata["sobel"] = RepresentationMeta(
                name="sobel",
                source="grayscale",
                width=width,
                height=height,
                dtype=str(sobel_img.dtype),
                color_space="GRADIENT_MAGNITUDE",
                params=params,
            )

        if settings.enable_lbp:
            lbp_img, params = self._make_lbp(gray, settings)
            result.lbp = lbp_img
            metadata["lbp"] = RepresentationMeta(
                name="lbp",
                source="grayscale",
                width=width,
                height=height,
                dtype=str(lbp_img.dtype),
                color_space="LBP_CODE",
                params=params,
            )

        return result

    # ------------------------------------------------------------------
    # Individual representation generators
    # ------------------------------------------------------------------

    @staticmethod
    def _make_clahe(gray: np.ndarray, settings: Settings) -> tuple[np.ndarray, dict]:
        params = {
            "clipLimit": settings.clahe_clip_limit,
            "tileGridSize": settings.clahe_tile_grid_size,
        }
        try:
            clahe = cv2.createCLAHE(
                clipLimit=settings.clahe_clip_limit,
                tileGridSize=(settings.clahe_tile_grid_size, settings.clahe_tile_grid_size),
            )
            out = clahe.apply(gray)
        except cv2.error as exc:
            raise RepresentationError(
                "Failed to compute CLAHE representation", details={"cv2_error": str(exc), "params": params}
            ) from exc
        return out, params

    @staticmethod
    def _make_canny(gray: np.ndarray, settings: Settings) -> tuple[np.ndarray, dict]:
        params = {
            "lowThreshold": settings.canny_low_threshold,
            "highThreshold": settings.canny_high_threshold,
            "apertureSize": settings.canny_aperture_size,
            "L2gradient": settings.canny_l2_gradient,
        }
        try:
            out = cv2.Canny(
                gray,
                settings.canny_low_threshold,
                settings.canny_high_threshold,
                apertureSize=settings.canny_aperture_size,
                L2gradient=settings.canny_l2_gradient,
            )
        except cv2.error as exc:
            raise RepresentationError(
                "Failed to compute Canny representation", details={"cv2_error": str(exc), "params": params}
            ) from exc
        return out, params

    @staticmethod
    def _make_sobel(gray: np.ndarray, settings: Settings) -> tuple[np.ndarray, dict]:
        """
        Gradient representation strategy: compute Sobel Gx and Gy
        (float64, to avoid clipping/overflow), combine into a gradient
        MAGNITUDE image via cv2.magnitude, then normalize back to uint8
        via cv2.convertScaleAbs for a displayable/storable representation.

        Gx and Gy themselves are not exposed as separate outputs — only
        the combined magnitude is kept as the "sobel" representation.
        The Phase 4 "gradient energy" feature will be derived from this
        magnitude image (or recomputed from gray, per that phase's own
        contract), not computed here.
        """
        params = {
            "ksize": settings.sobel_kernel_size,
            "scale": settings.sobel_scale,
            "delta": settings.sobel_delta,
            "combine": "magnitude(Gx, Gy)",
        }
        try:
            gx = cv2.Sobel(
                gray,
                cv2.CV_64F,
                1,
                0,
                ksize=settings.sobel_kernel_size,
                scale=settings.sobel_scale,
                delta=settings.sobel_delta,
            )
            gy = cv2.Sobel(
                gray,
                cv2.CV_64F,
                0,
                1,
                ksize=settings.sobel_kernel_size,
                scale=settings.sobel_scale,
                delta=settings.sobel_delta,
            )
            magnitude = cv2.magnitude(gx, gy)
            out = cv2.convertScaleAbs(magnitude)
        except cv2.error as exc:
            raise RepresentationError(
                "Failed to compute Sobel representation", details={"cv2_error": str(exc), "params": params}
            ) from exc
        return out, params

    @staticmethod
    def _make_lbp(gray: np.ndarray, settings: Settings) -> tuple[np.ndarray, dict]:
        params = {
            "P": settings.lbp_n_points,
            "R": settings.lbp_radius,
            "method": settings.lbp_method,
        }
        try:
            codes = local_binary_pattern(
                gray,
                P=settings.lbp_n_points,
                R=settings.lbp_radius,
                method=settings.lbp_method,
            )
        except Exception as exc:  # skimage doesn't raise a narrow exception type
            raise RepresentationError(
                "Failed to compute LBP representation", details={"error": str(exc), "params": params}
            ) from exc

        max_code = float(codes.max()) if codes.size else 0.0
        if max_code > 255.0:
            # Only possible with an unusually large P; keep the engine
            # honest instead of silently wrapping/clipping codes.
            raise RepresentationError(
                "LBP codes exceed uint8 range for the configured P",
                details={"max_code": max_code, "params": params},
            )
        out = codes.astype(np.uint8)
        return out, params

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(face_crop: np.ndarray) -> None:
        if face_crop is None:
            raise RepresentationError("face_crop is None")

        if not isinstance(face_crop, np.ndarray):
            raise RepresentationError(
                "face_crop must be a numpy ndarray", details={"type": str(type(face_crop))}
            )

        if face_crop.size == 0:
            raise RepresentationError("face_crop is empty", details={"shape": face_crop.shape})

        if face_crop.ndim != 3 or face_crop.shape[2] != _EXPECTED_CHANNELS:
            raise RepresentationError(
                "face_crop must have shape (H, W, 3)",
                details={"shape": face_crop.shape, "ndim": face_crop.ndim},
            )

        if face_crop.dtype != np.uint8:
            raise RepresentationError(
                "face_crop must be dtype uint8", details={"dtype": str(face_crop.dtype)}
            )

        h, w = face_crop.shape[:2]
        if h <= 0 or w <= 0:
            raise RepresentationError(
                "face_crop has invalid dimensions", details={"width": w, "height": h}
            )
