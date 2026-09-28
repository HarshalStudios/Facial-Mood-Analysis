"""
FER2013 sample -> RepresentationEngine-compatible image (Phase 5 spec
sec. 10-11).

FER2013 is already a cropped facial dataset, so — unlike the real-time
webcam pipeline — nothing here runs face detection. It only:

1. Parses the source pixel representation into a 2D array (CSV rows store
   pixels as a space-separated string; other supported layouts hand us an
   array directly).
2. Converts that array into the (H, W, 3) BGR uint8 shape
   `RepresentationEngine.generate()` requires.

Step 2 is a *compatibility* conversion, not a color reconstruction: FER2013
is grayscale, so converting to 3 channels by replication does not create
any new color information (Phase 5 spec sec. 11). This is documented
explicitly here and in the metadata this module's caller attaches to the
generated feature dataset, so nobody downstream mistakes "3-channel" for
"has real color".
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.exceptions import DatasetError

# Documents *why* a grayscale-replicated image is not new information, for
# anything (metadata, reports) that wants to reference this decision by name.
GRAYSCALE_TO_BGR_STRATEGY = "channel_replication_no_new_color_information"


def parse_pixel_string(pixel_str: str, width: int, height: int) -> np.ndarray:
    """
    Parse FER2013's classic space-separated pixel string into a (height,
    width) uint8 array.

    Raises DatasetError (not a bare ValueError) if the token count doesn't
    match width*height or a token isn't a valid 0-255 integer, so the
    caller can log this as one malformed record instead of crashing the
    whole load.
    """
    tokens = pixel_str.split()
    expected = width * height
    if len(tokens) != expected:
        raise DatasetError(
            "Pixel string has an unexpected token count",
            details={"expected": expected, "actual": len(tokens), "width": width, "height": height},
        )

    try:
        values = np.array([int(t) for t in tokens], dtype=np.int32)
    except ValueError as exc:
        raise DatasetError("Pixel string contains a non-integer token", details={"error": str(exc)}) from exc

    if values.min(initial=0) < 0 or values.max(initial=0) > 255:
        raise DatasetError(
            "Pixel string contains value(s) outside the 0-255 range",
            details={"min": int(values.min()), "max": int(values.max())},
        )

    return values.astype(np.uint8).reshape(height, width)


def to_representation_input(image: np.ndarray) -> np.ndarray:
    """
    Convert a loaded FER2013 image into the (H, W, 3) BGR uint8 array
    RepresentationEngine.generate() requires.

    - (H, W) grayscale -> channel-replicated to (H, W, 3). No new color
      information is created (see module docstring).
    - (H, W, 3) already-3-channel input is passed through unchanged
      (dtype-normalized to uint8 if needed), covering a dataset variant
      that already stores 3 channels.

    Raises DatasetError for any other shape/dtype, rather than guessing.
    """
    if image is None:
        raise DatasetError("image is None; cannot convert to representation input")

    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.floating):
            raise DatasetError("image has a non-numeric dtype", details={"dtype": str(image.dtype)})
        clipped = np.clip(image, 0, 255)
        image = clipped.astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image)

    raise DatasetError(
        "image has an unsupported shape for representation input",
        details={"shape": image.shape},
    )
