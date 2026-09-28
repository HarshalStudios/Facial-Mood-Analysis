"""
Optional debug export of individual representation images (Phase 3).

Explicitly opt-in — nothing in the pipeline calls this automatically
for every webcam frame. Intended for a developer inspecting one
selected face at a time (see scripts/test_representations.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.logging import get_logger
from app.schemas.representations import Representations

logger = get_logger(__name__)

_EXPORTABLE = ["rgb", "grayscale", "clahe", "canny", "sobel", "lbp"]


def export_representations(representations: Representations, output_dir: str | Path) -> dict[str, str]:
    """
    Save every representation present on `representations` as a PNG
    under `output_dir` (created if missing): rgb.png, grayscale.png,
    clahe.png, canny.png, sobel.png, lbp.png. Representations that were
    disabled (None) are skipped, not written as blank files.

    Returns a dict of {representation_name: path_written}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for name in _EXPORTABLE:
        array = getattr(representations, name, None)
        if array is None:
            continue

        image = array
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        path = output_dir / f"{name}.png"
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise IOError(f"cv2.imwrite failed to write {path}")
        written[name] = str(path)

    logger.info("Exported %d representation(s) to %s", len(written), output_dir)
    return written
