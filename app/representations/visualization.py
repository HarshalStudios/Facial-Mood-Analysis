"""
Development visualization for RepresentationEngine output (Phase 3).

Deliberately separate from engine.py: the RepresentationEngine has no
GUI dependency and must remain usable in a headless production/API
context. This module is only for a developer visually sanity-checking
that RGB/Grayscale/CLAHE/Canny/Sobel/LBP look like what they should —
it is never called from the core pipeline.

Layout produced:

    ┌────────────┬────────────┬────────────┐
    │ RGB        │ Grayscale  │ CLAHE      │
    ├────────────┼────────────┼────────────┤
    │ Canny      │ Sobel      │ LBP        │
    └────────────┴────────────┴────────────┘

Any representation that was disabled (missing from `Representations`)
is rendered as a black panel labeled "disabled" rather than raising,
so this works for any enable/disable configuration from Phase 9.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.logging import get_logger
from app.schemas.representations import Representations

logger = get_logger(__name__)

_PANEL_ORDER = ["rgb", "grayscale", "clahe", "canny", "sobel", "lbp"]
_PANEL_LABELS = {
    "rgb": "RGB",
    "grayscale": "Grayscale",
    "clahe": "CLAHE",
    "canny": "Canny",
    "sobel": "Sobel",
    "lbp": "LBP",
}


def _to_display_bgr(array: np.ndarray | None, panel_size: tuple[int, int]) -> np.ndarray:
    """Resize (for display only) and force to 3-channel BGR uint8 for concatenation."""
    w, h = panel_size

    if array is None:
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            panel, "disabled", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA
        )
        return panel

    img = array
    if img.dtype != np.uint8:
        # LBP codes / anything non-uint8 that slips through: normalize
        # for display purposes only, never for the actual data contract.
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


def build_representation_grid(
    representations: Representations,
    panel_size: tuple[int, int] = (200, 200),
) -> np.ndarray:
    """
    Build a single BGR image tiling all six representations in a 2x3
    grid with text labels, for on-screen display or saving to disk.
    """
    panels = []
    for name in _PANEL_ORDER:
        array = getattr(representations, name, None)
        panel = _to_display_bgr(array, panel_size)
        panel = panel.copy()
        cv2.putText(
            panel,
            _PANEL_LABELS[name],
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)

    top_row = np.hstack(panels[0:3])
    bottom_row = np.hstack(panels[3:6])
    return np.vstack([top_row, bottom_row])


def show_representations(
    representations: Representations,
    window_name: str = "Representations (press any key to close)",
    panel_size: tuple[int, int] = (200, 200),
) -> None:
    """
    Open an OpenCV window showing the 2x3 representation grid. Blocks
    until a key is pressed. Manual/dev use only (see
    scripts/test_representations.py) — never called from the pipeline.
    """
    grid = build_representation_grid(representations, panel_size)
    cv2.imshow(window_name, grid)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def save_representation_grid(
    representations: Representations,
    output_path: str,
    panel_size: tuple[int, int] = (200, 200),
) -> str:
    """Save the 2x3 representation grid to `output_path`. Returns the path written."""
    grid = build_representation_grid(representations, panel_size)
    ok = cv2.imwrite(output_path, grid)
    if not ok:
        raise IOError(f"cv2.imwrite failed to write grid to {output_path}")
    logger.info("Saved representation grid to %s", output_path)
    return output_path
