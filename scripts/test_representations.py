"""
Manual/integration check for Phase 3: webcam -> Phase 2 face detection
-> Phase 3 RepresentationEngine -> live display of all six
representations.

This is NOT a pytest test (it needs a real webcam and a display) and is
not run automatically. Run it directly:

    python scripts/test_representations.py

Controls:
    q       quit
    s       save the current representation grid + individual PNGs
            to outputs/representations/ (see app/representations/export.py)

What to look for:
    - RGB / Grayscale / CLAHE should look like recognizable variants
      of your face.
    - Canny should show clear edges around your face's contours.
    - Sobel should highlight directional intensity transitions.
    - LBP should look noisy/speckled — that's expected, it's a texture
      code map, not a photograph.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.camera.opencv_camera import OpenCVCameraSource  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import BackendError  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.detection.opencv_detector import HaarCascadeFaceDetector  # noqa: E402
from app.detection.service import FaceDetectionService  # noqa: E402
from app.representations.engine import RepresentationEngine  # noqa: E402
from app.representations.export import export_representations  # noqa: E402
from app.representations.visualization import build_representation_grid  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    detector = HaarCascadeFaceDetector(settings)
    detection_service = FaceDetectionService(detector, settings)
    representation_engine = RepresentationEngine(settings)

    window_name = "Phase 3 - Representations (q=quit, s=save)"

    with OpenCVCameraSource(settings) as camera:
        print("Webcam opened. Press 'q' to quit, 's' to save the current grid.")
        while True:
            frame = camera.read()
            if frame is None:
                logger.warning("Failed to read frame; stopping.")
                break

            result = detection_service.process(frame)
            primary = next((f for f in result.faces if f.is_primary), None)

            if primary is None:
                cv2.imshow(window_name, frame.image)
            else:
                try:
                    representations = representation_engine.generate(primary.crop)
                except BackendError as exc:
                    logger.error("Representation generation failed: %s", exc.message)
                    cv2.imshow(window_name, frame.image)
                else:
                    grid = build_representation_grid(representations)
                    cv2.imshow(window_name, grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and primary is not None:
                out_dir = settings.representation_debug_dir
                written = export_representations(representations, out_dir)
                cv2.imwrite(str(Path(out_dir) / "grid.png"), grid)
                print(f"Saved {len(written)} representation(s) + grid to {out_dir}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
