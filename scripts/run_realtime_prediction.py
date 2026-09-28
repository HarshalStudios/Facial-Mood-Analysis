"""
Phase 7 entrypoint: webcam -> RealTimePredictor -> on-screen result.

    python scripts/run_realtime_prediction.py
        [--show-representations] [--benchmark N] [--no-display]

Controls (interactive mode):
    q       quit

Modes:
    (default)               Opens the webcam, runs the full Phase 7
                             pipeline frame-by-frame, and overlays the
                             face bounding box, predicted emotion,
                             confidence, FPS, and latency on the video
                             feed (spec sec. 25). Development
                             visualization only -- not the production
                             frontend.
    --show-representations   Also opens the Phase 3 representation grid
                              (RGB/Grayscale/CLAHE/Canny/Sobel/LBP) for
                              the current primary face each frame (spec
                              sec. 26).
    --benchmark N             Headless: reads N frames (no display, no
                               window), measures actual per-frame and
                               per-stage latency, prints a summary, and
                               writes the measured results to
                               experiments/phase7_realtime_benchmark.json
                               (spec sec. 27/37). Requires a real
                               webcam; does not fabricate values -- if
                               fewer than N frames could be read, the
                               summary is computed over however many
                               were actually captured.

This script is a thin CLI wrapper around `RealTimePredictor` -- all
pipeline logic lives in `app.prediction.predictor`, not here (spec
sec. 24).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.camera.opencv_camera import OpenCVCameraSource  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import BackendError  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.prediction.predictor import RealTimePredictor  # noqa: E402
from app.representations.engine import RepresentationEngine  # noqa: E402
from app.representations.visualization import build_representation_grid  # noqa: E402

logger = get_logger(__name__)

WINDOW_NAME = "Phase 7 - Real-Time Prediction (q=quit)"
REPR_WINDOW_NAME = "Phase 7 - Representations (debug)"
BENCHMARK_OUTPUT_PATH = Path("experiments/phase7_realtime_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-representations",
        action="store_true",
        help="Also display the live Phase 3 representation grid for the primary face.",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=None,
        metavar="N",
        help="Headless mode: capture N frames, measure actual latency/FPS, write a benchmark JSON.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run the interactive loop without opening any OpenCV window (still reads keyboard is disabled; use --benchmark to quit automatically instead).",
    )
    return parser.parse_args()


def _draw_overlay(image, result, fps: float) -> None:
    """Draw bbox / emotion / confidence / FPS / latency onto `image` in place (spec sec. 25)."""
    if result.primary_face_bbox is not None:
        b = result.primary_face_bbox
        cv2.rectangle(image, (b.x, b.y), (b.x + b.width, b.y + b.height), (0, 200, 0), 2)

    lines = []
    if result.prediction_available:
        lines.append(f"Emotion: {result.predicted_emotion.capitalize()}")
        lines.append(f"Confidence: {result.confidence * 100:.1f}%")
    elif result.face_detected:
        lines.append(f"Prediction unavailable ({result.error_code})")
    else:
        lines.append("No face detected")

    lines.append(f"FPS: {fps:.1f}")
    if result.processing_time_ms is not None:
        lines.append(f"Latency: {result.processing_time_ms:.0f} ms")

    for i, line in enumerate(lines):
        y = 25 + i * 24
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def run_interactive(predictor: RealTimePredictor, show_representations: bool, no_display: bool) -> None:
    settings = get_settings()
    representation_engine = RepresentationEngine(settings) if show_representations else None

    frame_times: list[float] = []

    with OpenCVCameraSource(settings) as camera:
        print("Webcam opened. Press 'q' to quit.")
        while True:
            loop_start = time.perf_counter()
            frame = camera.read()
            if frame is None:
                logger.warning("Failed to read frame; stopping.")
                break

            result = predictor.predict(frame)

            frame_times.append(time.perf_counter() - loop_start)
            recent = frame_times[-30:]
            fps = len(recent) / sum(recent) if sum(recent) > 0 else 0.0

            if not no_display:
                display = frame.image.copy()
                _draw_overlay(display, result, fps)
                cv2.imshow(WINDOW_NAME, display)

                if show_representations and result.primary_face_bbox is not None:
                    try:
                        b = result.primary_face_bbox
                        crop = frame.image[b.y : b.y + b.height, b.x : b.x + b.width]
                        representations = representation_engine.generate(crop)
                        grid = build_representation_grid(representations)
                        cv2.imshow(REPR_WINDOW_NAME, grid)
                    except BackendError as exc:
                        logger.debug("Debug representation grid failed: %s", exc.message)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                # No window/keyboard available; caller must Ctrl+C or use --benchmark instead.
                pass


def run_benchmark(predictor: RealTimePredictor, n_frames: int) -> None:
    settings = get_settings()

    per_frame_total_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {
        "face_detection_ms": [],
        "representation_ms": [],
        "feature_extraction_ms": [],
        "model_inference_ms": [],
    }
    frames_read = 0
    faces_detected = 0
    predictions_available = 0

    print(f"Benchmarking: attempting to capture {n_frames} frames...")
    with OpenCVCameraSource(settings) as camera:
        for _ in range(n_frames):
            frame = camera.read()
            if frame is None:
                logger.warning("Camera returned no frame during benchmark; stopping early.")
                break
            frames_read += 1

            result = predictor.predict(frame)

            if result.processing_time_ms is not None:
                per_frame_total_ms.append(result.processing_time_ms)
            for key in stage_ms:
                value = getattr(result.timing, key)
                if value is not None:
                    stage_ms[key].append(value)
            if result.face_detected:
                faces_detected += 1
            if result.prediction_available:
                predictions_available += 1

    if frames_read == 0 or not per_frame_total_ms:
        print("Not measured: no frames were successfully captured/processed.")
        return

    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return float("nan")
        return statistics.quantiles(values, n=100)[int(p) - 1] if len(values) > 1 else values[0]

    avg_latency = statistics.mean(per_frame_total_ms)
    observed_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    summary = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "frames_requested": n_frames,
        "frames_read": frames_read,
        "faces_detected": faces_detected,
        "predictions_available": predictions_available,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": _percentile(per_frame_total_ms, 50),
        "p95_latency_ms": _percentile(per_frame_total_ms, 95),
        "observed_fps": observed_fps,
        "stage_avg_ms": {
            key: (statistics.mean(values) if values else None) for key, values in stage_ms.items()
        },
        "model_version": predictor._model_version,  # noqa: SLF001 - benchmark script, internal use
        "feature_schema_version": predictor._feature_schema_version,  # noqa: SLF001
        "note": "All values measured on this machine during this run; not a claimed/guaranteed rate.",
    }

    BENCHMARK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nBenchmark written to {BENCHMARK_OUTPUT_PATH}")


def main() -> int:
    setup_logging()
    args = parse_args()

    try:
        predictor = RealTimePredictor()
    except BackendError as exc:
        logger.error("Failed to initialize RealTimePredictor: %s", exc.message)
        print(f"FATAL: {exc.message}")
        return 1

    try:
        if args.benchmark is not None:
            run_benchmark(predictor, args.benchmark)
        else:
            run_interactive(predictor, args.show_representations, args.no_display)
    except BackendError as exc:
        logger.error("Fatal error during real-time run: %s", exc.message)
        print(f"FATAL: {exc.message}")
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
