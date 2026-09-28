"""
Phase 8 entrypoint: webcam -> TemporalPredictionEngine -> stable on-screen result.

    python scripts/run_temporal_prediction.py
        [--benchmark N] [--no-display] [--reset-on-key]

This is the Phase 8 analogue of `scripts/run_realtime_prediction.py`
(Phase 7), which it does not modify or duplicate -- it only adds a
`TemporalPredictionEngine` on top, exactly as the Phase 8 spec
requires (sec 24 / 36-37). Do not re-run Phase 7's own script for
temporal/quality behavior; this script is the one to use for that.

Controls (interactive mode):
    q       quit
    r       manually reset temporal history (useful for demoing spec
            sec 33's resettable-history requirement without waiting
            for the no-face timeout)

Modes:
    (default)          Opens the webcam, runs Phase 7 + Phase 8, and
                        overlays raw emotion/confidence, stable
                        (smoothed) emotion/confidence, quality status,
                        history length, FPS, and latency (spec sec 36).
    --benchmark N       Headless: reads N frames (no display), measures
                        actual latency, prints/writes a summary. Never
                        fabricates values -- computed over however many
                        frames were actually captured (spec sec 30/54).
    --no-display        Run without an OpenCV window (use --benchmark
                        or Ctrl+C to stop).

This script is a thin CLI wrapper: all pipeline logic lives in
`app.temporal.engine.TemporalPredictionEngine` (spec sec 24), not here.
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
from app.temporal.engine import TemporalPredictionEngine  # noqa: E402

logger = get_logger(__name__)

WINDOW_NAME = "Phase 8 - Temporal + Quality (q=quit, r=reset)"
BENCHMARK_OUTPUT_PATH = Path("experiments/phase8_temporal_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=int,
        default=None,
        metavar="N",
        help="Headless mode: capture N frames, measure actual latency, write a benchmark JSON.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run the interactive loop without opening any OpenCV window.",
    )
    return parser.parse_args()


def _draw_overlay(image, result, fps: float) -> None:
    """Overlay raw / stable / quality / history / FPS / latency (spec sec 36-37)."""
    lines: list[str] = []

    if not result.face_detected:
        lines.append("No face detected")
    else:
        if result.raw_emotion is not None:
            lines.append(f"Raw: {result.raw_emotion.capitalize()} ({result.raw_confidence * 100:.0f}%)")
        else:
            lines.append(f"Raw: unavailable ({result.error_code})")

        if result.smoothed_emotion is not None:
            lines.append(
                f"Stable: {result.smoothed_emotion.capitalize()} ({result.smoothed_confidence * 100:.0f}%)"
            )
        else:
            lines.append("Stable: (no history yet)")

        if result.quality is not None:
            if result.quality.status == "good":
                lines.append("Quality: Good")
            else:
                reason = ", ".join(result.quality.warnings) or "warning"
                lines.append(f"Quality: {result.quality.status.capitalize()} ({reason})")

        lines.append(f"History: {result.history_length}")

    lines.append(f"FPS: {fps:.1f}")

    for i, line in enumerate(lines):
        y = 25 + i * 24
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def run_interactive(engine: TemporalPredictionEngine, no_display: bool) -> None:
    settings = get_settings()
    frame_times: list[float] = []

    with OpenCVCameraSource(settings) as camera:
        print("Webcam opened. Press 'q' to quit, 'r' to reset temporal history.")
        while True:
            loop_start = time.perf_counter()
            frame = camera.read()
            if frame is None:
                logger.warning("Failed to read frame; stopping.")
                break

            result = engine.process(frame)

            frame_times.append(time.perf_counter() - loop_start)
            recent = frame_times[-30:]
            fps = len(recent) / sum(recent) if sum(recent) > 0 else 0.0

            if not no_display:
                display = frame.image.copy()
                _draw_overlay(display, result, fps)
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    engine.reset()
                    print("Temporal history manually reset.")
            else:
                pass


def run_benchmark(engine: TemporalPredictionEngine, n_frames: int) -> None:
    settings = get_settings()

    per_frame_ms: list[float] = []
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

            t0 = time.perf_counter()
            result = engine.process(frame)
            per_frame_ms.append((time.perf_counter() - t0) * 1000.0)

            if result.face_detected:
                faces_detected += 1
            if result.prediction_available:
                predictions_available += 1

    if frames_read == 0 or not per_frame_ms:
        print("Not measured: no frames were successfully captured/processed.")
        return

    avg_latency = statistics.mean(per_frame_ms)
    stats = engine.stability_stats

    summary = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "frames_requested": n_frames,
        "frames_read": frames_read,
        "faces_detected": faces_detected,
        "predictions_available": predictions_available,
        "avg_total_latency_ms": avg_latency,  # includes Phase 7 predict() + Phase 8 post-processing
        "observed_fps": 1000.0 / avg_latency if avg_latency > 0 else 0.0,
        "raw_label_transitions": stats["raw_transitions"],
        "smoothed_label_transitions": stats["smoothed_transitions"],
        "smoothing_method": settings.smoothing_method,
        "smoothing_window_size": settings.smoothing_window_size,
        "quality_mode": settings.quality_mode,
        "note": (
            "All values measured on this machine during this run; not a "
            "claimed/guaranteed rate. Fewer transitions in the smoothed row "
            "than the raw row is NOT by itself evidence of higher accuracy "
            "-- see Phase 9 for quantitative evaluation."
        ),
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
        engine = TemporalPredictionEngine()
    except BackendError as exc:
        logger.error("Failed to initialize TemporalPredictionEngine: %s", exc.message)
        print(f"FATAL: {exc.message}")
        return 1

    try:
        if args.benchmark is not None:
            run_benchmark(engine, args.benchmark)
        else:
            run_interactive(engine, args.no_display)
    except BackendError as exc:
        logger.error("Fatal error during temporal run: %s", exc.message)
        print(f"FATAL: {exc.message}")
        return 1
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
