"""
Phase 5 entrypoint: FER2013 -> {train,validation,test}_features.csv.

    python scripts/build_fer2013_features.py [--dataset-path PATH]
        [--emotion-backend deepface|mock] [--limit N] [--skip-validation]

Does NOT train a model, fit a scaler, or do anything Phase 6+ -- see
README "Phase 5" section and app/dataset/*.py module docstrings for the
full contract. If `--dataset-path` doesn't exist, this exits with a clear
error instead of fabricating or substituting data (Phase 5 spec sec. 5).

`--emotion-backend mock` exists for smoke-testing the pipeline wiring
without DeepFace/network access (mirrors Phase 4's own DeepFace-caveat --
see app/emotion/deepface_provider.py). Real feature-dataset generation
for the project report must use the real `deepface` backend (the
default, taken from FMA_EMOTION_MODEL_BACKEND / Settings).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.exceptions import BackendError  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.dataset.builder import CSV_FIELDNAMES, FeatureDatasetBuilder  # noqa: E402
from app.dataset.failure_log import FailureLogger  # noqa: E402
from app.dataset.fer2013_loader import FER2013Loader  # noqa: E402
from app.dataset.metadata import (  # noqa: E402
    build_dataset_metadata,
    build_feature_schema_metadata,
    write_json,
)
from app.dataset.splitter import assign_splits  # noqa: E402
from app.dataset.validation import validate_feature_dataset  # noqa: E402
from app.emotion.factory import get_emotion_model  # noqa: E402
from app.emotion.mock_provider import FixedEmotionModel  # noqa: E402
from app.features.extractor import StandardFeatureExtractor  # noqa: E402
from app.representations.engine import RepresentationEngine  # noqa: E402
from app.schemas.dataset import SPLIT_NAMES, ProcessingReport, SplitBuildResult  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=str, default=None, help="Override FMA_DATASET_PATH")
    parser.add_argument(
        "--emotion-backend",
        choices=["deepface", "fer2013_cnn", "fer2013_linear", "mock"],
        default=None,
        help="Override FMA_EMOTION_MODEL_BACKEND (use 'mock' only for pipeline smoke tests)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N loaded samples (debugging)"
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip the post-build validation pass"
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    settings = get_settings()

    dataset_path = args.dataset_path or settings.dataset_path

    logger.info("=== Phase 5: FER2013 feature-dataset generation ===")
    logger.info("Dataset path: %s", dataset_path)

    loader = FER2013Loader()
    try:
        loaded = loader.load(dataset_path)
    except BackendError as exc:
        logger.error("Dataset load failed: %s | %s", exc.message, exc.details)
        print(f"\nERROR: {exc.message}\nDetails: {exc.details}\n", file=sys.stderr)
        return 1

    logger.info(
        "Schema: %dx%d, %d channel(s), %s, official_split_column=%s, total_rows=%d",
        loaded.schema.width,
        loaded.schema.height,
        loaded.schema.channels,
        loaded.schema.dtype,
        loaded.schema.has_official_split_column,
        loaded.schema.total_rows,
    )
    logger.info("Valid samples: %d | Malformed rows: %d", len(loaded.samples), len(loaded.malformed))

    samples = loaded.samples
    if args.limit is not None:
        samples = samples[: args.limit]
        logger.info("--limit applied: processing %d sample(s)", len(samples))

    try:
        split_assignment, split_info = assign_splits(samples, settings)
    except BackendError as exc:
        logger.error("Split assignment failed: %s | %s", exc.message, exc.details)
        print(f"\nERROR: {exc.message}\nDetails: {exc.details}\n", file=sys.stderr)
        return 1

    backend = args.emotion_backend or settings.emotion_model_backend
    if backend == "mock":
        logger.warning(
            "Using the MOCK emotion provider -- this proves the pipeline wiring only. "
            "Feature values from the emotion_* columns are NOT meaningful and this "
            "dataset must not be used to train the Phase 6 fusion model."
        )
        emotion_model = FixedEmotionModel()
        effective_settings = settings.model_copy(update={"emotion_model_backend": "mock"})
    else:
        emotion_model = get_emotion_model(settings)
        effective_settings = settings

    engine = RepresentationEngine(settings)
    extractor = StandardFeatureExtractor(emotion_model, settings)
    builder = FeatureDatasetBuilder(engine, extractor, settings)

    features_raw_dir = settings.features_dir / "raw"
    features_meta_dir = settings.features_dir / "metadata"
    features_raw_dir.mkdir(parents=True, exist_ok=True)
    features_meta_dir.mkdir(parents=True, exist_ok=True)

    samples_by_split: dict[str, list] = {name: [] for name in SPLIT_NAMES}
    for sample in samples:
        split = split_assignment.get(sample.sample_id)
        if split is not None:
            samples_by_split[split].append(sample)

    split_results: dict[str, SplitBuildResult] = {}
    start = time.monotonic()
    with FailureLogger(settings.dataset_failures_path) as failure_logger:
        for split_name in SPLIT_NAMES:
            split_samples = samples_by_split[split_name]
            output_path = features_raw_dir / f"{split_name}_features.csv"
            logger.info("--- Building split '%s' (%d candidate sample(s)) ---", split_name, len(split_samples))
            result = builder.build_split(split_samples, split_name, output_path, failure_logger)
            split_results[split_name] = result
            logger.info(
                "[%s] done: processed=%d successful=%d failed=%d (already_completed=%d)",
                split_name,
                result.processed,
                result.successful,
                result.failed,
                result.already_completed,
            )
    elapsed = time.monotonic() - start

    report = ProcessingReport(
        dataset_name=settings.dataset_name,
        split_results=split_results,
        failures_path=str(settings.dataset_failures_path),
        malformed_load_count=len(loaded.malformed),
    )

    dataset_metadata = build_dataset_metadata(
        settings=effective_settings,
        schema=loaded.schema,
        split_strategy=split_info.strategy,
        split_details=split_info.details,
    )
    write_json(dataset_metadata, features_meta_dir / "dataset_metadata.json")
    write_json(build_feature_schema_metadata(), features_meta_dir / "feature_schema.json")

    processing_report_json = {
        "dataset_name": report.dataset_name,
        "elapsed_seconds": round(elapsed, 2),
        "malformed_load_count": report.malformed_load_count,
        "splits": {
            name: {
                "output_path": r.output_path,
                "total_candidates": r.total_candidates,
                "already_completed": r.already_completed,
                "processed": r.processed,
                "successful": r.successful,
                "failed": r.failed,
            }
            for name, r in report.split_results.items()
        },
        "total_successful": report.total_successful,
        "total_failed": report.total_failed,
        "failures_path": report.failures_path,
    }
    write_json(processing_report_json, features_meta_dir / "processing_report.json")

    validation_report = None
    if not args.skip_validation:
        csv_paths = {name: features_raw_dir / f"{name}_features.csv" for name in SPLIT_NAMES}
        validation_report = validate_feature_dataset(csv_paths, settings.dataset_name)
        settings.dataset_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.dataset_summary_path, "w", encoding="utf-8") as f:
            f.write(validation_report.to_text())

    _print_final_report(report, loaded, split_info, validation_report, backend)
    return 0


def _print_final_report(report, loaded, split_info, validation_report, backend) -> None:
    print("\n" + "=" * 70)
    print("PHASE 5 REPORT")
    print("=" * 70)

    print("\nA. Dataset information")
    print(f"  total samples loaded : {len(loaded.samples)}")
    for name, r in report.split_results.items():
        print(f"  {name:<12}: {r.total_candidates} candidate(s) -> {r.successful} successful, {r.failed} failed")
    print(f"  image dimensions     : {loaded.schema.width}x{loaded.schema.height}")
    print(f"  split strategy       : {split_info.strategy}")

    print("\nB. Processing pipeline")
    print("  FER2013 sample -> to_representation_input -> RepresentationEngine.generate ->")
    print("  StandardFeatureExtractor.extract -> feature row (Phase 3/4 components reused unchanged)")
    print(f"  emotion provider backend: {backend}")

    print("\nC. Feature dataset")
    print(f"  columns ({len(CSV_FIELDNAMES)}): {CSV_FIELDNAMES}")

    print("\nD. Failure report")
    total_processed = report.total_processed
    failure_rate = (report.total_failed / total_processed * 100) if total_processed else 0.0
    print(f"  successful: {report.total_successful}  failed: {report.total_failed}  rate: {failure_rate:.2f}%")
    print(f"  failure log: {report.failures_path}")
    print(f"  malformed rows at load time: {report.malformed_load_count}")

    print("\nE. Leakage checks")
    if validation_report:
        print(f"  no_cross_split_overlap: {validation_report.checks.get('no_cross_split_overlap')}")
    else:
        print("  validation skipped (--skip-validation)")

    print("\nF. Tests")
    print("  unit tests: tests/test_fer2013_*.py, tests/test_dataset_*.py (pytest)")
    print("  integration test (real dataset): none run automatically -- see README Phase 5 section")

    print("\nG. Generated files")
    print(f"  {report.failures_path}")
    for r in report.split_results.values():
        print(f"  {r.output_path}")
    print("  features/metadata/dataset_metadata.json")
    print("  features/metadata/feature_schema.json")
    print("  features/metadata/processing_report.json")

    print("\nH. Phase 6 handoff")
    print("  Load features/raw/{train,validation,test}_features.csv with pandas.")
    print("  X = df[FEATURE_NAMES].to_numpy(); y = df['emotion'] (or df['emotion_id']).")
    print("  Fit StandardScaler on TRAIN only, then transform validation/test.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
