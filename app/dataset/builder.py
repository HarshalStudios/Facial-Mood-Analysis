"""
FeatureDatasetBuilder (Phase 5 spec sec. 12-14, 25-29).

Orchestrates, per split:

    FER2013Sample.image
        -> app.dataset.preprocessing.to_representation_input   (this phase only)
        -> RepresentationEngine.generate()                     (Phase 3, reused)
        -> StandardFeatureExtractor.extract()                  (Phase 4, reused)
        -> 22D feature row + label -> append to the split's CSV

It does not implement any representation or feature-extraction math
itself (Phase 5 spec sec. 12-13) -- only the batch loop, failure
handling, resumability, checkpointed writes, and progress reporting
around the existing Phase 3/4 components.
"""

from __future__ import annotations

import csv
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import BackendError
from app.core.logging import get_logger
from app.dataset.failure_log import FailureLogger
from app.dataset.preprocessing import to_representation_input
from app.features.base import FeatureExtractor
from app.representations.base import RepresentationGenerator
from app.schemas.dataset import FailureRecord, FER2013Sample, SplitBuildResult
from app.schemas.feature_vector import FEATURE_NAMES

logger = get_logger(__name__)

CSV_FIELDNAMES: list[str] = ["sample_id", "split", "emotion", "emotion_id", *FEATURE_NAMES]

assert len(CSV_FIELDNAMES) == 4 + 22


class FeatureDatasetBuilder:
    """Builds one feature-dataset CSV per split from FER2013Sample records."""

    def __init__(
        self,
        representation_engine: RepresentationGenerator,
        feature_extractor: FeatureExtractor,
        settings: Settings,
    ) -> None:
        self._engine = representation_engine
        self._extractor = feature_extractor
        self._settings = settings

    def build_split(
        self,
        samples: list[FER2013Sample],
        split_name: str,
        output_path: str | Path,
        failure_logger: FailureLogger,
    ) -> SplitBuildResult:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        already_completed = self._read_completed_sample_ids(output_path)
        pending = [s for s in samples if s.sample_id not in already_completed]

        if already_completed:
            logger.info(
                "[%s] Resuming: %d sample(s) already present in %s, %d remaining",
                split_name,
                len(already_completed),
                output_path,
                len(pending),
            )

        write_header = not output_path.exists() or output_path.stat().st_size == 0
        batch_size = self._settings.dataset_checkpoint_batch_size
        progress_every = self._settings.dataset_progress_log_every

        successful = 0
        failed = 0

        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()

            rows_since_flush = 0
            for i, sample in enumerate(pending, start=1):
                row = self._process_one(sample, split_name, failure_logger)
                if row is None:
                    failed += 1
                else:
                    writer.writerow(row)
                    successful += 1
                    rows_since_flush += 1

                if rows_since_flush >= batch_size:
                    f.flush()
                    rows_since_flush = 0

                if i % progress_every == 0 or i == len(pending):
                    logger.info(
                        "[%s] Processing: %d / %d | Successful: %d | Failed: %d",
                        split_name,
                        i,
                        len(pending),
                        successful,
                        failed,
                    )

            f.flush()

        return SplitBuildResult(
            split=split_name,
            output_path=str(output_path),
            total_candidates=len(samples),
            already_completed=len(already_completed),
            processed=len(pending),
            successful=successful,
            failed=failed,
        )

    # ------------------------------------------------------------------

    def _process_one(
        self, sample: FER2013Sample, split_name: str, failure_logger: FailureLogger
    ) -> dict | None:
        stage = "preprocessing"
        try:
            bgr_image = to_representation_input(sample.image)

            stage = "representation"
            representations = self._engine.generate(bgr_image)

            stage = "feature_extraction"
            result = self._extractor.extract(representations)
        except BackendError as exc:
            failure_logger.log(
                FailureRecord(
                    sample_id=sample.sample_id,
                    split=split_name,
                    stage=stage,
                    error_type=type(exc).__name__,
                    error_message=exc.message if hasattr(exc, "message") else str(exc),
                )
            )
            return None

        row: dict = {
            "sample_id": sample.sample_id,
            "split": split_name,
            "emotion": sample.label,
            "emotion_id": sample.label_id,
        }
        row.update(result.features)
        return row

    @staticmethod
    def _read_completed_sample_ids(output_path: Path) -> set[str]:
        if not output_path.exists() or output_path.stat().st_size == 0:
            return set()
        completed: set[str] = set()
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("sample_id")
                if sid is not None:
                    completed.add(sid)
        return completed
