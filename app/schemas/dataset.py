"""
Data contracts for FER2013 loading and feature-dataset generation (Phase 5).

Mirrors the style of app.schemas.face / app.schemas.representations: plain
dataclasses that describe what moves between app.dataset components,
independent of how each component is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Canonical split names used everywhere downstream of the loader (the
# dataset builder, the CSV outputs, the validation report). Distinct from
# the *raw* Usage/split values a dataset file might use (see
# app.dataset.splitter.OFFICIAL_SPLIT_ALIASES for that mapping).
SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")


@dataclass
class DatasetSchemaInfo:
    """What FER2013Loader discovered about the supplied dataset file."""

    source_path: str
    format: str  # "csv" | "image_folder"
    columns: list[str] | None
    width: int
    height: int
    channels: int
    dtype: str
    pixel_min: float
    pixel_max: float
    has_official_split_column: bool
    total_rows: int


@dataclass
class MalformedRecord:
    """One dataset row/file the loader could not turn into a valid sample."""

    row_index: str
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class FER2013Sample:
    """One validated FER2013 record, prior to any preprocessing/representation work."""

    sample_id: str
    image: np.ndarray  # as loaded: typically (H, W) uint8 grayscale
    label: str  # canonical emotion name (app.emotion.validation.CANONICAL_EMOTION_LABELS)
    label_id: int  # index into CANONICAL_EMOTION_LABELS
    raw_label: str  # original label value from the source file, stringified
    raw_official_split: str | None  # original Usage/split value, if the file had one


@dataclass
class LoadedDataset:
    """Full result of FER2013Loader.load()."""

    samples: list[FER2013Sample]
    malformed: list[MalformedRecord]
    schema: DatasetSchemaInfo


@dataclass
class FailureRecord:
    """One sample that failed during representation/feature-extraction (Phase 5 spec sec. 25)."""

    sample_id: str
    split: str
    stage: str  # "preprocessing" | "representation" | "feature_extraction"
    error_type: str
    error_message: str


@dataclass
class SplitBuildResult:
    """Outcome of building the feature CSV for one split."""

    split: str
    output_path: str
    total_candidates: int
    already_completed: int  # skipped because resumability found them already written
    processed: int
    successful: int
    failed: int


@dataclass
class ProcessingReport:
    """Aggregate outcome of a full Phase 5 build run, across all splits."""

    dataset_name: str
    split_results: dict[str, SplitBuildResult]
    failures_path: str
    malformed_load_count: int

    @property
    def total_successful(self) -> int:
        return sum(r.successful for r in self.split_results.values())

    @property
    def total_failed(self) -> int:
        return sum(r.failed for r in self.split_results.values())

    @property
    def total_processed(self) -> int:
        return sum(r.processed for r in self.split_results.values())
