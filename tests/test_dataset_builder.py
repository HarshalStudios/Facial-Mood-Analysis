from __future__ import annotations

import csv

import numpy as np
import pytest

from app.core.config import Settings
from app.dataset.builder import CSV_FIELDNAMES, FeatureDatasetBuilder
from app.dataset.failure_log import FailureLogger
from app.emotion.mock_provider import FixedEmotionModel
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.features.extractor import StandardFeatureExtractor
from app.representations.engine import RepresentationEngine
from app.schemas.dataset import FER2013Sample
from app.schemas.feature_vector import FEATURE_NAMES


def _synthetic_gray_image(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    return image


def _sample(sample_id: str, label: str = "happy", image: np.ndarray | None = None) -> FER2013Sample:
    return FER2013Sample(
        sample_id=sample_id,
        image=image if image is not None else _synthetic_gray_image(int(sample_id) + 1),
        label=label,
        label_id=CANONICAL_EMOTION_LABELS.index(label),
        raw_label=label,
        raw_official_split=None,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def builder(settings) -> FeatureDatasetBuilder:
    engine = RepresentationEngine(settings)
    extractor = StandardFeatureExtractor(FixedEmotionModel(), settings)
    return FeatureDatasetBuilder(engine, extractor, settings)


def test_builds_successful_rows_with_correct_schema(tmp_path, builder, settings):
    samples = [_sample(str(i)) for i in range(5)]
    output_path = tmp_path / "train_features.csv"

    with FailureLogger(tmp_path / "failures.csv") as fl:
        result = builder.build_split(samples, "train", output_path, fl)

    assert result.successful == 5
    assert result.failed == 0

    with open(output_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 5
    assert list(rows[0].keys()) == CSV_FIELDNAMES
    feature_cols = [c for c in CSV_FIELDNAMES if c in FEATURE_NAMES]
    assert len(feature_cols) == 22
    for row in rows:
        for name in FEATURE_NAMES:
            assert np.isfinite(float(row[name]))


def test_resumability_skips_already_completed_samples(tmp_path, builder, settings):
    samples = [_sample(str(i)) for i in range(4)]
    output_path = tmp_path / "train_features.csv"

    with FailureLogger(tmp_path / "failures.csv") as fl:
        first = builder.build_split(samples, "train", output_path, fl)
    assert first.successful == 4

    with FailureLogger(tmp_path / "failures.csv") as fl:
        second = builder.build_split(samples, "train", output_path, fl)

    assert second.already_completed == 4
    assert second.processed == 0
    assert second.successful == 0

    with open(output_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4  # not duplicated


def test_failed_sample_is_logged_and_excluded_from_csv(tmp_path, builder, settings):
    good = _sample("0")
    # A degenerate image (wrong ndim) fails at the preprocessing stage.
    bad_image = np.zeros((4, 4, 4, 4), dtype=np.uint8)
    bad = _sample("1", image=bad_image)
    samples = [good, bad]
    output_path = tmp_path / "train_features.csv"
    failures_path = tmp_path / "failures.csv"

    with FailureLogger(failures_path) as fl:
        result = builder.build_split(samples, "train", output_path, fl)

    assert result.successful == 1
    assert result.failed == 1

    with open(output_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "0"

    with open(failures_path, newline="") as f:
        failure_rows = list(csv.DictReader(f))
    assert len(failure_rows) == 1
    assert failure_rows[0]["sample_id"] == "1"
    assert failure_rows[0]["stage"] == "preprocessing"


def test_checkpoint_flush_does_not_lose_rows(tmp_path, settings):
    settings = settings.model_copy(update={"dataset_checkpoint_batch_size": 2})
    engine = RepresentationEngine(settings)
    extractor = StandardFeatureExtractor(FixedEmotionModel(), settings)
    builder = FeatureDatasetBuilder(engine, extractor, settings)

    samples = [_sample(str(i)) for i in range(7)]
    output_path = tmp_path / "train_features.csv"

    with FailureLogger(tmp_path / "failures.csv") as fl:
        result = builder.build_split(samples, "train", output_path, fl)

    assert result.successful == 7
    with open(output_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7
