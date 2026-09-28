from __future__ import annotations

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import DatasetError
from app.dataset.splitter import assign_splits
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.schemas.dataset import FER2013Sample


def _sample(sample_id: str, label: str, official_split: str | None) -> FER2013Sample:
    return FER2013Sample(
        sample_id=sample_id,
        image=np.zeros((4, 4), dtype=np.uint8),
        label=label,
        label_id=CANONICAL_EMOTION_LABELS.index(label),
        raw_label=label,
        raw_official_split=official_split,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def test_official_split_is_preserved(settings):
    samples = [
        _sample("1", "happy", "Training"),
        _sample("2", "sad", "PublicTest"),
        _sample("3", "angry", "PrivateTest"),
    ]
    assignment, info = assign_splits(samples, settings)

    assert assignment == {"1": "train", "2": "validation", "3": "test"}
    assert info.strategy == "official"


def test_unrecognized_official_split_value_raises(settings):
    samples = [_sample("1", "happy", "SomeUnknownSplit")]
    with pytest.raises(DatasetError):
        assign_splits(samples, settings)


def test_mixed_official_and_missing_split_raises(settings):
    samples = [_sample("1", "happy", "Training"), _sample("2", "sad", None)]
    with pytest.raises(DatasetError):
        assign_splits(samples, settings)


def test_empty_samples_raises(settings):
    with pytest.raises(DatasetError):
        assign_splits([], settings)


def _make_balanced_samples(per_class: int = 20) -> list[FER2013Sample]:
    samples = []
    counter = 0
    for label in CANONICAL_EMOTION_LABELS:
        for _ in range(per_class):
            samples.append(_sample(str(counter), label, None))
            counter += 1
    return samples


def test_stratified_split_has_no_overlap_and_correct_total(settings):
    samples = _make_balanced_samples(per_class=20)
    assignment, info = assign_splits(samples, settings)

    assert info.strategy == "deterministic_stratified"
    assert len(assignment) == len(samples)

    train_ids = {sid for sid, split in assignment.items() if split == "train"}
    val_ids = {sid for sid, split in assignment.items() if split == "validation"}
    test_ids = {sid for sid, split in assignment.items() if split == "test"}

    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()
    assert train_ids | val_ids | test_ids == set(assignment.keys())


def test_stratified_split_is_deterministic_for_same_seed(settings):
    samples = _make_balanced_samples(per_class=20)
    assignment_a, _ = assign_splits(samples, settings)
    assignment_b, _ = assign_splits(samples, settings)
    assert assignment_a == assignment_b


def test_bad_ratios_raise(settings):
    bad_settings = settings.model_copy(
        update={"dataset_train_ratio": 0.8, "dataset_val_ratio": 0.3, "dataset_test_ratio": 0.1}
    )
    samples = _make_balanced_samples(per_class=5)
    with pytest.raises(DatasetError):
        assign_splits(samples, bad_settings)
