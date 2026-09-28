from __future__ import annotations

import pandas as pd
import pytest

from app.dataset.builder import CSV_FIELDNAMES
from app.dataset.validation import validate_feature_dataset
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.schemas.feature_vector import FEATURE_NAMES


def _row(sample_id: str, split: str, emotion: str = "happy") -> dict:
    row = {
        "sample_id": sample_id,
        "split": split,
        "emotion": emotion,
        "emotion_id": CANONICAL_EMOTION_LABELS.index(emotion),
    }
    for i, name in enumerate(FEATURE_NAMES):
        row[name] = 0.1 * (i + 1)
    return row


def _write(tmp_path, name: str, rows: list[dict]):
    df = pd.DataFrame(rows, columns=CSV_FIELDNAMES)
    path = tmp_path / f"{name}_features.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def three_valid_splits(tmp_path):
    train = _write(tmp_path, "train", [_row("1", "train"), _row("2", "train", "sad")])
    val = _write(tmp_path, "validation", [_row("3", "validation")])
    test = _write(tmp_path, "test", [_row("4", "test")])
    return {"train": train, "validation": val, "test": test}


def test_valid_dataset_passes_all_checks(three_valid_splits):
    report = validate_feature_dataset(three_valid_splits, "FER2013")
    assert report.is_valid
    assert report.split_counts == {"train": 2, "validation": 1, "test": 1}


def test_nan_value_fails_check(tmp_path):
    rows = [_row("1", "train")]
    rows[0][FEATURE_NAMES[0]] = float("nan")
    train = _write(tmp_path, "train", rows)
    val = _write(tmp_path, "validation", [_row("2", "validation")])
    test = _write(tmp_path, "test", [_row("3", "test")])

    report = validate_feature_dataset({"train": train, "validation": val, "test": test}, "FER2013")

    assert not report.checks["no_nan"]
    assert not report.is_valid


def test_overlapping_sample_ids_fails_check(tmp_path):
    train = _write(tmp_path, "train", [_row("shared", "train")])
    val = _write(tmp_path, "validation", [_row("shared", "validation")])
    test = _write(tmp_path, "test", [_row("3", "test")])

    report = validate_feature_dataset({"train": train, "validation": val, "test": test}, "FER2013")

    assert not report.checks["no_cross_split_overlap"]
    assert not report.is_valid


def test_missing_feature_column_fails_schema_check(tmp_path):
    row = _row("1", "train")
    del row[FEATURE_NAMES[-1]]
    df = pd.DataFrame([row])
    train = tmp_path / "train_features.csv"
    df.to_csv(train, index=False)
    val = _write(tmp_path, "validation", [_row("2", "validation")])
    test = _write(tmp_path, "test", [_row("3", "test")])

    report = validate_feature_dataset({"train": train, "validation": val, "test": test}, "FER2013")

    assert not report.checks["schema_matches_contract"]


def test_emotion_id_mismatch_fails_check(tmp_path):
    rows = [_row("1", "train")]
    rows[0]["emotion_id"] = 99
    train = _write(tmp_path, "train", rows)
    val = _write(tmp_path, "validation", [_row("2", "validation")])
    test = _write(tmp_path, "test", [_row("3", "test")])

    report = validate_feature_dataset({"train": train, "validation": val, "test": test}, "FER2013")

    assert not report.checks["valid_emotion_ids"]


def test_class_distribution_counts(three_valid_splits):
    report = validate_feature_dataset(three_valid_splits, "FER2013")
    assert report.class_distribution["train"] == {"happy": 1, "sad": 1}


def test_missing_split_file_fails(tmp_path):
    train = _write(tmp_path, "train", [_row("1", "train")])
    report = validate_feature_dataset(
        {"train": train, "validation": tmp_path / "nope.csv"}, "FER2013"
    )
    assert not report.is_valid
