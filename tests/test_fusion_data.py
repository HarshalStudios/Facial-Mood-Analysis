from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import DatasetError
from app.dataset.builder import CSV_FIELDNAMES
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.fusion.data import (
    assert_all_classes_present_in_training,
    check_splits_exist,
    load_split,
    validate_datasets,
)
from app.schemas.feature_vector import FEATURE_NAMES


def _row(sample_id: str, emotion: str, feature_values: list[float] | None = None) -> dict:
    row = {
        "sample_id": sample_id,
        "split": "train",
        "emotion": emotion,
        "emotion_id": CANONICAL_EMOTION_LABELS.index(emotion),
    }
    values = feature_values or [0.1 * (i + 1) for i in range(len(FEATURE_NAMES))]
    row.update(dict(zip(FEATURE_NAMES, values)))
    return row


def _write_csv(tmp_path, name: str, rows: list[dict]):
    df = pd.DataFrame(rows, columns=CSV_FIELDNAMES)
    path = tmp_path / f"{name}_features.csv"
    df.to_csv(path, index=False)
    return path


def test_load_split_preserves_canonical_feature_order_even_if_csv_columns_are_shuffled(tmp_path):
    row = _row("1", "happy")
    df = pd.DataFrame([row])
    shuffled_columns = list(df.columns)[::-1]  # deliberately reverse column order on disk
    path = tmp_path / "train_features.csv"
    df[shuffled_columns].to_csv(path, index=False)

    split = load_split(path, "train")

    assert split.X.shape == (1, 22)
    expected = np.array([[row[name] for name in FEATURE_NAMES]])
    assert np.allclose(split.X, expected)


def test_load_split_maps_emotion_to_canonical_id(tmp_path):
    rows = [_row("1", "happy"), _row("2", "sad"), _row("3", "angry")]
    path = _write_csv(tmp_path, "train", rows)

    split = load_split(path, "train")

    assert split.y.tolist() == [
        CANONICAL_EMOTION_LABELS.index("happy"),
        CANONICAL_EMOTION_LABELS.index("sad"),
        CANONICAL_EMOTION_LABELS.index("angry"),
    ]


def test_load_split_rejects_missing_feature_column(tmp_path):
    row = _row("1", "happy")
    del row[FEATURE_NAMES[-1]]
    df = pd.DataFrame([row])
    path = tmp_path / "train_features.csv"
    df.to_csv(path, index=False)

    with pytest.raises(DatasetError):
        load_split(path, "train")


def test_load_split_rejects_unknown_label(tmp_path):
    row = _row("1", "happy")
    row["emotion"] = "confused"  # not one of the 7 canonical labels
    path = tmp_path / "train_features.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(DatasetError):
        load_split(path, "train")


def test_load_split_rejects_nan_feature_values(tmp_path):
    row = _row("1", "happy")
    row[FEATURE_NAMES[0]] = float("nan")
    path = tmp_path / "train_features.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(DatasetError):
        load_split(path, "train")


def test_check_splits_exist_reports_missing_paths(tmp_path):
    present = tmp_path / "train_features.csv"
    present.write_text("x")
    missing = tmp_path / "validation_features.csv"

    result = check_splits_exist({"train": present, "validation": missing})

    assert result == [str(missing)]


def test_assert_all_classes_present_raises_when_a_class_is_missing(tmp_path):
    # Only 2 of 7 classes present in the training split.
    rows = [_row(str(i), "happy") for i in range(3)] + [_row(str(i), "sad") for i in range(3, 6)]
    path = _write_csv(tmp_path, "train", rows)
    split = load_split(path, "train")

    with pytest.raises(DatasetError):
        assert_all_classes_present_in_training(split)


def test_assert_all_classes_present_passes_when_all_7_present(tmp_path):
    rows = [_row(str(i), label) for i, label in enumerate(CANONICAL_EMOTION_LABELS)]
    path = _write_csv(tmp_path, "train", rows)
    split = load_split(path, "train")

    assert_all_classes_present_in_training(split)  # should not raise


def test_validate_datasets_raises_on_structurally_invalid_data(tmp_path):
    rows = [_row("shared", "happy")]
    train = _write_csv(tmp_path, "train", rows)
    validation = _write_csv(tmp_path, "validation", rows)  # same sample_id -> overlap
    test = _write_csv(tmp_path, "test", [_row("3", "sad")])

    with pytest.raises(DatasetError):
        validate_datasets({"train": train, "validation": validation, "test": test}, "FER2013")
