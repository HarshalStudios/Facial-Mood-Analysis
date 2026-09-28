"""
Tests use small, synthetic, clearly-fake CSVs built in tmp_path -- these
are NOT FER2013 and must never be mistaken for it. They only exercise
FER2013Loader's parsing/schema-detection logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import DatasetError
from app.dataset.fer2013_loader import FER2013Loader

SIDE = 4  # tiny synthetic image size, unrelated to FER2013's real 48x48


def _pixel_string(seed: int) -> str:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 255, size=SIDE * SIDE)
    return " ".join(str(v) for v in values)


def _write_csv(tmp_path, rows, columns=("emotion", "pixels", "Usage")) -> str:
    df = pd.DataFrame(rows, columns=list(columns))
    path = tmp_path / "synthetic_fer2013.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_loads_valid_rows(tmp_path):
    rows = [
        [0, _pixel_string(1), "Training"],
        [3, _pixel_string(2), "PublicTest"],
        [6, _pixel_string(3), "PrivateTest"],
    ]
    path = _write_csv(tmp_path, rows)

    loaded = FER2013Loader().load(path)

    assert len(loaded.samples) == 3
    assert loaded.malformed == []
    assert loaded.schema.width == SIDE
    assert loaded.schema.height == SIDE
    assert loaded.schema.has_official_split_column is True
    labels = {s.label for s in loaded.samples}
    assert labels == {"angry", "happy", "neutral"}


def test_malformed_pixel_row_is_reported_not_raised(tmp_path):
    rows = [
        [0, _pixel_string(1), "Training"],
        [1, "1 2 3", "Training"],  # wrong token count
    ]
    path = _write_csv(tmp_path, rows)

    loaded = FER2013Loader().load(path)

    assert len(loaded.samples) == 1
    assert len(loaded.malformed) == 1


def test_invalid_label_row_is_reported_not_raised(tmp_path):
    rows = [
        [0, _pixel_string(1), "Training"],
        [99, _pixel_string(2), "Training"],  # out-of-range label
    ]
    path = _write_csv(tmp_path, rows)

    loaded = FER2013Loader().load(path)

    assert len(loaded.samples) == 1
    assert len(loaded.malformed) == 1
    assert "label" in loaded.malformed[0].reason.lower() or "range" in loaded.malformed[0].reason.lower()


def test_alternate_column_names_detected(tmp_path):
    rows = [[3, _pixel_string(1), "train"]]
    path = _write_csv(tmp_path, rows, columns=("label", "pixel_values", "split"))

    loaded = FER2013Loader().load(path)

    assert len(loaded.samples) == 1
    assert loaded.samples[0].label == "happy"


def test_missing_required_columns_raises(tmp_path):
    df = pd.DataFrame({"foo": [1, 2], "bar": ["a", "b"]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(DatasetError):
        FER2013Loader().load(str(path))


def test_nonexistent_path_raises():
    with pytest.raises(DatasetError):
        FER2013Loader().load("/nonexistent/path/does/not/exist.csv")


def test_directory_with_single_csv_resolves(tmp_path):
    rows = [[0, _pixel_string(1), "Training"]]
    _write_csv(tmp_path, rows)  # writes synthetic_fer2013.csv inside tmp_path

    loaded = FER2013Loader().load(str(tmp_path))

    assert len(loaded.samples) == 1


def test_directory_with_no_csv_raises(tmp_path):
    (tmp_path / "not_a_csv.txt").write_text("hello")
    with pytest.raises(DatasetError):
        FER2013Loader().load(str(tmp_path))


def test_no_official_split_column_when_absent(tmp_path):
    rows = [[0, _pixel_string(1)], [1, _pixel_string(2)]]
    path = _write_csv(tmp_path, rows, columns=("emotion", "pixels"))

    loaded = FER2013Loader().load(path)

    assert loaded.schema.has_official_split_column is False
    assert all(s.raw_official_split is None for s in loaded.samples)
