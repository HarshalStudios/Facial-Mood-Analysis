"""
Loading + validating the Phase 5 feature-dataset CSVs for Phase 6 training
(spec sec. 5-8, 41).

This module does NOT re-implement Phase 5's structural validation (column
presence, NaN/Inf, cross-split sample_id overlap, emotion/emotion_id
consistency) -- that already exists in app.dataset.validation and is reused
as-is. What this module adds on top, specific to feeding a scikit-learn
model:

  - reads each split's feature columns in the exact FEATURE_NAMES order
    (Phase 4 sec. 6), never trusting incidental CSV column order
  - converts labels to their canonical emotion_id (int), so a model fit on
    these ids gets classes_ == [0..6] in CANONICAL_EMOTION_LABELS order
    rather than whatever order sklearn would derive from sorting the raw
    string labels alphabetically (spec sec. 18 warns explicitly against
    letting alphabetical sorting silently change class ordering)
  - confirms every one of the 7 canonical classes is present in the
    training split, since a fusion model missing a class in training can
    never predict it and predict_proba would return fewer than 7 columns
    (breaking the Phase 7 contract, spec sec. 12/42)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.exceptions import DatasetError
from app.dataset.validation import ValidationReport, validate_feature_dataset
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.schemas.dataset import SPLIT_NAMES
from app.schemas.feature_vector import FEATURE_NAMES, FEATURE_VECTOR_LENGTH


@dataclass
class SplitData:
    """One split's features/labels, ready for scikit-learn."""

    split: str
    X: np.ndarray  # shape (n, 22), FEATURE_NAMES column order
    y: np.ndarray  # shape (n,), int emotion_id in CANONICAL_EMOTION_LABELS order
    sample_ids: list[str]


def default_split_paths(features_dir: Path) -> dict[str, Path]:
    raw_dir = Path(features_dir) / "raw"
    return {split: raw_dir / f"{split}_features.csv" for split in SPLIT_NAMES}


def check_splits_exist(csv_paths: dict[str, Path]) -> list[str]:
    """Return the list of expected paths that do not exist (empty list == all present)."""
    return [str(path) for path in csv_paths.values() if not Path(path).exists()]


def validate_datasets(csv_paths: dict[str, Path], dataset_name: str) -> ValidationReport:
    """
    Run Phase 5's structural validation. Raises DatasetError (does not
    train) if anything is wrong -- per Phase 6 spec sec. 7: "If validation
    fails: STOP training and report the problem. Do not silently repair
    corrupted data."
    """
    report = validate_feature_dataset(csv_paths, dataset_name)
    if not report.is_valid:
        failed_checks = [name for name, passed in report.checks.items() if not passed]
        raise DatasetError(
            "Feature-dataset validation failed; refusing to train on it.",
            details={
                "failed_checks": failed_checks,
                "issues": [
                    {"severity": i.severity, "message": i.message, "details": i.details}
                    for i in report.issues
                ],
            },
        )
    return report


def load_split(csv_path: Path, split_name: str) -> SplitData:
    """
    Load one split CSV into (X, y, sample_ids), enforcing the canonical
    FEATURE_NAMES column order regardless of the CSV's on-disk column
    order, and converting emotion -> emotion_id via CANONICAL_EMOTION_LABELS
    so the resulting y is unambiguous.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise DatasetError(
            f"'{split_name}' feature CSV is missing required column(s)",
            details={"missing": missing, "path": str(csv_path)},
        )
    if "emotion" not in df.columns:
        raise DatasetError(f"'{split_name}' feature CSV is missing the 'emotion' column", details={"path": str(csv_path)})

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)  # explicit order -- see module docstring

    if X.shape[1] != FEATURE_VECTOR_LENGTH:
        raise DatasetError(
            f"'{split_name}' produced a feature matrix with the wrong width",
            details={"expected": FEATURE_VECTOR_LENGTH, "actual": X.shape[1]},
        )
    if not np.all(np.isfinite(X)):
        raise DatasetError(f"'{split_name}' contains NaN or infinite feature value(s) after load", details={"path": str(csv_path)})

    unknown_labels = sorted(set(df["emotion"]) - set(CANONICAL_EMOTION_LABELS))
    if unknown_labels:
        raise DatasetError(
            f"'{split_name}' contains label(s) outside the canonical 7-class schema",
            details={"unknown_labels": unknown_labels},
        )

    label_to_id = {label: idx for idx, label in enumerate(CANONICAL_EMOTION_LABELS)}
    y = df["emotion"].map(label_to_id).to_numpy(dtype=np.int64)

    sample_ids = df["sample_id"].astype(str).tolist() if "sample_id" in df.columns else [str(i) for i in range(len(df))]

    return SplitData(split=split_name, X=X, y=y, sample_ids=sample_ids)


def load_all_splits(csv_paths: dict[str, Path]) -> dict[str, SplitData]:
    return {split: load_split(path, split) for split, path in csv_paths.items()}


def assert_all_classes_present_in_training(train: SplitData) -> None:
    """
    A class entirely absent from the training split can never be predicted
    and would shrink predict_proba's column count below 7, silently
    breaking the Phase 7 contract (spec sec. 12, 42). Fail clearly instead.
    """
    present = set(int(v) for v in np.unique(train.y))
    expected = set(range(len(CANONICAL_EMOTION_LABELS)))
    missing = expected - present
    if missing:
        missing_labels = [CANONICAL_EMOTION_LABELS[i] for i in sorted(missing)]
        raise DatasetError(
            "Training split is missing sample(s) for one or more canonical classes; "
            "a model trained on it could not produce all 7 output probabilities.",
            details={"missing_classes": missing_labels},
        )
