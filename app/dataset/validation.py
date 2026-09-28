"""
Feature-dataset validation (Phase 5 spec sec. 22-23, 33, 37).

Runs entirely against the on-disk split CSVs after FeatureDatasetBuilder
has produced them -- a final, independent check that doesn't trust the
builder's own bookkeeping. Nothing here trains or evaluates a model; that
is explicitly out of scope for Phase 5 (spec sec. 32).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.dataset.builder import CSV_FIELDNAMES
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.schemas.dataset import SPLIT_NAMES
from app.schemas.feature_vector import FEATURE_NAMES


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    dataset_name: str
    split_counts: dict[str, int]
    checks: dict[str, bool]
    issues: list[ValidationIssue]
    class_distribution: dict[str, dict[str, int]]

    @property
    def is_valid(self) -> bool:
        return all(self.checks.values())

    def to_text(self) -> str:
        lines = [f"Dataset: {self.dataset_name}", ""]
        lines.append("Total samples: " + str(sum(self.split_counts.values())))
        for split in SPLIT_NAMES:
            lines.append(f"  {split.capitalize()}: {self.split_counts.get(split, 0)}")
        lines.append("")
        lines.append("Feature dimensions: " + str(len(FEATURE_NAMES)))
        lines.append("")
        lines.append("Class distribution:")
        for split in SPLIT_NAMES:
            dist = self.class_distribution.get(split, {})
            lines.append(f"  {split}:")
            for label in CANONICAL_EMOTION_LABELS:
                lines.append(f"    {label.capitalize()}: {dist.get(label, 0)}")
        lines.append("")
        lines.append("Checks:")
        for name, passed in self.checks.items():
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if self.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.message} {issue.details or ''}".rstrip())
        return "\n".join(lines)


def validate_feature_dataset(csv_paths: dict[str, Path], dataset_name: str) -> ValidationReport:
    """
    csv_paths: {"train": Path(...), "validation": Path(...), "test": Path(...)}
    (a subset of SPLIT_NAMES is fine, e.g. when validating a partial run).
    """
    issues: list[ValidationIssue] = []
    checks: dict[str, bool] = {}
    split_counts: dict[str, int] = {}
    class_distribution: dict[str, dict[str, int]] = {}
    sample_id_sets: dict[str, set[str]] = {}

    schema_ok = True
    no_nan = True
    no_inf = True
    valid_emotion_ids = True

    dfs: dict[str, pd.DataFrame] = {}
    for split, path in csv_paths.items():
        path = Path(path)
        if not path.exists():
            issues.append(ValidationIssue("error", f"Missing feature CSV for split '{split}'", {"path": str(path)}))
            schema_ok = False
            continue
        df = pd.read_csv(path)
        dfs[split] = df

        if list(df.columns) != CSV_FIELDNAMES:
            schema_ok = False
            issues.append(
                ValidationIssue(
                    "error",
                    f"Column mismatch in '{split}' feature CSV",
                    {"expected": CSV_FIELDNAMES, "actual": list(df.columns)},
                )
            )

        split_counts[split] = len(df)
        sample_id_sets[split] = set(df["sample_id"].astype(str)) if "sample_id" in df.columns else set()

        if "emotion" in df.columns:
            class_distribution[split] = df["emotion"].value_counts().to_dict()

        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
        if len(feature_cols) != len(FEATURE_NAMES):
            schema_ok = False
            issues.append(
                ValidationIssue(
                    "error",
                    f"'{split}' is missing feature column(s)",
                    {"missing": sorted(set(FEATURE_NAMES) - set(feature_cols))},
                )
            )
        elif len(df) > 0:
            values = df[feature_cols].to_numpy(dtype=np.float64)
            if np.isnan(values).any():
                no_nan = False
                issues.append(ValidationIssue("error", f"'{split}' contains NaN feature value(s)"))
            finite_mask = np.isfinite(values)
            if not finite_mask.all():
                no_inf = False
                issues.append(ValidationIssue("error", f"'{split}' contains infinite feature value(s)"))

        if "emotion_id" in df.columns and "emotion" in df.columns and len(df) > 0:
            expected_ids = df["emotion"].map(
                {label: idx for idx, label in enumerate(CANONICAL_EMOTION_LABELS)}
            )
            if not (df["emotion_id"] == expected_ids).all():
                valid_emotion_ids = False
                issues.append(ValidationIssue("error", f"'{split}' has emotion/emotion_id mismatch(es)"))

    # Cross-split overlap check (spec sec. 37).
    no_overlap = True
    present_splits = list(sample_id_sets.keys())
    for i in range(len(present_splits)):
        for j in range(i + 1, len(present_splits)):
            a, b = present_splits[i], present_splits[j]
            overlap = sample_id_sets[a] & sample_id_sets[b]
            if overlap:
                no_overlap = False
                issues.append(
                    ValidationIssue(
                        "error",
                        f"sample_id overlap between '{a}' and '{b}'",
                        {"overlap_count": len(overlap), "example_ids": sorted(overlap)[:5]},
                    )
                )

    checks["schema_matches_contract"] = schema_ok
    checks["no_nan"] = no_nan
    checks["no_inf"] = no_inf
    checks["exactly_22_feature_columns"] = schema_ok  # covered above; kept as its own named check
    checks["valid_emotion_ids"] = valid_emotion_ids
    checks["no_cross_split_overlap"] = no_overlap
    checks["all_requested_splits_present"] = len(dfs) == len(csv_paths)

    return ValidationReport(
        dataset_name=dataset_name,
        split_counts=split_counts,
        checks=checks,
        issues=issues,
        class_distribution=class_distribution,
    )
