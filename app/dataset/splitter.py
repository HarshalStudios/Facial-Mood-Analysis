"""
Train/validation/test split assignment (Phase 5 spec sec. 16-17).

Two strategies, chosen automatically and never mixed silently:

1. "official" — the supplied dataset has a Usage/split column (FER2013's
   classic Training/PublicTest/PrivateTest, or a recognized equivalent)
   and every sample has a mappable value. That column is preserved as-is,
   not reshuffled.
2. "deterministic_stratified" — no sample has an official split value.
   A stratified split is computed once, with a recorded random seed and
   the configured train/val/test proportions.

If a dataset supplies an official-split column for *some* but not *all*
samples, or contains values this module doesn't recognize, that is
treated as a malformed/ambiguous dataset (DatasetError) rather than
guessed at — mixing "trust the file" and "override the file" logic would
make the split silently non-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from app.core.config import Settings
from app.core.exceptions import DatasetError
from app.core.logging import get_logger
from app.schemas.dataset import SPLIT_NAMES, FER2013Sample

logger = get_logger(__name__)

# Normalized (lower-cased, no separators) raw Usage value -> canonical split.
# Extend if a supplied dataset uses another spelling; do not guess-map an
# unrecognized value (see module docstring).
OFFICIAL_SPLIT_ALIASES: dict[str, str] = {
    "training": "train",
    "train": "train",
    "publictest": "validation",
    "validation": "validation",
    "val": "validation",
    "dev": "validation",
    "privatetest": "test",
    "test": "test",
}


@dataclass
class SplitAssignmentInfo:
    strategy: str  # "official" | "deterministic_stratified"
    details: dict


def _normalize_usage_value(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def assign_splits(
    samples: list[FER2013Sample], settings: Settings
) -> tuple[dict[str, str], SplitAssignmentInfo]:
    """
    Return (sample_id -> split_name, SplitAssignmentInfo). split_name is
    always one of SPLIT_NAMES.
    """
    if not samples:
        raise DatasetError("Cannot assign splits: no samples were loaded")

    have_official = [s.raw_official_split is not None for s in samples]

    if all(have_official):
        return _assign_official_splits(samples)
    if not any(have_official):
        return _assign_stratified_splits(samples, settings)

    missing = sum(1 for h in have_official if not h)
    raise DatasetError(
        "Dataset has an official split column for some rows but not others -- "
        "this is ambiguous and is not silently resolved.",
        details={"rows_missing_split_value": missing, "total_rows": len(samples)},
    )


def _assign_official_splits(samples: list[FER2013Sample]) -> tuple[dict[str, str], SplitAssignmentInfo]:
    assignment: dict[str, str] = {}
    unmapped_values: set[str] = set()

    for sample in samples:
        normalized = _normalize_usage_value(sample.raw_official_split)
        canonical = OFFICIAL_SPLIT_ALIASES.get(normalized)
        if canonical is None:
            unmapped_values.add(sample.raw_official_split)
            continue
        assignment[sample.sample_id] = canonical

    if unmapped_values:
        raise DatasetError(
            "Dataset's official split column contains unrecognized value(s).",
            details={"unrecognized_values": sorted(unmapped_values)},
        )

    counts = {name: sum(1 for v in assignment.values() if v == name) for name in SPLIT_NAMES}
    logger.info("Using official dataset split column. Counts: %s", counts)
    return assignment, SplitAssignmentInfo(
        strategy="official",
        details={"source": "dataset Usage/split column", "counts": counts},
    )


def _assign_stratified_splits(
    samples: list[FER2013Sample], settings: Settings
) -> tuple[dict[str, str], SplitAssignmentInfo]:
    ratios = (settings.dataset_train_ratio, settings.dataset_val_ratio, settings.dataset_test_ratio)
    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 1e-6:
        raise DatasetError(
            "dataset_train_ratio + dataset_val_ratio + dataset_test_ratio must sum to 1.0",
            details={"ratios": ratios, "sum": total_ratio},
        )

    sample_ids = [s.sample_id for s in samples]
    labels = [s.label for s in samples]
    seed = settings.dataset_split_seed

    train_ids, rest_ids, train_labels, rest_labels = train_test_split(
        sample_ids,
        labels,
        train_size=settings.dataset_train_ratio,
        random_state=seed,
        stratify=labels,
    )

    # Split "rest" into validation/test, preserving the configured ratio
    # between them.
    val_of_rest = settings.dataset_val_ratio / (settings.dataset_val_ratio + settings.dataset_test_ratio)
    val_ids, test_ids = train_test_split(
        rest_ids,
        train_size=val_of_rest,
        random_state=seed,
        stratify=rest_labels,
    )

    assignment: dict[str, str] = {}
    for sid in train_ids:
        assignment[sid] = "train"
    for sid in val_ids:
        assignment[sid] = "validation"
    for sid in test_ids:
        assignment[sid] = "test"

    counts = {name: sum(1 for v in assignment.values() if v == name) for name in SPLIT_NAMES}
    logger.info(
        "No official split column found; using deterministic stratified split "
        "(seed=%s, ratios=%s). Counts: %s",
        seed,
        ratios,
        counts,
    )
    return assignment, SplitAssignmentInfo(
        strategy="deterministic_stratified",
        details={
            "random_seed": seed,
            "train_ratio": settings.dataset_train_ratio,
            "val_ratio": settings.dataset_val_ratio,
            "test_ratio": settings.dataset_test_ratio,
            "stratification": "by canonical emotion label",
            "counts": counts,
        },
    )
