"""
Failure logging for samples that fail during representation or feature
extraction (Phase 5 spec sec. 25-26).

Never silently dropped: every failed sample gets one row here, with the
split, the pipeline stage it failed at, and the error. This is separate
from FER2013Loader's malformed-row reporting (app.schemas.dataset.
MalformedRecord), which covers rows that never became a valid
FER2013Sample in the first place.
"""

from __future__ import annotations

import csv
from pathlib import Path

from app.core.logging import get_logger
from app.schemas.dataset import FailureRecord

logger = get_logger(__name__)

_FIELDNAMES = ["sample_id", "split", "stage", "error_type", "error_message"]


class FailureLogger:
    """Appends FailureRecord rows to a CSV file, creating it (with header) on first use."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None

    def __enter__(self) -> "FailureLogger":
        write_header = not self._path.exists() or self._path.stat().st_size == 0
        self._file = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_FIELDNAMES)
        if write_header:
            self._writer.writeheader()
            self._file.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()

    def log(self, record: FailureRecord) -> None:
        self._writer.writerow(
            {
                "sample_id": record.sample_id,
                "split": record.split,
                "stage": record.stage,
                "error_type": record.error_type,
                "error_message": record.error_message,
            }
        )
        self._file.flush()

    @property
    def path(self) -> Path:
        return self._path
