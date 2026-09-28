"""
FER2013Loader (Phase 5 spec sec. 5-9).

Responsible only for: locate the developer-supplied dataset file, detect
its schema, parse it into FER2013Sample records, and report (not silently
drop) any row it cannot parse. It does not preprocess images for the
representation pipeline, split the dataset, fit anything, or compute
metrics -- see app.dataset.preprocessing / app.dataset.splitter /
app.dataset.builder for those.

Supported source formats, auto-detected from the path:

- A single CSV file with an emotion/label column, a pixels/image column
  (FER2013's classic space-separated-pixel-string format), and optionally
  a Usage/split column. Column names are matched case-insensitively
  against a small set of known aliases (see _EMOTION_COL_ALIASES etc.)
  rather than assumed positionally.
- A directory containing exactly one such CSV file.

If neither pattern is found, or the recognizable columns are missing,
this raises DatasetError with the columns it did find -- per Phase 5 spec
sec. 5, a missing/malformed/unexpected-format dataset is reported, never
silently substituted with something else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.exceptions import DatasetError
from app.core.logging import get_logger
from app.dataset.label_mapping import UnmappableLabelError, parse_fer2013_label
from app.dataset.preprocessing import parse_pixel_string
from app.schemas.dataset import DatasetSchemaInfo, FER2013Sample, LoadedDataset, MalformedRecord

logger = get_logger(__name__)

_EMOTION_COL_ALIASES = {"emotion", "label", "class", "emotion_id", "target"}
_PIXELS_COL_ALIASES = {"pixels", "pixel_values", "image", "pixel"}
_USAGE_COL_ALIASES = {"usage", "split", "dataset", "set", "subset"}
_ID_COL_ALIASES = {"sample_id", "id", "index", "row_id"}

_DEFAULT_SIDE = 48  # FER2013's conventional square image side, verified per-dataset below


class FER2013Loader:
    """Locates and parses the developer-supplied FER2013 dataset file."""

    def load(self, dataset_path: str | Path) -> LoadedDataset:
        path = Path(dataset_path)
        csv_path = self._resolve_csv_path(path)

        logger.info("Loading FER2013 dataset from %s", csv_path)

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pandas can raise many different parser errors
            raise DatasetError(
                f"Failed to read dataset CSV at {csv_path}", details={"error": str(exc)}
            ) from exc

        if df.empty:
            raise DatasetError(f"Dataset CSV at {csv_path} has no rows", details={"path": str(csv_path)})

        columns = {str(c).strip().lower(): c for c in df.columns}
        emotion_col = self._find_column(columns, _EMOTION_COL_ALIASES)
        pixels_col = self._find_column(columns, _PIXELS_COL_ALIASES)
        usage_col = self._find_column(columns, _USAGE_COL_ALIASES)
        id_col = self._find_column(columns, _ID_COL_ALIASES)

        if emotion_col is None or pixels_col is None:
            raise DatasetError(
                "Could not detect required columns in the supplied dataset CSV. "
                "Expected an emotion/label column and a pixels/image column.",
                details={
                    "columns_found": list(df.columns),
                    "recognized_emotion_column": emotion_col,
                    "recognized_pixels_column": pixels_col,
                },
            )

        side = self._infer_square_side(df[pixels_col])
        has_official_split = usage_col is not None

        samples: list[FER2013Sample] = []
        malformed: list[MalformedRecord] = []

        for row_index, row in df.iterrows():
            sample_id = str(row[id_col]) if id_col is not None else str(row_index)
            try:
                sample = self._parse_row(row, sample_id, pixels_col, emotion_col, usage_col, side)
            except (DatasetError, UnmappableLabelError) as exc:
                malformed.append(
                    MalformedRecord(row_index=sample_id, reason=str(exc), details={"stage": "loader"})
                )
                continue
            samples.append(sample)

        if not samples:
            raise DatasetError(
                "Every row in the supplied dataset CSV was malformed; nothing to load.",
                details={"malformed_count": len(malformed)},
            )

        schema = DatasetSchemaInfo(
            source_path=str(csv_path),
            format="csv",
            columns=list(df.columns),
            width=side,
            height=side,
            channels=1,
            dtype="uint8",
            pixel_min=0.0,
            pixel_max=255.0,
            has_official_split_column=has_official_split,
            total_rows=len(df),
        )

        logger.info(
            "Loaded %d valid sample(s), %d malformed row(s) from %s",
            len(samples),
            len(malformed),
            csv_path,
        )
        if malformed:
            logger.warning("First malformed row(s): %s", malformed[: min(5, len(malformed))])

        return LoadedDataset(samples=samples, malformed=malformed, schema=schema)

    # ------------------------------------------------------------------

    def _parse_row(self, row, sample_id, pixels_col, emotion_col, usage_col, side) -> FER2013Sample:
        raw_pixels = row[pixels_col]
        if not isinstance(raw_pixels, str):
            raise DatasetError(f"pixels value is not a string: {type(raw_pixels)!r}")
        image = parse_pixel_string(raw_pixels, width=side, height=side)

        raw_label = row[emotion_col]
        label, label_id = parse_fer2013_label(raw_label)

        raw_official_split = None
        if usage_col is not None:
            usage_value = row[usage_col]
            if isinstance(usage_value, str) and usage_value.strip():
                raw_official_split = usage_value.strip()

        return FER2013Sample(
            sample_id=sample_id,
            image=image,
            label=label,
            label_id=label_id,
            raw_label=str(raw_label),
            raw_official_split=raw_official_split,
        )

    @staticmethod
    def _find_column(columns_lower: dict[str, str], aliases: set[str]) -> str | None:
        for alias in aliases:
            if alias in columns_lower:
                return columns_lower[alias]
        return None

    @staticmethod
    def _infer_square_side(pixels_series: pd.Series) -> int:
        """
        Determine the image side length from the first parseable row,
        rather than assuming FER2013's conventional 48 without checking
        (Phase 5 spec sec. 9).
        """
        for value in pixels_series.head(20):
            if not isinstance(value, str):
                continue
            token_count = len(value.split())
            side = int(round(token_count**0.5))
            if side * side == token_count and side > 0:
                return side
        raise DatasetError(
            "Could not infer a square image side length from the pixels column "
            "(no row in the first 20 had a perfect-square token count).",
            details={"default_side_would_have_been": _DEFAULT_SIDE},
        )

    @staticmethod
    def _resolve_csv_path(path: Path) -> Path:
        if path.is_file():
            return path
        if path.is_dir():
            csv_files = sorted(path.glob("*.csv"))
            if len(csv_files) == 1:
                return csv_files[0]
            if len(csv_files) > 1:
                raise DatasetError(
                    f"Directory {path} contains multiple CSV files; specify the exact file.",
                    details={"candidates": [str(p) for p in csv_files]},
                )
            raise DatasetError(f"Directory {path} contains no CSV file.", details={"path": str(path)})
        raise DatasetError(
            f"Dataset path does not exist: {path}. Supply the real FER2013 file "
            f"(see README 'Phase 5' section) -- it is never auto-downloaded or substituted.",
            details={"path": str(path)},
        )
