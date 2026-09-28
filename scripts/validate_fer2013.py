#!/usr/bin/env python3
"""Strict, read-only validation for a real FER2013 CSV."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import pandas as pd

EXPECTED_USAGE = {"training", "publictest", "privatetest"}
EXPECTED_LABELS = set(range(7))


def normalize_col(x: object) -> str:
    return str(x).strip().lower()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    args = p.parse_args()
    path = Path(args.dataset_path)
    if not path.is_file():
        print(f"ERROR: dataset file does not exist: {path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR: cannot read CSV: {exc}", file=sys.stderr)
        return 1

    cols = {normalize_col(c): c for c in df.columns}
    emotion = cols.get("emotion") or cols.get("label")
    pixels = cols.get("pixels") or cols.get("pixel_values") or cols.get("image")
    usage = cols.get("usage") or cols.get("split")
    errors: list[str] = []
    if emotion is None: errors.append("missing emotion/label column")
    if pixels is None: errors.append("missing pixels column")
    if usage is None: errors.append("missing Usage/split column; official FER2013 should have one")
    if errors:
        print("ERROR: " + "; ".join(errors), file=sys.stderr)
        print(f"Columns found: {list(df.columns)}", file=sys.stderr)
        return 1

    labels = pd.to_numeric(df[emotion], errors="coerce")
    invalid_labels = int(labels.isna().sum() + (~labels.isin(EXPECTED_LABELS)).sum())
    usage_norm = df[usage].astype(str).str.strip().str.lower().str.replace(" ", "", regex=False).str.replace("_", "", regex=False).str.replace("-", "", regex=False)
    invalid_usage = int((~usage_norm.isin(EXPECTED_USAGE)).sum())

    bad_pixel_rows = 0
    pixel_min = 255
    pixel_max = 0
    for value in df[pixels]:
        if not isinstance(value, str):
            bad_pixel_rows += 1
            continue
        tokens = value.split()
        if len(tokens) != 2304:
            bad_pixel_rows += 1
            continue
        try:
            vals = [int(x) for x in tokens]
        except ValueError:
            bad_pixel_rows += 1
            continue
        if any(v < 0 or v > 255 for v in vals):
            bad_pixel_rows += 1
            continue
        pixel_min = min(pixel_min, min(vals))
        pixel_max = max(pixel_max, max(vals))

    print("FER2013 VALIDATION")
    print(f"file: {path}")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    print(f"image shape: 48x48 grayscale (expected)")
    print(f"pixel rows invalid: {bad_pixel_rows}")
    print(f"pixel range observed: {pixel_min}..{pixel_max}")
    print(f"invalid labels: {invalid_labels}")
    print(f"invalid Usage values: {invalid_usage}")
    print("class distribution:")
    for label, count in sorted(Counter(labels.dropna().astype(int)).items()):
        print(f"  {label}: {count}")
    print("split distribution:")
    for value, count in usage_norm.value_counts().sort_index().items():
        print(f"  {value}: {count}")

    expected_counts = {"training": 28709, "publictest": 3589, "privatetest": 3589}
    print("expected official counts (validation only):")
    for key, expected in expected_counts.items():
        actual = int((usage_norm == key).sum())
        print(f"  {key}: {actual} (expected {expected})")

    ok = (
        len(df) == 35887
        and bad_pixel_rows == 0
        and invalid_labels == 0
        and invalid_usage == 0
        and all(int((usage_norm == k).sum()) == v for k, v in expected_counts.items())
    )
    print(f"VALID: {ok}")
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
