"""
Phase 5 dataset summary report (spec sec. 33).

    python scripts/inspect_dataset.py

Reads the already-generated feature-dataset CSVs under
{features_dir}/raw/ and prints sample counts, image dimensions (from
dataset_metadata.json), class distribution per split, and NaN/Inf
counts. Reports no model accuracy -- no model has been trained at this
phase (Phase 5 spec sec. 32-33).

Run scripts/build_fer2013_features.py first; this script does not
generate anything itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.dataset.validation import validate_feature_dataset  # noqa: E402
from app.schemas.dataset import SPLIT_NAMES  # noqa: E402
from app.schemas.feature_vector import FEATURE_NAMES, FEATURE_VECTOR_LENGTH  # noqa: E402


def main() -> int:
    settings = get_settings()
    raw_dir = settings.features_dir / "raw"
    meta_dir = settings.features_dir / "metadata"

    csv_paths = {name: raw_dir / f"{name}_features.csv" for name in SPLIT_NAMES}
    existing = {name: p for name, p in csv_paths.items() if p.exists()}

    if not existing:
        print(f"No feature-dataset CSVs found under {raw_dir}. Run scripts/build_fer2013_features.py first.")
        return 1

    metadata_path = meta_dir / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    report = validate_feature_dataset(existing, settings.dataset_name)

    print(f"Dataset: {metadata.get('dataset', settings.dataset_name)}")
    total = sum(report.split_counts.values())
    print(f"Total samples: {total}")
    for name in SPLIT_NAMES:
        print(f"  {name.capitalize()}: {report.split_counts.get(name, 0)}")

    schema = metadata.get("dataset_schema", {})
    print(f"\nImage dimensions: {schema.get('width', '?')}x{schema.get('height', '?')}")
    print(f"Channels (source): {schema.get('channels', '?')}")

    print("\nClass distribution:")
    for name in SPLIT_NAMES:
        dist = report.class_distribution.get(name, {})
        print(f"  {name}:")
        for label in ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"):
            print(f"    {label.capitalize()}: {dist.get(label, 0)}")

    print(f"\nFeature dimensions: {FEATURE_VECTOR_LENGTH}")

    nan_count = 0
    inf_count = 0
    for name, path in existing.items():
        df = pd.read_csv(path)
        cols = [c for c in FEATURE_NAMES if c in df.columns]
        if cols and len(df) > 0:
            values = df[cols].to_numpy(dtype=np.float64)
            nan_count += int(np.isnan(values).sum())
            inf_count += int(np.isinf(values).sum())

    print(f"\nNaN count: {nan_count}")
    print(f"Infinity count: {inf_count}")

    print("\nValidation checks:")
    for name, passed in report.checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    return 0 if report.is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
