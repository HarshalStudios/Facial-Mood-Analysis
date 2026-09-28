"""
Phase 5 single-sample inspection (spec sec. 34).

    python scripts/inspect_sample.py --split train --sample-id 123
    python scripts/inspect_sample.py --split train --sample-id 123 --save-representations

Prints one generated feature row's 22 named values and label. With
--save-representations, also re-loads that sample from the original
dataset file, re-runs RepresentationEngine on it, and exports the
individual representation PNGs via app.representations.export (Phase 3)
to outputs/representations/{split}_{sample_id}/, so image -> representation
-> feature can be checked by eye.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.dataset.fer2013_loader import FER2013Loader  # noqa: E402
from app.dataset.preprocessing import to_representation_input  # noqa: E402
from app.representations.engine import RepresentationEngine  # noqa: E402
from app.representations.export import export_representations  # noqa: E402
from app.schemas.feature_vector import FEATURE_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--save-representations", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    csv_path = settings.features_dir / "raw" / f"{args.split}_features.csv"
    if not csv_path.exists():
        print(f"No feature CSV at {csv_path}. Run scripts/build_fer2013_features.py first.")
        return 1

    df = pd.read_csv(csv_path)
    row = df[df["sample_id"].astype(str) == str(args.sample_id)]
    if row.empty:
        print(f"sample_id {args.sample_id!r} not found in {csv_path}")
        return 1
    row = row.iloc[0]

    print(f"sample_id : {row['sample_id']}")
    print(f"split     : {row['split']}")
    print(f"emotion   : {row['emotion']} (id={row['emotion_id']})")
    print("\n22 feature values:")
    for name in FEATURE_NAMES:
        print(f"  {name:<16}: {row[name]:.6f}")

    if args.save_representations:
        loaded = FER2013Loader().load(settings.dataset_path)
        match = next((s for s in loaded.samples if s.sample_id == str(args.sample_id)), None)
        if match is None:
            print(f"\nCould not re-locate sample_id {args.sample_id!r} in {settings.dataset_path} "
                  f"to regenerate representations (dataset file may have changed).")
            return 1
        bgr = to_representation_input(match.image)
        representations = RepresentationEngine(settings).generate(bgr)
        out_dir = settings.representation_debug_dir / f"{args.split}_{args.sample_id}"
        written = export_representations(representations, out_dir)
        print(f"\nExported {len(written)} representation image(s) to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
