"""
Phase 6 entrypoint: load a saved fusion model and evaluate it on a chosen split.

    python scripts/evaluate_fusion_model.py [--split test] [--features-dir PATH]

Uses the same FusionModel.load() contract Phase 7 will use, so this also
doubles as a sanity check that the saved artifacts load and validate
correctly (Phase 6 spec sec. 28, 42) before handing off to Phase 7.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import BackendError  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.emotion.validation import CANONICAL_EMOTION_LABELS  # noqa: E402
from app.fusion.data import default_split_paths, load_split  # noqa: E402
from app.fusion.metrics import sanity_check_predictions  # noqa: E402
from app.fusion.model import LogisticRegressionFusionModel  # noqa: E402
from app.fusion.training import evaluate_split  # noqa: E402
from app.schemas.dataset import SPLIT_NAMES  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLIT_NAMES, default="test")
    parser.add_argument("--features-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--scaler-path", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    settings = get_settings()

    features_dir = Path(args.features_dir) if args.features_dir else settings.features_dir
    csv_path = default_split_paths(features_dir)[args.split]
    model_path = Path(args.model_path) if args.model_path else settings.fusion_model_path
    scaler_path = Path(args.scaler_path) if args.scaler_path else settings.fusion_scaler_path

    if not csv_path.exists():
        print(f"\nERROR: feature CSV not found for split '{args.split}': {csv_path}\n", file=sys.stderr)
        return 1

    model = LogisticRegressionFusionModel()
    try:
        model.load(str(model_path), str(scaler_path))
    except BackendError as exc:
        print(f"\nERROR: could not load fusion model: {exc.message}\nDetails: {exc.details}\n", file=sys.stderr)
        return 1

    split_data = load_split(csv_path, args.split)

    y_pred = model.predict_ids(split_data.X)
    y_proba = model.predict_proba_matrix(split_data.X)
    sanity_check_predictions(split_data.y, y_pred, y_proba)

    metrics = evaluate_split(model, split_data)

    print("\n" + "=" * 70)
    print(f"FUSION MODEL EVALUATION -- split: {args.split}")
    print("=" * 70)
    print(f"model  : {model_path}")
    print(f"scaler : {scaler_path}")
    print(f"samples: {metrics.n_samples}")
    print(f"\naccuracy        : {metrics.accuracy:.4f}")
    print(f"macro precision : {metrics.macro_precision:.4f}")
    print(f"macro recall    : {metrics.macro_recall:.4f}")
    print(f"macro F1        : {metrics.macro_f1:.4f}")
    print(f"weighted F1     : {metrics.weighted_f1:.4f}")
    print("\nper-class:")
    for label in CANONICAL_EMOTION_LABELS:
        row = metrics.per_class_report.get(label, {})
        print(
            f"  {label:<10} precision={row.get('precision', 0):.3f} "
            f"recall={row.get('recall', 0):.3f} f1={row.get('f1-score', 0):.3f} "
            f"support={int(row.get('support', 0))}"
        )
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
