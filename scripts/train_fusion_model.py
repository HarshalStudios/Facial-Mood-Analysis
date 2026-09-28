"""
Phase 6 entrypoint: {train,validation,test}_features.csv -> trained fusion model.

    python scripts/train_fusion_model.py [--features-dir PATH]
        [--candidate-cs 0.01,0.1,1,10,100] [--max-iter N] [--random-state N]
        [--class-weight balanced] [--skip-model-selection] [--overwrite]

Requires Phase 5's output to already exist at
{features-dir}/raw/{train,validation,test}_features.csv. If any is
missing, this exits with a clear error instead of fabricating results
(Phase 6 spec sec. 43) -- run scripts/build_fer2013_features.py against
the real FER2013 dataset first.

Strict separation is enforced throughout (spec sec. 8, 20-22):
TRAIN fits the scaler and every candidate model: VALIDATION is used only
for model selection; TEST is transformed with the train-fitted pipeline
and evaluated exactly once, after selection is final.
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
from app.fusion.data import (  # noqa: E402
    assert_all_classes_present_in_training,
    check_splits_exist,
    default_split_paths,
    load_all_splits,
    validate_datasets,
)
from app.fusion.persistence import (  # noqa: E402
    build_metadata,
    save_pipeline,
    save_scaler,
    write_coefficients_csv,
    write_json,
)
from app.fusion.training import (  # noqa: E402
    DEFAULT_CANDIDATE_CS,
    DEFAULT_MAX_ITER,
    DEFAULT_RANDOM_STATE,
    evaluate_split,
    fit_final_model,
    select_model,
)
from app.fusion.visualization import save_confusion_matrix_image  # noqa: E402
from app.schemas.dataset import SPLIT_NAMES  # noqa: E402
from app.schemas.fusion import TrainingConfig  # noqa: E402

logger = get_logger(__name__)

MODEL_VERSION = "fusion_model_v1"
EXPERIMENTS_DIR = Path("experiments") / "fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=str, default=None, help="Override FMA feature dataset dir (default: settings.features_dir)")
    parser.add_argument(
        "--candidate-cs",
        type=str,
        default=None,
        help=f"Comma-separated C values for model selection (default: {DEFAULT_CANDIDATE_CS})",
    )
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Baseline is 'none' (spec sec. 40); 'balanced' is a documented model-selection experiment, not a default.",
    )
    parser.add_argument(
        "--skip-model-selection",
        action="store_true",
        help="Skip the multi-C sweep and train a single model at --single-c (default C=1.0)",
    )
    parser.add_argument("--single-c", type=float, default=1.0, help="C to use when --skip-model-selection is set")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing model artifacts")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    settings = get_settings()

    features_dir = Path(args.features_dir) if args.features_dir else settings.features_dir
    csv_paths = default_split_paths(features_dir)

    logger.info("=== Phase 6: fusion model training ===")
    logger.info("Feature dataset dir: %s", features_dir)

    # --- Blocker check: Phase 5 output must already exist -----------------
    missing = check_splits_exist(csv_paths)
    if missing:
        message = (
            "Cannot train: the Phase 5 feature-dataset CSV(s) do not exist yet.\n"
            f"Missing: {missing}\n\n"
            "This is expected if Phase 5 has not been run against a real FER2013 "
            "dataset file in this environment. Run:\n"
            "    python scripts/build_fer2013_features.py --dataset-path <path-to-fer2013.csv>\n"
            "first, then re-run this script. No model will be trained and no results "
            "will be fabricated (Phase 6 spec sec. 43)."
        )
        logger.error(message)
        print(f"\nERROR: {message}\n", file=sys.stderr)
        return 1

    # --- Data validation (spec sec. 7, 41) --------------------------------
    try:
        validate_datasets(csv_paths, settings.dataset_name)
        splits = load_all_splits(csv_paths)
        assert_all_classes_present_in_training(splits["train"])
    except BackendError as exc:
        logger.error("Data validation failed: %s | %s", exc.message, exc.details)
        print(f"\nERROR: {exc.message}\nDetails: {exc.details}\n", file=sys.stderr)
        return 1

    train, validation, test = splits["train"], splits["validation"], splits["test"]
    dataset_sizes = {"train": len(train.y), "validation": len(validation.y), "test": len(test.y)}
    logger.info("Dataset sizes: %s", dataset_sizes)

    class_weight = None if args.class_weight == "none" else args.class_weight

    # --- Model selection (TRAIN + VALIDATION only, spec sec. 16, 20) ------
    if args.skip_model_selection:
        selected_config = TrainingConfig(
            C=args.single_c, max_iter=args.max_iter, random_state=args.random_state, class_weight=class_weight
        )
        selection_result = None
        logger.info("--skip-model-selection: training a single model with C=%s", args.single_c)
    else:
        candidate_cs = (
            [float(c) for c in args.candidate_cs.split(",")] if args.candidate_cs else DEFAULT_CANDIDATE_CS
        )
        selection_result = select_model(
            train,
            validation,
            candidate_cs=candidate_cs,
            max_iter=args.max_iter,
            random_state=args.random_state,
            class_weight=class_weight,
        )
        selected_config = selection_result.selected_config
        logger.info(selection_result.selection_reason)

    # --- Final model: fit on TRAIN only (spec sec. 21) ---------------------
    model = fit_final_model(train, selected_config)

    # --- Validation performance report (of the FINAL selected model) ------
    validation_metrics = evaluate_split(model, validation)

    # --- Final, one-time TEST evaluation (spec sec. 22-23) -----------------
    test_metrics = evaluate_split(model, test)

    # --- Save artifacts ------------------------------------------------------
    try:
        model_path = save_pipeline(model.pipeline, settings.fusion_model_path, overwrite=args.overwrite)
        scaler_path = save_scaler(model.pipeline, settings.fusion_scaler_path, overwrite=args.overwrite)
    except BackendError as exc:
        logger.error("Refusing to save artifacts: %s", exc.message)
        print(f"\nERROR: {exc.message}\nPass --overwrite to replace the existing model.\n", file=sys.stderr)
        return 1

    metadata = build_metadata(
        config=selected_config,
        dataset_sizes=dataset_sizes,
        training_dataset=settings.dataset_name,
        feature_schema_version="1.0",
        model_version=MODEL_VERSION,
    )
    metadata_path = write_json(metadata.to_json(), settings.fusion_model_path.parent / "fusion_model_metadata.json")

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    coefficients_path = write_coefficients_csv(model.pipeline, EXPERIMENTS_DIR / "coefficients.csv")

    model_config_json = {
        "model_version": MODEL_VERSION,
        "selected_config": {
            "C": selected_config.C,
            "max_iter": selected_config.max_iter,
            "random_state": selected_config.random_state,
            "solver": selected_config.solver,
            "class_weight": selected_config.class_weight,
        },
        "selection_metric": selection_result.selection_metric if selection_result else None,
        "selection_reason": selection_result.selection_reason if selection_result else "Model selection skipped (--skip-model-selection).",
        "candidates": (
            [
                {
                    "C": c.config.C,
                    "converged": c.converged,
                    "n_iter": c.n_iter,
                    "validation_macro_f1": c.validation_metrics.macro_f1,
                    "validation_accuracy": c.validation_metrics.accuracy,
                }
                for c in selection_result.candidates
            ]
            if selection_result
            else []
        ),
        "dataset_sizes": dataset_sizes,
    }
    write_json(model_config_json, EXPERIMENTS_DIR / "model_config.json")
    write_json(validation_metrics.to_json(), EXPERIMENTS_DIR / "validation_results.json")
    write_json(test_metrics.to_json(), EXPERIMENTS_DIR / "test_results.json")
    write_json(
        {"validation": validation_metrics.per_class_report, "test": test_metrics.per_class_report},
        EXPERIMENTS_DIR / "classification_report.json",
    )
    save_confusion_matrix_image(
        validation_metrics.confusion_matrix,
        CANONICAL_EMOTION_LABELS,
        "Validation Confusion Matrix",
        EXPERIMENTS_DIR / "confusion_matrix_validation.png",
    )
    save_confusion_matrix_image(
        test_metrics.confusion_matrix,
        CANONICAL_EMOTION_LABELS,
        "Test Confusion Matrix",
        EXPERIMENTS_DIR / "confusion_matrix.png",
    )

    _print_final_report(
        selected_config, dataset_sizes, selection_result, validation_metrics, test_metrics,
        model_path, scaler_path, metadata_path, coefficients_path,
    )
    return 0


def _print_final_report(
    config, dataset_sizes, selection_result, validation_metrics, test_metrics,
    model_path, scaler_path, metadata_path, coefficients_path,
) -> None:
    print("\n" + "=" * 70)
    print("PHASE 6 REPORT")
    print("=" * 70)

    print("\nA. Training configuration")
    print(f"  model       : LogisticRegression")
    print(f"  solver      : {config.solver}")
    print(f"  C           : {config.C}")
    print(f"  max_iter    : {config.max_iter}")
    print(f"  random_state: {config.random_state}")
    print(f"  class_weight: {config.class_weight}")
    print(f"  scaler      : StandardScaler (fit on train only)")

    print("\nB. Dataset sizes")
    for split in SPLIT_NAMES:
        print(f"  {split:<12}: {dataset_sizes.get(split, 0)}")

    print("\nC. Model selection")
    if selection_result:
        print(f"  metric: {selection_result.selection_metric}")
        for c in selection_result.candidates:
            print(
                f"    C={c.config.C:<8} macro_f1={c.validation_metrics.macro_f1:.4f} "
                f"accuracy={c.validation_metrics.accuracy:.4f} converged={c.converged}"
            )
        print(f"  {selection_result.selection_reason}")
    else:
        print("  skipped (--skip-model-selection)")

    print("\nD. Validation results (final selected model)")
    print(f"  accuracy         : {validation_metrics.accuracy:.4f}")
    print(f"  macro precision  : {validation_metrics.macro_precision:.4f}")
    print(f"  macro recall     : {validation_metrics.macro_recall:.4f}")
    print(f"  macro F1         : {validation_metrics.macro_f1:.4f}")
    print(f"  weighted F1      : {validation_metrics.weighted_f1:.4f}")

    print("\nE. Final TEST results (evaluated once, after model selection)")
    print(f"  accuracy         : {test_metrics.accuracy:.4f}")
    print(f"  macro precision  : {test_metrics.macro_precision:.4f}")
    print(f"  macro recall     : {test_metrics.macro_recall:.4f}")
    print(f"  macro F1         : {test_metrics.macro_f1:.4f}")
    print(f"  weighted F1      : {test_metrics.weighted_f1:.4f}")

    print("\nF. Artifacts")
    print(f"  {model_path}")
    print(f"  {scaler_path}")
    print(f"  {metadata_path}")
    print(f"  {coefficients_path}")
    print(f"  {EXPERIMENTS_DIR / 'confusion_matrix.png'} (test)")
    print(f"  {EXPERIMENTS_DIR / 'confusion_matrix_validation.png'}")
    print(f"  {EXPERIMENTS_DIR / 'model_config.json'}")
    print(f"  {EXPERIMENTS_DIR / 'validation_results.json'}")
    print(f"  {EXPERIMENTS_DIR / 'test_results.json'}")
    print(f"  {EXPERIMENTS_DIR / 'classification_report.json'}")

    print("\nG. Phase 7 handoff")
    print("  Load with app.fusion.model.LogisticRegressionFusionModel().load(model_path, scaler_path)")
    print("  then call .predict(feature_vector) -> EmotionPrediction.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
