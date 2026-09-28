"""
Artifact persistence for the Phase 6 fusion model (spec sec. 25-27, 37).

Saves:
  models/fusion_model.pkl   -- the full fitted Pipeline (scaler + classifier)
  models/scaler.pkl         -- the fitted StandardScaler, extracted from the
                                pipeline, saved separately for transparency/
                                debugging (spec sec. 11) -- inference always
                                goes through the pipeline above, never
                                through this file directly, so the two can
                                never drift apart
  models/fusion_model_metadata.json

Existing artifacts are never silently overwritten (spec sec. 37): call
sites must pass overwrite=True once they've decided that's intended.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from app.core.exceptions import ConfigurationError
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.schemas.feature_vector import FEATURE_NAMES
from app.schemas.fusion import FusionModelMetadata, TrainingConfig


def _refuse_if_exists(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ConfigurationError(
            f"Refusing to overwrite existing artifact at {path} without --overwrite "
            "(Phase 6 spec sec. 37: do not overwrite previous experiments without warning)."
        )


def save_pipeline(pipeline: Pipeline, model_path: str | Path, *, overwrite: bool = False) -> Path:
    model_path = Path(model_path)
    _refuse_if_exists(model_path, overwrite)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return model_path


def save_scaler(pipeline: Pipeline, scaler_path: str | Path, *, overwrite: bool = False) -> Path:
    scaler_path = Path(scaler_path)
    _refuse_if_exists(scaler_path, overwrite)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["scaler"], scaler_path)
    return scaler_path


def load_pipeline(model_path: str | Path) -> Pipeline:
    return joblib.load(Path(model_path))


def build_metadata(
    *,
    config: TrainingConfig,
    dataset_sizes: dict[str, int],
    training_dataset: str,
    feature_schema_version: str,
    model_version: str,
) -> FusionModelMetadata:
    return FusionModelMetadata(
        model_type="LogisticRegression",
        feature_count=len(FEATURE_NAMES),
        class_count=len(CANONICAL_EMOTION_LABELS),
        classes=list(CANONICAL_EMOTION_LABELS),
        feature_schema_version=feature_schema_version,
        scaler="StandardScaler",
        random_state=config.random_state,
        hyperparameters={
            "C": config.C,
            "max_iter": config.max_iter,
            "solver": config.solver,
            "class_weight": config.class_weight,
        },
        training_dataset=training_dataset,
        model_version=model_version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        dataset_sizes=dataset_sizes,
    )


def write_json(data: dict, path: str | Path, *, overwrite: bool = True) -> Path:
    path = Path(path)
    if not overwrite:
        _refuse_if_exists(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def write_coefficients_csv(pipeline: Pipeline, path: str | Path) -> Path:
    """
    Export learned coefficients as feature x class rows (spec sec. 32-33):
    one row per (feature, class) pair, plus a `feature_group` column for
    the descriptive Deep/LBP/DIP grouping. These are model coefficients /
    learned feature weights, not a causal-importance claim.
    """
    classifier = pipeline.named_steps["classifier"]
    coef = classifier.coef_  # shape (n_classes, n_features) for multinomial/OvR alike
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "feature_group", "class", "coefficient"])
        for class_idx, class_label in enumerate(CANONICAL_EMOTION_LABELS):
            for feature_idx, feature_name in enumerate(FEATURE_NAMES):
                writer.writerow(
                    [feature_name, _feature_group(feature_name), class_label, float(coef[class_idx, feature_idx])]
                )
    return path


def _feature_group(feature_name: str) -> str:
    if feature_name.startswith("emotion_"):
        return "deep_emotion"
    if feature_name.startswith("lbp_bin_"):
        return "lbp"
    return "dip_image_quality"
