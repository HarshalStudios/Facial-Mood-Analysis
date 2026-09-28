"""
Fusion model artifact loading for the real-time predictor (Phase 7).

Wraps the Phase 6 -> Phase 7 handoff contract documented in the README
("Phase 6 -> Phase 7 handoff contract") behind one function,
`load_fusion_model()`, so `RealTimePredictor` never touches joblib or
model paths directly.

Two layers of validation happen here:

1. `LogisticRegressionFusionModel.load()` (Phase 6) already refuses to
   load a pipeline that doesn't expect exactly `FEATURE_VECTOR_LENGTH`
   features or exactly the 7 canonical classes in order -- see that
   module's docstring. This is the authoritative check, since it
   inspects the actual fitted estimator.
2. This module additionally reads `fusion_model_metadata.json`, if
   present next to the model artifact, and cross-checks its recorded
   `feature_count`/`classes`/`feature_schema_version` against the
   feature schema the running code actually uses
   (`app.schemas.feature_vector`). This catches a *documented*
   mismatch even in the (unusual) case the pipeline-level check above
   would not, and lets the predictor surface `model_version` /
   `feature_schema_version` for traceability (spec sec. 40).

A missing model/scaler file, a corrupt artifact, or any of the above
mismatches raises `ModelLoadError` -- a fatal initialization failure
(spec sec. 23), never silently ignored or worked around.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelLoadError
from app.core.logging import get_logger
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.fusion.model import LogisticRegressionFusionModel
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_VERSION

logger = get_logger(__name__)

DEFAULT_METADATA_FILENAME = "fusion_model_metadata.json"


def _metadata_path_for(model_path: Path) -> Path:
    return model_path.parent / DEFAULT_METADATA_FILENAME


def load_metadata(model_path: Path) -> dict | None:
    """
    Read `fusion_model_metadata.json` next to `model_path`, if it
    exists. Returns None (not an error) if it's simply absent -- Phase
    6 always writes it, but a hand-placed/older model artifact might
    not have one, and the pipeline-level check in
    `LogisticRegressionFusionModel.load()` is still authoritative
    either way.
    """
    metadata_path = _metadata_path_for(model_path)
    if not metadata_path.exists():
        logger.warning(
            "No fusion model metadata found at %s; proceeding with pipeline-level "
            "validation only (LogisticRegressionFusionModel.load() still enforces "
            "feature count and class order).",
            metadata_path,
        )
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(
            f"Failed to read fusion model metadata at {metadata_path}: {exc}"
        ) from exc


def _validate_metadata(metadata: dict) -> None:
    """Cross-check recorded metadata against the feature schema this code runs. Raises ModelLoadError on mismatch."""
    feature_count = metadata.get("feature_count")
    if feature_count is not None and int(feature_count) != FEATURE_VECTOR_LENGTH:
        raise ModelLoadError(
            "Fusion model metadata feature_count does not match the current feature schema",
            details={"metadata_feature_count": feature_count, "expected_feature_count": FEATURE_VECTOR_LENGTH},
        )

    classes = metadata.get("classes")
    if classes is not None and list(classes) != list(CANONICAL_EMOTION_LABELS):
        raise ModelLoadError(
            "Fusion model metadata class order does not match CANONICAL_EMOTION_LABELS",
            details={"metadata_classes": classes, "expected_classes": CANONICAL_EMOTION_LABELS},
        )

    schema_version = metadata.get("feature_schema_version")
    if schema_version is not None and str(schema_version) != str(FEATURE_VECTOR_VERSION):
        # Not necessarily fatal on its own -- the pipeline-level feature/class
        # count check already ran -- but this is exactly the "dangerous ML
        # error" the spec calls out (correct count, silently different
        # definition), so it must not pass silently.
        raise ModelLoadError(
            "Fusion model metadata feature_schema_version does not match the running "
            "code's FEATURE_VECTOR_VERSION. Refusing to load a model that may have been "
            "trained against a different feature definition.",
            details={
                "metadata_feature_schema_version": schema_version,
                "running_feature_vector_version": FEATURE_VECTOR_VERSION,
            },
        )


def load_fusion_model(
    settings: Settings | None = None,
) -> tuple[LogisticRegressionFusionModel, dict | None]:
    """
    Load and validate the Phase 6 fusion model artifact referenced by
    `settings.fusion_model_path` / `settings.fusion_scaler_path`.

    Returns `(model, metadata)`, where `metadata` is the parsed
    `fusion_model_metadata.json` dict, or None if it wasn't found.
    Raises `ModelLoadError` for any missing file, deserialization
    failure, or schema mismatch.
    """
    settings = settings or get_settings()
    model_path = Path(settings.fusion_model_path)
    scaler_path = Path(settings.fusion_scaler_path)

    metadata = load_metadata(model_path)
    if metadata is not None:
        _validate_metadata(metadata)

    model = LogisticRegressionFusionModel()
    model.load(str(model_path), str(scaler_path))

    logger.info(
        "Fusion model loaded from %s (model_version=%s, feature_schema_version=%s)",
        model_path,
        (metadata or {}).get("model_version", "unknown"),
        (metadata or {}).get("feature_schema_version", "unknown"),
    )
    return model, metadata
