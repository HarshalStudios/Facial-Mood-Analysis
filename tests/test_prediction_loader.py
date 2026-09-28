"""
Phase 7 tests for app.prediction.loader.

Covers spec sec. 35 tests 1-2 (model loading, schema compatibility) at
the loader level -- i.e. metadata cross-validation in addition to the
pipeline-level checks already covered by tests/test_fusion_model.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from app.core.exceptions import ModelLoadError
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.fusion.model import build_pipeline
from app.prediction.loader import load_fusion_model, load_metadata
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_VERSION
from app.schemas.fusion import TrainingConfig


def _fitted_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(70, FEATURE_VECTOR_LENGTH))
    y = np.tile(np.arange(7), 10)
    pipeline = build_pipeline(TrainingConfig(C=1.0, max_iter=1000, random_state=42))
    pipeline.fit(X, y)

    model_path = tmp_path / "fusion_model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline.named_steps["scaler"], scaler_path)
    return model_path, scaler_path


def _write_metadata(model_path, **overrides):
    metadata = {
        "model_type": "LogisticRegression",
        "feature_count": FEATURE_VECTOR_LENGTH,
        "class_count": 7,
        "classes": list(CANONICAL_EMOTION_LABELS),
        "feature_schema_version": str(FEATURE_VECTOR_VERSION),
        "model_version": "fusion_model_v1",
    }
    metadata.update(overrides)
    metadata_path = model_path.parent / "fusion_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return metadata_path


def test_load_metadata_returns_none_when_file_absent(tmp_path):
    assert load_metadata(tmp_path / "fusion_model.pkl") is None


def test_load_fusion_model_succeeds_with_matching_metadata(tmp_path):
    model_path, scaler_path = _fitted_artifacts(tmp_path)
    _write_metadata(model_path)
    settings = SimpleNamespace(fusion_model_path=model_path, fusion_scaler_path=scaler_path)

    model, metadata = load_fusion_model(settings)

    assert metadata["model_version"] == "fusion_model_v1"
    fv_values = np.zeros(FEATURE_VECTOR_LENGTH)
    from app.schemas.feature_vector import FeatureVector

    result = model.predict(FeatureVector(values=fv_values))
    assert result.emotion in CANONICAL_EMOTION_LABELS


def test_load_fusion_model_succeeds_without_metadata_file(tmp_path):
    model_path, scaler_path = _fitted_artifacts(tmp_path)
    settings = SimpleNamespace(fusion_model_path=model_path, fusion_scaler_path=scaler_path)

    model, metadata = load_fusion_model(settings)

    assert metadata is None


def test_load_fusion_model_raises_on_feature_count_mismatch_in_metadata(tmp_path):
    model_path, scaler_path = _fitted_artifacts(tmp_path)
    _write_metadata(model_path, feature_count=21)
    settings = SimpleNamespace(fusion_model_path=model_path, fusion_scaler_path=scaler_path)

    with pytest.raises(ModelLoadError):
        load_fusion_model(settings)


def test_load_fusion_model_raises_on_class_order_mismatch_in_metadata(tmp_path):
    model_path, scaler_path = _fitted_artifacts(tmp_path)
    scrambled = list(reversed(CANONICAL_EMOTION_LABELS))
    _write_metadata(model_path, classes=scrambled)
    settings = SimpleNamespace(fusion_model_path=model_path, fusion_scaler_path=scaler_path)

    with pytest.raises(ModelLoadError):
        load_fusion_model(settings)


def test_load_fusion_model_raises_on_feature_schema_version_mismatch(tmp_path):
    model_path, scaler_path = _fitted_artifacts(tmp_path)
    _write_metadata(model_path, feature_schema_version="999")
    settings = SimpleNamespace(fusion_model_path=model_path, fusion_scaler_path=scaler_path)

    with pytest.raises(ModelLoadError):
        load_fusion_model(settings)


def test_load_fusion_model_raises_on_missing_artifact(tmp_path):
    settings = SimpleNamespace(
        fusion_model_path=tmp_path / "nope.pkl", fusion_scaler_path=tmp_path / "also_nope.pkl"
    )
    with pytest.raises(ModelLoadError):
        load_fusion_model(settings)
