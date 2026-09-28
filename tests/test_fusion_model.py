from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import ModelLoadError, PredictionError
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.fusion.model import LogisticRegressionFusionModel, build_pipeline
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH, FeatureVector
from app.schemas.fusion import TrainingConfig


def _separable_dataset(n_per_class: int = 20, seed: int = 0):
    """22D data where feature i is a strong signal for class i (i<7); rest is noise."""
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for class_id in range(7):
        base = rng.normal(0.0, 0.1, size=(n_per_class, FEATURE_VECTOR_LENGTH))
        base[:, class_id] += 5.0  # strong per-class signal
        X_parts.append(base)
        y_parts.append(np.full(n_per_class, class_id))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def test_build_pipeline_has_scaler_and_classifier_steps():
    config = TrainingConfig(C=1.0, max_iter=500, random_state=42)
    pipeline = build_pipeline(config)
    assert list(pipeline.named_steps.keys()) == ["scaler", "classifier"]
    assert pipeline.named_steps["classifier"].C == 1.0


def test_fitted_model_predicts_all_7_classes_with_probabilities_summing_to_one():
    X, y = _separable_dataset()
    config = TrainingConfig(C=1.0, max_iter=1000, random_state=42)
    pipeline = build_pipeline(config)
    pipeline.fit(X, y)
    model = LogisticRegressionFusionModel.from_fitted_pipeline(pipeline)

    # classes_ must be [0..6] in CANONICAL_EMOTION_LABELS order (not
    # alphabetically re-sorted string labels).
    assert list(pipeline.named_steps["classifier"].classes_) == list(range(7))

    fv = FeatureVector(values=X[0].astype(np.float64))
    result = model.predict(fv)

    assert result.emotion in CANONICAL_EMOTION_LABELS
    assert set(result.probabilities.keys()) == set(CANONICAL_EMOTION_LABELS)
    assert pytest.approx(sum(result.probabilities.values()), abs=1e-6) == 1.0
    assert result.confidence == max(result.probabilities.values())


def test_model_achieves_high_accuracy_on_clearly_separable_synthetic_data():
    # This is a synthetic wiring/correctness check, NOT a claim about
    # real-world FER2013 performance.
    X, y = _separable_dataset()
    config = TrainingConfig(C=1.0, max_iter=1000, random_state=42)
    pipeline = build_pipeline(config)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.95


def test_predict_before_load_or_fit_raises():
    model = LogisticRegressionFusionModel()
    fv = FeatureVector(values=np.zeros(FEATURE_VECTOR_LENGTH))
    with pytest.raises(PredictionError):
        model.predict(fv)


def test_load_raises_when_model_file_missing(tmp_path):
    model = LogisticRegressionFusionModel()
    with pytest.raises(ModelLoadError):
        model.load(str(tmp_path / "nope.pkl"), str(tmp_path / "also_nope.pkl"))


def test_load_raises_when_scaler_file_missing(tmp_path):
    X, y = _separable_dataset()
    pipeline = build_pipeline(TrainingConfig(C=1.0, max_iter=500, random_state=42))
    pipeline.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(pipeline, model_path)

    model = LogisticRegressionFusionModel()
    with pytest.raises(ModelLoadError):
        model.load(str(model_path), str(tmp_path / "missing_scaler.pkl"))


def test_load_raises_when_artifact_is_not_a_pipeline(tmp_path):
    joblib.dump({"not": "a pipeline"}, tmp_path / "model.pkl")
    joblib.dump(StandardScaler(), tmp_path / "scaler.pkl")

    model = LogisticRegressionFusionModel()
    with pytest.raises(ModelLoadError):
        model.load(str(tmp_path / "model.pkl"), str(tmp_path / "scaler.pkl"))


def test_load_raises_when_feature_count_does_not_match_schema(tmp_path):
    # Deliberately fit on 21 features instead of the required 22.
    X = np.random.default_rng(0).normal(size=(50, 21))
    y = np.tile(np.arange(7), 8)[:50]
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=500))])
    pipeline.fit(X, y)
    model_path, scaler_path = tmp_path / "model.pkl", tmp_path / "scaler.pkl"
    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline.named_steps["scaler"], scaler_path)

    model = LogisticRegressionFusionModel()
    with pytest.raises(ModelLoadError):
        model.load(str(model_path), str(scaler_path))


def test_load_raises_when_class_count_does_not_match_schema(tmp_path):
    # Only 3 classes instead of the required 7.
    X = np.random.default_rng(0).normal(size=(30, FEATURE_VECTOR_LENGTH))
    y = np.tile(np.arange(3), 10)
    pipeline = build_pipeline(TrainingConfig(C=1.0, max_iter=500, random_state=42))
    pipeline.fit(X, y)
    model_path, scaler_path = tmp_path / "model.pkl", tmp_path / "scaler.pkl"
    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline.named_steps["scaler"], scaler_path)

    model = LogisticRegressionFusionModel()
    with pytest.raises(ModelLoadError):
        model.load(str(model_path), str(scaler_path))


def test_load_succeeds_for_a_well_formed_artifact_and_predicts(tmp_path):
    X, y = _separable_dataset()
    pipeline = build_pipeline(TrainingConfig(C=1.0, max_iter=1000, random_state=42))
    pipeline.fit(X, y)
    model_path, scaler_path = tmp_path / "model.pkl", tmp_path / "scaler.pkl"
    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline.named_steps["scaler"], scaler_path)

    model = LogisticRegressionFusionModel()
    model.load(str(model_path), str(scaler_path))

    fv = FeatureVector(values=X[0].astype(np.float64))
    result = model.predict(fv)
    assert result.emotion in CANONICAL_EMOTION_LABELS
