"""
LogisticRegressionFusionModel (Phase 6 spec sec. 12, 25, 28-30).

Wraps a scikit-learn Pipeline([("scaler", StandardScaler()), ("classifier",
LogisticRegression(...))]) behind the app.fusion.base.FusionModel interface
so Phase 7 never has to know it's a Pipeline, and can never accidentally
apply a scaler that wasn't fitted together with the classifier (spec sec.
11: "training and inference must apply exactly the same fitted
transformation").

The classifier is always fit on integer emotion_id labels in
CANONICAL_EMOTION_LABELS order (see app.fusion.data), so `classes_` is
guaranteed to be [0, 1, ..., 6] in that exact order -- predict_proba's
columns therefore line up with CANONICAL_EMOTION_LABELS by position,
without relying on scikit-learn's default alphabetical label sorting.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import ModelLoadError, PredictionError
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.fusion.base import FusionModel
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH, FeatureVector
from app.schemas.fusion import TrainingConfig
from app.schemas.prediction import EmotionPrediction

EXPECTED_CLASS_COUNT = len(CANONICAL_EMOTION_LABELS)


def build_pipeline(config: TrainingConfig) -> Pipeline:
    """Construct an (unfitted) scaler + Logistic Regression pipeline for one config."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config.C,
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                    solver=config.solver,
                    class_weight=config.class_weight,
                ),
            ),
        ]
    )


class LogisticRegressionFusionModel(FusionModel):
    """The Phase 6 feature-level fusion classifier: StandardScaler -> LogisticRegression."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None

    # -- training-time construction (not part of the FusionModel interface) --

    @classmethod
    def from_fitted_pipeline(cls, pipeline: Pipeline) -> "LogisticRegressionFusionModel":
        instance = cls()
        instance._pipeline = pipeline
        return instance

    # -- FusionModel interface --

    def load(self, model_path: str, scaler_path: str, *, metadata: dict | None = None) -> None:
        """
        Load the fitted pipeline from `model_path` (a single artifact that
        already contains both the fitted scaler and the fitted classifier,
        spec sec. 25). `scaler_path` is also required to exist -- Phase 6
        additionally persists the scaler on its own (spec sec. 11) for
        transparency/debugging -- but the pipeline at `model_path` is what
        is actually used for inference, so a mismatch between the two
        files is structurally impossible.

        Validates the loaded object against the Phase 6/7 contract (spec
        sec. 28): must exist, must expect exactly 22 features, must
        produce exactly 7 class probabilities. Fails clearly (raises
        ModelLoadError) rather than silently reshaping or truncating.
        """
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)

        if not model_path.exists():
            raise ModelLoadError(f"Fusion model artifact not found: {model_path}")
        if not scaler_path.exists():
            raise ModelLoadError(f"Fusion scaler artifact not found: {scaler_path}")

        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001 - re-raised as a BackendError
            raise ModelLoadError(f"Failed to deserialize fusion model at {model_path}: {exc}") from exc

        if not isinstance(pipeline, Pipeline):
            raise ModelLoadError(f"Artifact at {model_path} is not a scikit-learn Pipeline")
        if "scaler" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
            raise ModelLoadError(f"Pipeline at {model_path} is missing expected 'scaler'/'classifier' steps")

        classifier = pipeline.named_steps["classifier"]
        n_features = getattr(classifier, "n_features_in_", None)
        if n_features != FEATURE_VECTOR_LENGTH:
            raise ModelLoadError(
                f"Fusion model expects {n_features} feature(s), but the feature schema is "
                f"{FEATURE_VECTOR_LENGTH}-dimensional. Refusing to load a mismatched model.",
                details={"model_feature_count": n_features, "expected_feature_count": FEATURE_VECTOR_LENGTH},
            )

        classes = list(getattr(classifier, "classes_", []))
        if len(classes) != EXPECTED_CLASS_COUNT or classes != list(range(EXPECTED_CLASS_COUNT)):
            raise ModelLoadError(
                f"Fusion model has {len(classes)} class(es) with labels {classes}; expected "
                f"exactly {EXPECTED_CLASS_COUNT} integer classes [0..{EXPECTED_CLASS_COUNT - 1}] "
                "in CANONICAL_EMOTION_LABELS order.",
                details={"model_classes": classes},
            )

        self._pipeline = pipeline

    def predict(self, feature_vector: FeatureVector) -> EmotionPrediction:
        """
        Run inference on a single 22D feature vector (spec sec. 29-30).
        Confidence is defined, without embellishment, as the maximum
        predicted class probability -- these are raw Logistic Regression
        probabilities, not calibrated probabilities (spec sec. 31).
        """
        if self._pipeline is None:
            raise PredictionError("LogisticRegressionFusionModel.predict() called before load()/fit()")

        x = feature_vector.values.reshape(1, -1)
        try:
            proba = self._pipeline.predict_proba(x)[0]
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(f"Fusion model inference failed: {exc}") from exc

        if proba.shape != (EXPECTED_CLASS_COUNT,):
            raise PredictionError(
                f"Fusion model returned {proba.shape[0]} probabilities, expected {EXPECTED_CLASS_COUNT}"
            )

        probabilities = {label: float(p) for label, p in zip(CANONICAL_EMOTION_LABELS, proba)}
        best_idx = int(np.argmax(proba))
        return EmotionPrediction(
            emotion=CANONICAL_EMOTION_LABELS[best_idx],
            confidence=float(proba[best_idx]),
            probabilities=probabilities,
        )

    # -- batch helpers used by training/evaluation (not part of the FusionModel interface) --

    def predict_ids(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise PredictionError("LogisticRegressionFusionModel.predict_ids() called before load()/fit()")
        return self._pipeline.predict(X)

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise PredictionError("LogisticRegressionFusionModel.predict_proba_matrix() called before load()/fit()")
        return self._pipeline.predict_proba(X)

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise PredictionError("Pipeline not yet fitted/loaded")
        return self._pipeline
