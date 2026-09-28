"""
Phase 6 training + model-selection orchestration.

Strict order, mirroring the spec (sec. 4, 8, 20-22):

    TRAIN      -> fit scaler + candidate models
    VALIDATION -> pick the best candidate by macro F1 (spec sec. 17)
    (refit is NOT needed: each candidate was already fit on train only)
    TEST       -> transform with the *same* fitted pipeline, evaluate once

The test split is never touched until `evaluate_test` is called, and
nothing in this module lets a caller accidentally feed test data into
`select_model`.
"""

from __future__ import annotations

import warnings

import numpy as np

from app.core.logging import get_logger
from app.fusion.data import SplitData
from app.fusion.metrics import compute_split_metrics, sanity_check_predictions
from app.fusion.model import LogisticRegressionFusionModel, build_pipeline
from app.schemas.fusion import CandidateResult, ModelSelectionResult, SplitMetrics, TrainingConfig

logger = get_logger(__name__)

DEFAULT_CANDIDATE_CS: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
DEFAULT_MAX_ITER = 2000
DEFAULT_RANDOM_STATE = 42
SELECTION_METRIC = "macro_f1"


def select_model(
    train: SplitData,
    validation: SplitData,
    *,
    candidate_cs: list[float] | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    class_weight: str | None = None,
) -> ModelSelectionResult:
    """
    Fit one candidate Logistic Regression per C on TRAIN only, evaluate each
    on VALIDATION, and select the candidate with the highest macro F1 (spec
    sec. 16-17, 20). Ties are broken by higher accuracy, then by smaller C
    (prefer the simpler/more-regularized model, all else equal).
    """
    candidate_cs = candidate_cs or DEFAULT_CANDIDATE_CS
    candidates: list[CandidateResult] = []

    for C in candidate_cs:
        config = TrainingConfig(
            C=C, max_iter=max_iter, random_state=random_state, class_weight=class_weight
        )
        pipeline = build_pipeline(config)

        converged = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(train.X, train.y)
            for w in caught:
                if "ConvergenceWarning" in str(w.category):
                    converged = False
                    logger.warning(
                        "Candidate C=%s did not converge within max_iter=%d: %s", C, max_iter, w.message
                    )

        n_iter = pipeline.named_steps["classifier"].n_iter_
        model = LogisticRegressionFusionModel.from_fitted_pipeline(pipeline)

        y_pred = model.predict_ids(validation.X)
        val_metrics = compute_split_metrics("validation", validation.y, y_pred)

        candidates.append(
            CandidateResult(
                config=config,
                validation_metrics=val_metrics,
                converged=converged,
                n_iter=list(np.atleast_1d(n_iter)),
            )
        )
        logger.info(
            "Candidate C=%s -> validation macro_f1=%.4f accuracy=%.4f (converged=%s)",
            C,
            val_metrics.macro_f1,
            val_metrics.accuracy,
            converged,
        )

    best = max(
        candidates,
        key=lambda c: (c.validation_metrics.macro_f1, c.validation_metrics.accuracy, -c.config.C),
    )
    reason = (
        f"Selected C={best.config.C} because it achieved the highest validation macro F1 "
        f"({best.validation_metrics.macro_f1:.4f}) among candidates {candidate_cs}, "
        f"evaluated on the validation split only."
    )

    return ModelSelectionResult(
        candidates=candidates,
        selection_metric=SELECTION_METRIC,
        selected_config=best.config,
        selection_reason=reason,
    )


def fit_final_model(train: SplitData, config: TrainingConfig) -> LogisticRegressionFusionModel:
    """Fit the selected configuration on the training split (spec sec. 21)."""
    pipeline = build_pipeline(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(train.X, train.y)
        for w in caught:
            if "ConvergenceWarning" in str(w.category):
                logger.warning("Final model (C=%s) did not converge within max_iter=%d: %s", config.C, config.max_iter, w.message)
    return LogisticRegressionFusionModel.from_fitted_pipeline(pipeline)


def evaluate_split(model: LogisticRegressionFusionModel, split: SplitData) -> SplitMetrics:
    """
    Evaluate an already-fitted model on one split, running the model
    sanity checks from spec sec. 42 before returning metrics.
    """
    y_pred = model.predict_ids(split.X)
    y_proba = model.predict_proba_matrix(split.X)
    sanity_check_predictions(split.y, y_pred, y_proba)
    return compute_split_metrics(split.split, split.y, y_pred)
