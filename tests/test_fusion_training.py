from __future__ import annotations

import inspect

import numpy as np

from app.fusion.data import SplitData
from app.fusion.training import evaluate_split, fit_final_model, select_model
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH


def _separable_split(name: str, n_per_class: int = 15, seed: int = 0) -> SplitData:
    rng = np.random.default_rng(seed)
    X_parts, y_parts, ids = [], [], []
    counter = 0
    for class_id in range(7):
        base = rng.normal(0.0, 0.15, size=(n_per_class, FEATURE_VECTOR_LENGTH))
        base[:, class_id] += 5.0
        X_parts.append(base)
        y_parts.append(np.full(n_per_class, class_id))
        for _ in range(n_per_class):
            ids.append(f"{name}_{counter}")
            counter += 1
    return SplitData(split=name, X=np.vstack(X_parts), y=np.concatenate(y_parts), sample_ids=ids)


def test_select_model_signature_has_no_test_split_parameter():
    # Structural guard against leakage: model selection can only ever be
    # called with train + validation data -- there is no parameter through
    # which a test split could be passed in.
    params = list(inspect.signature(select_model).parameters.keys())
    assert "test" not in params
    assert "train" in params and "validation" in params


def test_select_model_picks_a_candidate_and_reports_all_candidates():
    train = _separable_split("train", seed=1)
    validation = _separable_split("validation", seed=2)

    result = select_model(train, validation, candidate_cs=[0.01, 1.0, 100.0], max_iter=1000, random_state=42)

    assert result.selected_config.C in [0.01, 1.0, 100.0]
    assert len(result.candidates) == 3
    assert result.selection_metric == "macro_f1"
    # The winning candidate should be at least as good as every other candidate on macro F1.
    best_f1 = result.selected_config
    selected_candidate = next(c for c in result.candidates if c.config.C == result.selected_config.C)
    assert all(
        selected_candidate.validation_metrics.macro_f1 >= c.validation_metrics.macro_f1 for c in result.candidates
    )


def test_fit_final_model_then_evaluate_end_to_end_on_separable_data():
    train = _separable_split("train", seed=1)
    validation = _separable_split("validation", seed=2)
    test = _separable_split("test", seed=3)

    selection = select_model(train, validation, candidate_cs=[0.1, 1.0, 10.0], max_iter=1000, random_state=42)
    model = fit_final_model(train, selection.selected_config)

    val_metrics = evaluate_split(model, validation)
    test_metrics = evaluate_split(model, test)

    # Synthetic, clearly-separable data -- a correctness check on the
    # pipeline wiring, not a claim about real FER2013 performance.
    assert val_metrics.accuracy > 0.9
    assert test_metrics.accuracy > 0.9
    assert val_metrics.split == "validation"
    assert test_metrics.split == "test"


def test_final_model_is_fit_only_on_train_not_on_train_plus_validation():
    train = _separable_split("train", n_per_class=15, seed=1)
    validation = _separable_split("validation", n_per_class=15, seed=2)

    from app.schemas.fusion import TrainingConfig

    config = TrainingConfig(C=1.0, max_iter=1000, random_state=42)
    model = fit_final_model(train, config)

    n_samples_seen = model.pipeline.named_steps["classifier"].n_iter_  # not a sample count, just proves it fit
    assert n_samples_seen is not None
    # The scaler's mean_ must match train's mean, not a train+validation blend.
    scaler = model.pipeline.named_steps["scaler"]
    assert np.allclose(scaler.mean_, train.X.mean(axis=0))
    assert not np.allclose(scaler.mean_, np.vstack([train.X, validation.X]).mean(axis=0))
