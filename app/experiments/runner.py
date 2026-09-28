"""Reproducible Phase 9 experiment runner.

Real experiments are deliberately blocked until validated Phase 5 feature
CSVs exist. This module never substitutes synthetic data for project data.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.core.config import Settings, get_settings
from app.experiments.metrics import evaluate
from app.experiments.registry import EXPERIMENTS, ExperimentSpec, feature_names_for
from app.fusion.data import (
    assert_all_classes_present_in_training,
    check_splits_exist,
    default_split_paths,
    load_all_splits,
    validate_datasets,
)
from app.fusion.model import build_pipeline
from app.schemas.fusion import TrainingConfig
from app.schemas.feature_vector import FEATURE_NAMES


class Phase9BlockedError(RuntimeError):
    """Raised when real Phase 9 artifacts are not available."""


def validate_artifacts(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    paths = default_split_paths(settings.features_dir)
    missing = check_splits_exist(paths)
    return {
        "ready": not missing,
        "missing": missing,
        "paths": {k: str(v) for k, v in paths.items()},
        "message": "ready" if not missing else "Real Phase 9 evaluation is blocked until these files exist.",
    }


def _load(settings: Settings):
    gate = validate_artifacts(settings)
    if not gate["ready"]:
        raise Phase9BlockedError(json.dumps(gate, indent=2))
    paths = default_split_paths(settings.features_dir)
    validate_datasets(paths, settings.dataset_name)
    splits = load_all_splits(paths)
    assert_all_classes_present_in_training(splits["train"])
    return splits


def run_experiment(
    spec: ExperimentSpec,
    *,
    settings: Settings | None = None,
    C_candidates: list[float] | None = None,
    random_state: int = 42,
) -> dict:
    settings = settings or get_settings()
    splits = _load(settings)
    feature_names = feature_names_for(spec)
    indices = [FEATURE_NAMES.index(n) for n in feature_names]

    train, val, test = splits["train"], splits["validation"], splits["test"]
    Xtr, Xv, Xt = train.X[:, indices], val.X[:, indices], test.X[:, indices]

    candidates = C_candidates or [0.01, 0.1, 1.0, 10.0, 100.0]
    best = None
    for C in candidates:
        pipeline = build_pipeline(TrainingConfig(C=C, max_iter=2000, random_state=random_state))
        pipeline.fit(Xtr, train.y)
        pred = pipeline.predict(Xv)
        metrics = evaluate(val.y, pred)
        candidate = (metrics["macro_f1"], metrics["accuracy"], -C, C)
        if best is None or candidate > best[0]:
            best = (candidate, C)

    assert best is not None
    selected_C = best[1]
    final_pipeline = build_pipeline(
        TrainingConfig(C=selected_C, max_iter=2000, random_state=random_state)
    )
    final_pipeline.fit(Xtr, train.y)
    val_result = evaluate(val.y, final_pipeline.predict(Xv))
    test_result = evaluate(test.y, final_pipeline.predict(Xt))

    return {
        "experiment_id": spec.experiment_id,
        "name": spec.name,
        "status": "COMPLETED",
        "feature_groups": list(spec.feature_groups),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "model": spec.model,
        "selected_C": selected_C,
        "random_state": random_state,
        "validation": val_result,
        "test": test_result,
        "dataset": {
            "train_samples": len(train.y),
            "validation_samples": len(val.y),
            "test_samples": len(test.y),
        },
    }


def write_result(result: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result['experiment_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return path
