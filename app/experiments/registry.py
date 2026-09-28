"""Authoritative Phase 9 feature-group and experiment registry."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.feature_vector import FEATURE_NAMES

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "deep_emotion": tuple(FEATURE_NAMES[0:7]),
    "lbp": tuple(FEATURE_NAMES[7:17]),
    "edge": (FEATURE_NAMES[17],),
    "gradient": (FEATURE_NAMES[18],),
    "image_quality": tuple(FEATURE_NAMES[19:22]),
    "all_handcrafted": tuple(FEATURE_NAMES[7:22]),
    "all_features": tuple(FEATURE_NAMES),
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    name: str
    feature_groups: tuple[str, ...]
    model: str = "logistic_regression"


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("exp_001", "FER2013 Emotion-Probability Baseline", ("deep_emotion",)),
    ExperimentSpec("exp_002", "FER2013 Emotion Probabilities + LBP", ("deep_emotion", "lbp")),
    ExperimentSpec("exp_003", "FER2013 Emotion Probabilities + Edge", ("deep_emotion", "edge")),
    ExperimentSpec("exp_004", "FER2013 Emotion Probabilities + Gradient", ("deep_emotion", "gradient")),
    ExperimentSpec("exp_005", "FER2013 Emotion Probabilities + Image Quality", ("deep_emotion", "image_quality")),
    ExperimentSpec("exp_006", "FER2013 Emotion Probabilities + All Handcrafted", ("all_features",)),
    ExperimentSpec("exp_007", "Handcrafted Only", ("all_handcrafted",)),
)


def feature_names_for(spec: ExperimentSpec) -> list[str]:
    names: list[str] = []
    for group in spec.feature_groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {group}")
        for name in FEATURE_GROUPS[group]:
            if name not in names:
                names.append(name)
    return names
