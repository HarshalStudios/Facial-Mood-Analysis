"""
Data contracts for the Phase 6 fusion-model training/evaluation pipeline.

Mirrors the style of app.schemas.dataset: plain dataclasses describing what
moves between app.fusion components. Nothing here trains or evaluates a
model -- that lives in app.fusion.training.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainingConfig:
    """One concrete Logistic Regression configuration (Phase 6 spec sec. 14/16)."""

    C: float
    max_iter: int
    random_state: int
    solver: str = "lbfgs"
    class_weight: str | None = None


@dataclass
class SplitMetrics:
    """Metrics computed on one split (train/validation/test) for one model config."""

    split: str
    n_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: list[list[int]]  # rows/cols in CANONICAL_EMOTION_LABELS order
    confusion_matrix_labels: list[str]
    per_class_report: dict  # sklearn classification_report(output_dict=True) shape

    def to_json(self) -> dict:
        return {
            "split": self.split,
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_labels": self.confusion_matrix_labels,
            "per_class_report": self.per_class_report,
        }


@dataclass
class CandidateResult:
    """Validation outcome for one candidate configuration during model selection."""

    config: TrainingConfig
    validation_metrics: SplitMetrics
    converged: bool
    n_iter: list[int] = field(default_factory=list)


@dataclass
class ModelSelectionResult:
    """Outcome of evaluating a small, documented set of candidate configurations."""

    candidates: list[CandidateResult]
    selection_metric: str  # e.g. "macro_f1"
    selected_config: TrainingConfig
    selection_reason: str


@dataclass
class FusionModelMetadata:
    """Persisted alongside the model artifact (Phase 6 spec sec. 26)."""

    model_type: str
    feature_count: int
    class_count: int
    classes: list[str]
    feature_schema_version: str
    scaler: str
    random_state: int
    hyperparameters: dict
    training_dataset: str
    model_version: str
    trained_at: str
    dataset_sizes: dict[str, int]

    def to_json(self) -> dict:
        return {
            "model_type": self.model_type,
            "feature_count": self.feature_count,
            "class_count": self.class_count,
            "classes": self.classes,
            "feature_schema_version": self.feature_schema_version,
            "scaler": self.scaler,
            "random_state": self.random_state,
            "hyperparameters": self.hyperparameters,
            "training_dataset": self.training_dataset,
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "dataset_sizes": self.dataset_sizes,
        }


@dataclass
class FusionTrainingReport:
    """Full outcome of one Phase 6 training run."""

    config: TrainingConfig
    dataset_sizes: dict[str, int]
    model_selection: ModelSelectionResult | None
    validation_metrics: SplitMetrics
    test_metrics: SplitMetrics
    model_path: str
    scaler_path: str
    metadata_path: str
    coefficients_path: str
