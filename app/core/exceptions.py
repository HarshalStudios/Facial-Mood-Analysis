"""
Central exception hierarchy for the facial mood analysis backend.

Every subsystem raises one of these instead of letting raw exceptions
(KeyError, ValueError, cv2 errors, etc.) leak out of module boundaries.
This keeps error handling predictable across phases and lets the API
layer (Phase 10) translate exceptions into clean HTTP responses without
needing to know about implementation details of each subsystem.
"""


class BackendError(Exception):
    """Base class for all application-raised errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(BackendError):
    """Raised when required configuration is missing or invalid."""


class CameraError(BackendError):
    """Raised on webcam initialization, capture, or read failures."""


class DetectionError(BackendError):
    """Raised when face detection fails or produces invalid results."""


class RepresentationError(BackendError):
    """Raised when generating an image representation (CLAHE, LBP, etc.) fails."""


class FeatureExtractionError(BackendError):
    """Raised when building the feature vector fails or produces bad shape/values."""


class DatasetError(BackendError):
    """Raised when locating, parsing, or validating a dataset (e.g. FER2013) fails."""


class ModelLoadError(BackendError):
    """Raised when a model artifact (fusion model, scaler, emotion model) fails to load."""


class PredictionError(BackendError):
    """Raised when the fusion model or emotion model fails during inference."""
