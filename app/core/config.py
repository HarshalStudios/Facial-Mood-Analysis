"""
Centralized configuration.

Every tunable value used anywhere in the backend must be read from here,
never hard-coded inline in a module. Values are sourced from environment
variables (see .env.example) with sensible defaults for local development.

Later phases (camera, detection, representations, fusion, temporal, API)
add their settings to this same object rather than creating their own
ad-hoc config files, so there is exactly one source of truth.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FMA_",
        extra="ignore",
    )

    # --- General -----------------------------------------------------
    app_name: str = "facial-mood-analysis-backend"
    environment: str = Field(default="development")  # development | production | test

    # --- Logging -------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=Path("outputs/logs"))

    # --- API -----------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Camera (used starting Phase 2) --------------------------------
    camera_index: int = Field(default=0)
    camera_width: int = Field(default=640)
    camera_height: int = Field(default=480)
    camera_target_fps: int = Field(default=30)

    # --- Face detection (Phase 2) ---------------------------------------
    face_detector_backend: str = Field(default="opencv_haar")
    face_min_confidence: float = Field(default=0.5)
    face_padding_ratio: float = Field(default=0.0)  # 0.10 = 10% margin around bbox
    min_face_width: int = Field(default=40)
    min_face_height: int = Field(default=40)
    primary_face_strategy: str = Field(default="largest")  # largest | highest_confidence | center_most
    haar_scale_factor: float = Field(default=1.1)
    haar_min_neighbors: int = Field(default=5)

    # --- Representations / DIP params (Phase 3) --------------------------
    # Which representations the RepresentationEngine computes. RGB is
    # always produced (it is just the validated face crop) and has no
    # toggle. Disabling one here is what Phase 9 ablation experiments
    # will use to run e.g. "RGB + CLAHE only" without touching code.
    enable_grayscale: bool = Field(default=True)
    enable_clahe: bool = Field(default=True)
    enable_canny: bool = Field(default=True)
    enable_sobel: bool = Field(default=True)
    enable_lbp: bool = Field(default=True)

    clahe_clip_limit: float = Field(default=2.0)
    clahe_tile_grid_size: int = Field(default=8)

    canny_low_threshold: int = Field(default=100)
    canny_high_threshold: int = Field(default=200)
    canny_aperture_size: int = Field(default=3)  # must be odd, 3/5/7
    canny_l2_gradient: bool = Field(default=False)

    sobel_kernel_size: int = Field(default=3)  # must be odd
    sobel_scale: float = Field(default=1.0)
    sobel_delta: float = Field(default=0.0)

    lbp_radius: int = Field(default=1)
    lbp_n_points: int = Field(default=8)
    lbp_num_bins: int = Field(default=10)
    lbp_method: str = Field(default="uniform")  # passed straight to skimage

    # Where `export_representations` writes debug PNGs when explicitly
    # asked to (see app/representations/export.py). Never written to
    # automatically for every frame.
    representation_debug_dir: Path = Field(default=Path("outputs/representations"))

    # --- Feature vector contract (Phase 4) -------------------------------
    feature_vector_length: int = Field(default=22)

    # --- Emotion model (Phase 4/6) ---------------------------------------
    emotion_model_backend: str = Field(default="fer2013_linear")
    fer2013_emotion_model_path: Path = Field(default=Path("models/fer2013_emotion_cnn.pt"))
    fer2013_linear_emotion_model_path: Path = Field(default=Path("models/fer2013_emotion_linear.joblib"))
    emotion_labels: list[str] = Field(
        default_factory=lambda: [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]
    )

    # --- Fusion model (Phase 6) --------------------------------------------
    fusion_model_path: Path = Field(default=Path("models/fusion_model.pkl"))
    fusion_scaler_path: Path = Field(default=Path("models/scaler.pkl"))

    # --- Temporal smoothing (Phase 8) ---------------------------------------
    # All values below are initial engineering defaults, not experimentally
    # optimized numbers (Phase 8 spec sec 31/32) -- Phase 9's ablation engine
    # is what actually evaluates alternative settings against held-out data.
    temporal_enabled: bool = Field(default=True)
    smoothing_method: str = Field(default="moving_average")  # moving_average | ema | none
    smoothing_window_size: int = Field(default=5)
    ema_alpha: float = Field(default=0.3)

    # Consecutive no-face frames tolerated before temporal history is reset
    # (spec sec 10/11) so a stale emotion is never displayed indefinitely.
    max_missing_face_frames: int = Field(default=15)
    # Bounding-box IoU below which a newly-primary face is treated as a
    # different subject and temporal history is reset (spec sec 12).
    face_continuity_iou_threshold: float = Field(default=0.2)

    # --- Quality control (Phase 8) -------------------------------------------
    quality_enabled: bool = Field(default=True)
    quality_mode: str = Field(default="advisory")  # advisory | strict (spec sec 21)

    min_face_area_ratio: float = Field(default=0.05)  # face_area / frame_area
    min_brightness: float = Field(default=40.0)
    max_brightness: float = Field(default=220.0)
    min_sharpness: float = Field(default=60.0)  # variance of Laplacian
    min_contrast: float = Field(default=20.0)

    # --- FER2013 dataset / feature-dataset generation (Phase 5) --------------
    # Where the developer places the real, supplied FER2013 file (CSV with
    # emotion/pixels/Usage columns, or another schema the loader can detect —
    # see app/dataset/fer2013_loader.py). Never auto-downloaded, never
    # substituted with synthetic data (Phase 5 spec sec. 5).
    dataset_name: str = Field(default="FER2013")
    dataset_path: Path = Field(default=Path("dataset/fer2013.csv"))

    # Output locations (Phase 5 spec sec. 41).
    features_dir: Path = Field(default=Path("features"))  # -> {features_dir}/raw, {features_dir}/metadata
    dataset_failures_path: Path = Field(default=Path("outputs/dataset_failures.csv"))
    dataset_summary_path: Path = Field(default=Path("outputs/dataset_summary.txt"))

    # Deterministic stratified split, used only when the supplied dataset has
    # no official Training/PublicTest/PrivateTest (or equivalent) column to
    # preserve (Phase 5 spec sec. 16-17).
    dataset_split_seed: int = Field(default=42)
    dataset_train_ratio: float = Field(default=0.8)
    dataset_val_ratio: float = Field(default=0.1)
    dataset_test_ratio: float = Field(default=0.1)

    # Batch processing (Phase 5 spec sec. 27-29).
    dataset_checkpoint_batch_size: int = Field(default=200)  # rows between CSV flushes
    dataset_progress_log_every: int = Field(default=250)  # rows between progress log lines


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
