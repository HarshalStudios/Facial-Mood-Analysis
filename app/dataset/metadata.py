"""
Metadata generation (Phase 5 spec sec. 21, 30).

Every value written here must be a real, observed value from this run --
never a placeholder or an invented default. Callers pass in exactly what
they measured; this module only assembles and serializes it.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import skimage

from app.core.config import Settings
from app.dataset.preprocessing import GRAYSCALE_TO_BGR_STRATEGY
from app.schemas.dataset import DatasetSchemaInfo
from app.schemas.feature_vector import FEATURE_METADATA, FEATURE_NAMES, FEATURE_VECTOR_VERSION

FEATURE_SCHEMA_VERSION = "1.0"
PREPROCESSING_VERSION = "fer2013-phase5-v1"


def build_dataset_metadata(
    *,
    settings: Settings,
    schema: DatasetSchemaInfo,
    split_strategy: str,
    split_details: dict,
) -> dict:
    return {
        "dataset": settings.dataset_name,
        "dataset_source_path": schema.source_path,
        "feature_count": len(FEATURE_NAMES),
        "classes": 7,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_vector_version": FEATURE_VECTOR_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "grayscale_to_bgr_strategy": GRAYSCALE_TO_BGR_STRATEGY,
        "representation_config": {
            "enable_grayscale": settings.enable_grayscale,
            "enable_clahe": settings.enable_clahe,
            "enable_canny": settings.enable_canny,
            "enable_sobel": settings.enable_sobel,
            "enable_lbp": settings.enable_lbp,
            "clahe_clip_limit": settings.clahe_clip_limit,
            "clahe_tile_grid_size": settings.clahe_tile_grid_size,
            "canny_low_threshold": settings.canny_low_threshold,
            "canny_high_threshold": settings.canny_high_threshold,
            "sobel_kernel_size": settings.sobel_kernel_size,
            "lbp_radius": settings.lbp_radius,
            "lbp_n_points": settings.lbp_n_points,
            "lbp_num_bins": settings.lbp_num_bins,
            "lbp_method": settings.lbp_method,
        },
        "emotion_model_backend": settings.emotion_model_backend,
        "dataset_schema": asdict(schema),
        "split_strategy": split_strategy,
        "split_details": split_details,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_image": skimage.__version__,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_feature_schema_metadata() -> dict:
    return {
        "feature_vector_version": FEATURE_VECTOR_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "feature_order": FEATURE_NAMES,
        "features": {name: asdict(meta) for name, meta in FEATURE_METADATA.items()},
    }


def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
