"""Validate backend readiness without requiring a webcam or model load."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.experiments.runner import validate_artifacts


def main() -> int:
    settings = get_settings()
    print("Backend readiness")
    print("=================")
    print(f"environment: {settings.environment}")
    print(f"feature schema: {settings.feature_vector_length}")
    print(f"model: {settings.fusion_model_path}")
    print(f"scaler: {settings.fusion_scaler_path}")
    print()
    gate = validate_artifacts(settings)
    print("Phase 9 feature artifacts:")
    print(json.dumps(gate, indent=2))
    print()
    model_ready = Path(settings.fusion_model_path).exists() and Path(settings.fusion_scaler_path).exists()
    print(f"Model artifacts ready: {model_ready}")
    print(f"Phase 9 data ready: {gate['ready']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
