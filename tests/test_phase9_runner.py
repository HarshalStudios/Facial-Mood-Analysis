from pathlib import Path

from app.core.config import Settings
from app.experiments.runner import validate_artifacts


def test_phase9_artifact_gate_blocks_missing_data(tmp_path: Path):
    settings = Settings(
        features_dir=tmp_path / "features",
        dataset_path=tmp_path / "fer2013.csv",
    )
    gate = validate_artifacts(settings)
    assert gate["ready"] is False
    assert len(gate["missing"]) == 3
