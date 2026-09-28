from pathlib import Path
import pandas as pd

from app.dataset.fer2013_loader import FER2013Loader


def test_icml_face_data_style_headers_are_accepted(tmp_path: Path):
    pixels = " ".join(["0"] * 2304)
    path = tmp_path / "icml_face_data.csv"
    pd.DataFrame([
        {"emotion": 0, " Usage": "Training", " pixels": pixels},
    ]).to_csv(path, index=False)
    loaded = FER2013Loader().load(path)
    assert len(loaded.samples) == 1
    assert loaded.samples[0].raw_official_split == "Training"
    assert loaded.schema.has_official_split_column is True
