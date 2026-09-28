import numpy as np
import pytest

from app.schemas.feature_vector import (
    FEATURE_NAMES,
    FEATURE_VECTOR_LENGTH,
    FeatureVector,
)


def test_feature_names_length_matches_contract():
    assert len(FEATURE_NAMES) == FEATURE_VECTOR_LENGTH == 22


def test_feature_names_are_unique_and_ordered_as_documented():
    assert FEATURE_NAMES[0] == "emotion_angry"
    assert FEATURE_NAMES[6] == "emotion_neutral"
    assert FEATURE_NAMES[7] == "lbp_bin_0"
    assert FEATURE_NAMES[16] == "lbp_bin_9"
    assert FEATURE_NAMES[17:] == [
        "edge_density",
        "gradient_energy",
        "brightness",
        "contrast",
        "sharpness",
    ]
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_feature_vector_rejects_wrong_shape():
    with pytest.raises(ValueError):
        FeatureVector(values=np.zeros(10))


def test_feature_vector_accepts_correct_shape_and_named_access():
    fv = FeatureVector(values=np.arange(22, dtype=float))
    assert fv["sharpness"] == 21.0
    assert fv.as_dict()["emotion_happy"] == 3.0
