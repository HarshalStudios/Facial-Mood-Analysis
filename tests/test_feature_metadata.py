from app.schemas.feature_vector import (
    FEATURE_COUNT,
    FEATURE_INDICES,
    FEATURE_METADATA,
    FEATURE_NAMES,
    FEATURE_VECTOR_LENGTH,
)


def test_feature_count_matches_length():
    assert FEATURE_COUNT == FEATURE_VECTOR_LENGTH == 22


def test_feature_indices_cover_0_to_21_with_no_gaps():
    assert set(FEATURE_INDICES.values()) == set(range(22))
    for name, index in FEATURE_INDICES.items():
        assert FEATURE_NAMES[index] == name


def test_every_feature_has_metadata():
    assert set(FEATURE_METADATA.keys()) == set(FEATURE_NAMES)
    for name, meta in FEATURE_METADATA.items():
        assert meta.name == name
        assert meta.index == FEATURE_INDICES[name]
        assert meta.source
        assert meta.description
        assert meta.expected_range


def test_emotion_features_marked_normalized():
    for name in FEATURE_NAMES[0:7]:
        assert FEATURE_METADATA[name].normalized is True


def test_lbp_features_marked_normalized():
    for name in FEATURE_NAMES[7:17]:
        assert FEATURE_METADATA[name].normalized is True


def test_dip_quality_features_have_correct_sources():
    assert "Canny" in FEATURE_METADATA["edge_density"].source
    assert "Grayscale" in FEATURE_METADATA["brightness"].source
    assert "Grayscale" in FEATURE_METADATA["contrast"].source
    assert "Laplacian" in FEATURE_METADATA["sharpness"].source
