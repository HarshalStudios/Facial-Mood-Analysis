import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import FeatureExtractionError, PredictionError
from app.emotion.mock_provider import FixedEmotionModel, FunctionEmotionModel
from app.emotion.validation import CANONICAL_EMOTION_LABELS
from app.features.extractor import StandardFeatureExtractor
from app.representations.engine import RepresentationEngine
from app.schemas.feature_vector import FEATURE_VECTOR_LENGTH


def make_test_image(size: int = 96) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    gradient = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    for c in range(3):
        image[:, :, c] = gradient
    image[20:40, 20:60, :] = 220
    yy, xx = np.ogrid[:size, :size]
    circle_mask = (xx - 70) ** 2 + (yy - 70) ** 2 <= 15**2
    image[circle_mask] = (30, 30, 30)
    return image


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def representations(settings):
    return RepresentationEngine(settings=settings).generate(make_test_image())


@pytest.fixture
def fixed_probabilities():
    return {
        "angry": 0.01,
        "disgust": 0.01,
        "fear": 0.01,
        "happy": 0.9,
        "sad": 0.02,
        "surprise": 0.03,
        "neutral": 0.02,
    }


def test_produces_a_vector_of_length_22(settings, representations, fixed_probabilities):
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    result = extractor.extract(representations)
    assert result.vector.values.shape == (FEATURE_VECTOR_LENGTH,)


def test_all_values_finite(settings, representations, fixed_probabilities):
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    result = extractor.extract(representations)
    assert np.all(np.isfinite(result.vector.values))


def test_emotion_values_land_at_correct_indices(settings, representations, fixed_probabilities):
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    result = extractor.extract(representations)
    assert result.vector["emotion_happy"] == pytest.approx(0.9)
    assert result.vector["emotion_angry"] == pytest.approx(0.01)


def test_lbp_bins_sum_to_approximately_one(settings, representations, fixed_probabilities):
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    result = extractor.extract(representations)
    lbp_sum = sum(result.features[f"lbp_bin_{i}"] for i in range(10))
    assert lbp_sum == pytest.approx(1.0)


def test_edge_density_in_expected_range(settings, representations, fixed_probabilities):
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    result = extractor.extract(representations)
    assert 0.0 <= result.features["edge_density"] <= 1.0


def test_brightness_in_uint8_range(settings, representations, fixed_probabilities):
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    result = extractor.extract(representations)
    assert 0.0 <= result.features["brightness"] <= 255.0


def test_deterministic_for_fixed_input_and_provider(settings, representations, fixed_probabilities):
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    first = extractor.extract(representations).vector.values
    second = extractor.extract(representations).vector.values
    np.testing.assert_array_equal(first, second)


def test_uses_function_emotion_model_output(settings, representations):
    def fn(face_crop):
        return {label: 1.0 / 7.0 for label in CANONICAL_EMOTION_LABELS}

    extractor = StandardFeatureExtractor(FunctionEmotionModel(fn), settings)
    result = extractor.extract(representations)
    for label in CANONICAL_EMOTION_LABELS:
        assert result.vector[f"emotion_{label}"] == pytest.approx(1.0 / 7.0)


def test_missing_lbp_representation_raises(settings, fixed_probabilities):
    settings.enable_lbp = False
    reps = RepresentationEngine(settings=settings).generate(make_test_image())
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    with pytest.raises(FeatureExtractionError):
        extractor.extract(reps)


def test_missing_canny_representation_raises(settings, fixed_probabilities):
    settings.enable_canny = False
    reps = RepresentationEngine(settings=settings).generate(make_test_image())
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    with pytest.raises(FeatureExtractionError):
        extractor.extract(reps)


def test_gradient_energy_works_even_when_sobel_representation_disabled(
    settings, fixed_probabilities
):
    settings.enable_sobel = False
    reps = RepresentationEngine(settings=settings).generate(make_test_image())
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    result = extractor.extract(reps)  # must not raise
    assert np.isfinite(result.features["gradient_energy"])


def test_invalid_emotion_provider_output_propagates_prediction_error(settings, representations):
    bad_provider = FixedEmotionModel({"happy": 1.0})  # missing 6 labels
    extractor = StandardFeatureExtractor(bad_provider, settings)
    with pytest.raises(PredictionError):
        extractor.extract(representations)


def test_result_features_dict_matches_vector(settings, representations, fixed_probabilities):
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    result = extractor.extract(representations)
    assert result.features == result.vector.as_dict()


def test_metadata_records_emotion_provider_type(settings, representations, fixed_probabilities):
    extractor = StandardFeatureExtractor(FixedEmotionModel(fixed_probabilities), settings)
    result = extractor.extract(representations)
    assert result.metadata["emotion_provider"] == "FixedEmotionModel"
