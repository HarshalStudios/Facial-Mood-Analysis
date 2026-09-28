import numpy as np
import pytest

from app.core.exceptions import DatasetError
from app.dataset.preprocessing import parse_pixel_string, to_representation_input


def test_parse_pixel_string_correct_shape():
    pixels = " ".join(str(i % 256) for i in range(16))
    image = parse_pixel_string(pixels, width=4, height=4)
    assert image.shape == (4, 4)
    assert image.dtype == np.uint8


def test_parse_pixel_string_wrong_token_count_raises():
    pixels = " ".join(str(i) for i in range(10))  # not 16
    with pytest.raises(DatasetError):
        parse_pixel_string(pixels, width=4, height=4)


def test_parse_pixel_string_non_integer_token_raises():
    pixels = "1 2 three 4 " * 4
    with pytest.raises(DatasetError):
        parse_pixel_string(pixels, width=4, height=4)


def test_parse_pixel_string_out_of_range_value_raises():
    values = [0] * 15 + [300]
    pixels = " ".join(str(v) for v in values)
    with pytest.raises(DatasetError):
        parse_pixel_string(pixels, width=4, height=4)


def test_grayscale_replicated_to_three_channels():
    gray = np.random.default_rng(0).integers(0, 255, size=(48, 48), dtype=np.uint8)
    bgr = to_representation_input(gray)
    assert bgr.shape == (48, 48, 3)
    assert bgr.dtype == np.uint8
    # Replication means every channel equals the original grayscale value.
    assert np.array_equal(bgr[:, :, 0], gray)
    assert np.array_equal(bgr[:, :, 1], gray)
    assert np.array_equal(bgr[:, :, 2], gray)


def test_three_channel_input_passed_through():
    rgb = np.random.default_rng(0).integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
    out = to_representation_input(rgb)
    assert np.array_equal(out, rgb)


def test_unsupported_shape_raises():
    bad = np.zeros((48, 48, 4), dtype=np.uint8)
    with pytest.raises(DatasetError):
        to_representation_input(bad)


def test_none_raises():
    with pytest.raises(DatasetError):
        to_representation_input(None)
