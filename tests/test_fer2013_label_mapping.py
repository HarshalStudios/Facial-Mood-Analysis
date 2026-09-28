import pytest

from app.dataset.label_mapping import UnmappableLabelError, parse_fer2013_label
from app.emotion.validation import CANONICAL_EMOTION_LABELS


@pytest.mark.parametrize("idx,label", list(enumerate(CANONICAL_EMOTION_LABELS)))
def test_all_seven_canonical_numeric_ids(idx, label):
    parsed_label, parsed_id = parse_fer2013_label(idx)
    assert parsed_label == label
    assert parsed_id == idx


@pytest.mark.parametrize("idx,label", list(enumerate(CANONICAL_EMOTION_LABELS)))
def test_numeric_ids_as_strings(idx, label):
    parsed_label, parsed_id = parse_fer2013_label(str(idx))
    assert parsed_label == label
    assert parsed_id == idx


@pytest.mark.parametrize("name,expected", [("Happy", "happy"), ("SURPRISED", "surprise"), (" sad ", "sad")])
def test_string_labels_and_synonyms(name, expected):
    parsed_label, parsed_id = parse_fer2013_label(name)
    assert parsed_label == expected
    assert parsed_id == CANONICAL_EMOTION_LABELS.index(expected)


def test_out_of_range_integer_is_unmappable():
    with pytest.raises(UnmappableLabelError):
        parse_fer2013_label(7)


def test_unrecognized_string_is_unmappable():
    with pytest.raises(UnmappableLabelError):
        parse_fer2013_label("ecstatic")


def test_none_is_unmappable():
    with pytest.raises(UnmappableLabelError):
        parse_fer2013_label(None)


def test_empty_string_is_unmappable():
    with pytest.raises(UnmappableLabelError):
        parse_fer2013_label("   ")
