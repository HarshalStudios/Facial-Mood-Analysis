"""
FER2013 label parsing -> canonical label (Phase 5 spec sec. 6).

The project has exactly one canonical emotion order, already defined in
app.emotion.validation.CANONICAL_EMOTION_LABELS (used for the deep-emotion
probabilities in the 22D feature vector). This module maps whatever label
representation the supplied FER2013 file actually uses onto that same
list, rather than inventing a second, parallel mapping.

Do not assume the numeric encoding blindly: this module validates that any
integer label falls in range and documents that FER2013's conventional
0..6 = angry..neutral ordering happens to already match
CANONICAL_EMOTION_LABELS, rather than hard-coding that coincidence as if
it were guaranteed.
"""

from __future__ import annotations

from app.emotion.validation import CANONICAL_EMOTION_LABELS

# FER2013's traditional numeric encoding (0=Angry .. 6=Neutral) matches
# CANONICAL_EMOTION_LABELS index-for-index. This constant makes that
# assumption explicit and inspectable, rather than silently relying on
# list order matching by coincidence in two different files.
FER2013_CANONICAL_NUMERIC_ORDER: list[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]
assert FER2013_CANONICAL_NUMERIC_ORDER == CANONICAL_EMOTION_LABELS, (
    "FER2013's documented numeric label order no longer matches "
    "app.emotion.validation.CANONICAL_EMOTION_LABELS -- update the mapping "
    "in app.dataset.label_mapping deliberately, do not just silence this."
)

# Common string spellings seen across different FER2013 redistributions /
# CSV exports, normalized (lower-cased, stripped) -> canonical label.
# Extend this table if a supplied dataset uses another spelling; do not
# guess-map an unrecognized string.
_STRING_SYNONYMS: dict[str, str] = {
    "angry": "angry",
    "anger": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "scared": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
    "calm": "neutral",
}


class UnmappableLabelError(ValueError):
    """Raised when a raw label value cannot be mapped to a canonical emotion."""


def parse_fer2013_label(raw_label) -> tuple[str, int]:
    """
    Map one raw FER2013 label value (int, numeric string, or emotion name)
    onto (canonical_label, canonical_label_id).

    Raises UnmappableLabelError for anything that isn't a recognized
    integer 0-6 or a recognized emotion name/synonym. Callers (the loader)
    are expected to catch this per-row and record it as a malformed
    record rather than letting one bad label abort the whole dataset.
    """
    if raw_label is None:
        raise UnmappableLabelError("label is None")

    text = str(raw_label).strip()
    if text == "":
        raise UnmappableLabelError("label is empty")

    # Try integer encoding first (FER2013's conventional format).
    try:
        as_int = int(text)
    except ValueError:
        as_int = None

    if as_int is not None:
        if 0 <= as_int < len(CANONICAL_EMOTION_LABELS):
            return CANONICAL_EMOTION_LABELS[as_int], as_int
        raise UnmappableLabelError(
            f"integer label {as_int} is outside the valid range "
            f"0-{len(CANONICAL_EMOTION_LABELS) - 1}"
        )

    # Fall back to a recognized emotion name/synonym.
    normalized = text.lower()
    canonical = _STRING_SYNONYMS.get(normalized)
    if canonical is None:
        raise UnmappableLabelError(f"unrecognized label value: {raw_label!r}")
    return canonical, CANONICAL_EMOTION_LABELS.index(canonical)
