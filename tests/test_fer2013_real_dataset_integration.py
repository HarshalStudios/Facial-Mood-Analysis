"""
INTEGRATION TEST -- not a unit test.

Exercises FER2013Loader against the real, developer-supplied FER2013 file
at Settings.dataset_path. Skips itself when that file is not present,
rather than fabricating a pass/fail -- mirrors
tests/test_deepface_integration.py's pattern for the same reason (Phase 5
spec sec. 5: never substitute a missing dataset with something else, not
even in a test).

As of this Phase 5 implementation, this test has NOT been run against a
real FER2013 file in the development sandbox: no dataset was supplied
there (dataset/ only contains .gitkeep). Run this after placing the real
FER2013 CSV at the configured dataset_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.dataset.fer2013_loader import FER2013Loader

settings = get_settings()

if not Path(settings.dataset_path).exists():
    pytest.skip(
        f"Real FER2013 dataset not found at {settings.dataset_path} "
        f"(see module docstring); skipping integration test.",
        allow_module_level=True,
    )


def test_real_dataset_loads_with_few_or_no_malformed_rows():
    loaded = FER2013Loader().load(settings.dataset_path)
    assert len(loaded.samples) > 0
    malformed_ratio = len(loaded.malformed) / (len(loaded.samples) + len(loaded.malformed))
    assert malformed_ratio < 0.01, f"Unexpectedly high malformed-row ratio: {malformed_ratio:.4f}"


def test_real_dataset_has_seven_classes_represented():
    loaded = FER2013Loader().load(settings.dataset_path)
    labels = {s.label for s in loaded.samples}
    assert len(labels) == 7
