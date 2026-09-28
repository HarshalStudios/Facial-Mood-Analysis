"""
Debug display for a FeatureExtractionResult (Phase 4 spec sec. 33).

Dev/debugging utility only — not used by the extraction pipeline
itself. Useful when wiring up Phase 5/6.
"""

from __future__ import annotations

from app.schemas.feature_extraction import FeatureExtractionResult
from app.schemas.feature_vector import FEATURE_NAMES


def format_feature_vector(result: FeatureExtractionResult) -> str:
    lines = ["FEATURE VECTOR", "-" * 50]

    for i, name in enumerate(FEATURE_NAMES[0:7]):
        lines.append(f"{i:<3}{name:<20}= {result.features[name]:.4f}")
    lines.append("")
    for i, name in enumerate(FEATURE_NAMES[7:17], start=7):
        lines.append(f"{i:<3}{name:<20}= {result.features[name]:.4f}")
    lines.append("")
    for i, name in enumerate(FEATURE_NAMES[17:22], start=17):
        lines.append(f"{i:<3}{name:<20}= {result.features[name]:.4f}")

    lines.append("")
    lines.append(f"Length: {len(result.vector.values)}")
    lines.append(f"Valid:  {len(result.vector.values) == 22}")
    return "\n".join(lines)


def print_feature_vector(result: FeatureExtractionResult) -> None:
    print(format_feature_vector(result))
