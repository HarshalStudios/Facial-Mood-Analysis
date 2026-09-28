"""
Quality-control result contract (Phase 8 spec sec 13-21).

`QualityResult` is what `app.quality.base.QualityAnalyzer.analyze()`
returns. It deliberately separates two different things that must
never be conflated (spec sec 20):

    is_acceptable   -- an engineering *policy* decision (see
                       `quality_mode` in app.core.config) about whether
                       a prediction should be treated as usable.
    warnings        -- the underlying *observations* (too dark, too
                       blurry, ...) that a caller can inspect even when
                       is_acceptable is True, so a quality problem is
                       never silently hidden behind a single boolean.

`quality_score`, when present, is an explicit engineering heuristic
(spec sec 19) -- not a calibrated or scientifically validated
confidence measure, and must never be presented as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class QualityWarning:
    """Enum-like string constants (spec sec 18) -- never build these ad hoc."""

    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    LOW_BRIGHTNESS = "LOW_BRIGHTNESS"
    HIGH_BRIGHTNESS = "HIGH_BRIGHTNESS"
    LOW_SHARPNESS = "LOW_SHARPNESS"
    LOW_CONTRAST = "LOW_CONTRAST"
    NO_FACE = "NO_FACE"


# Which warnings are ever capable of making a prediction unavailable
# in "strict" quality mode (spec sec 21). This is an explicit,
# documented engineering choice, not a derived scientific threshold:
# a face too small to resolve real texture makes the entire 22D
# feature vector suspect, whereas brightness/contrast/sharpness issues
# degrade *some* signal but the model may still be usably confident
# (spec sec 20) -- so those remain advisory-only regardless of mode
# (spec sec 17: "do not automatically reject ... just because contrast
# is low", generalized to brightness/sharpness as well).
SEVERE_WARNINGS = frozenset({QualityWarning.FACE_TOO_SMALL, QualityWarning.NO_FACE})

VALID_QUALITY_MODES = frozenset({"advisory", "strict"})


@dataclass
class QualityResult:
    """Structured quality report for a single face/frame (spec sec 18)."""

    is_acceptable: bool
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    quality_score: float | None = None
    status: str = "good"  # "good" | "warning" | "reject"

    def to_json(self) -> dict:
        return {
            "is_acceptable": self.is_acceptable,
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
            "quality_score": self.quality_score,
            "status": self.status,
        }
