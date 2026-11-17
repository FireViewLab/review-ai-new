"""Offline policy indices aligned with ``review_evidence_tier_v3``.

The tier is an input decided by the preserved v3 policy.  This module only
places a deterministic ordinal index inside that tier and records a content
length deduction.  The index is not a probability, calibrated trust score,
or operating RTI replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from app.scoring.meta_scorer import classify_rti
from app.training.p_text_v3.review_evidence_tier_v3 import short_text_penalty


PolicyLevel = Literal["SAFE", "WARN", "DANGER"]
INTERPRETATION = "ORDINAL_V3_POLICY_INDEX_NOT_PROBABILITY"


class PolicyIndexV3Error(ValueError):
    """Fail-closed input/configuration error."""


@dataclass(frozen=True, slots=True)
class PolicyIndexV3Config:
    name: str
    danger_anchor: tuple[float, float]
    warn_anchor: tuple[float, float]
    safe_anchor: tuple[float, float]
    maximum_length_penalty: float


@dataclass(frozen=True, slots=True)
class PolicyIndexV3Result:
    method: str
    original_rti: float
    candidate_level_v3: PolicyLevel
    policy_index: float
    score_level: PolicyLevel
    level_consistent: bool
    base_index_before_length_penalty: float
    length_penalty: float
    length_deficit_ratio: float
    short_text_penalty_band: str
    boundary_clamped: bool
    interpretation: str = INTERPRETATION


GENTLE_WIDE = PolicyIndexV3Config(
    name="GENTLE_WIDE_V1",
    danger_anchor=(3.0, 37.0),
    warn_anchor=(43.0, 68.0),
    safe_anchor=(73.0, 98.0),
    maximum_length_penalty=4.0,
)

BALANCED_INTERIOR = PolicyIndexV3Config(
    name="BALANCED_INTERIOR_V1",
    danger_anchor=(5.0, 35.0),
    warn_anchor=(45.0, 65.0),
    safe_anchor=(75.0, 95.0),
    maximum_length_penalty=6.0,
)

CONSERVATIVE_LENGTH = PolicyIndexV3Config(
    name="CONSERVATIVE_LENGTH_V1",
    danger_anchor=(7.0, 35.0),
    warn_anchor=(47.0, 65.0),
    safe_anchor=(77.0, 95.0),
    maximum_length_penalty=8.0,
)

POLICY_INDEX_CANDIDATES = (
    GENTLE_WIDE,
    BALANCED_INTERIOR,
    CONSERVATIVE_LENGTH,
)


_BANDS: dict[str, tuple[float, float]] = {
    "DANGER": (0.0, 39.0),
    "WARN": (40.0, 69.0),
    "SAFE": (70.0, 100.0),
}


def _validate_config(config: PolicyIndexV3Config) -> None:
    if not config.name or not isfinite(config.maximum_length_penalty):
        raise PolicyIndexV3Error("CONFIG_INVALID")
    if config.maximum_length_penalty <= 0:
        raise PolicyIndexV3Error("MAXIMUM_LENGTH_PENALTY_INVALID")
    for level, anchor in (
        ("DANGER", config.danger_anchor),
        ("WARN", config.warn_anchor),
        ("SAFE", config.safe_anchor),
    ):
        lower, upper = anchor
        band_lower, band_upper = _BANDS[level]
        if not all(isfinite(value) for value in anchor):
            raise PolicyIndexV3Error("ANCHOR_NOT_FINITE")
        if not band_lower <= lower < upper <= band_upper:
            raise PolicyIndexV3Error(f"{level}_ANCHOR_OUTSIDE_TIER")


def _validate_inputs(
    original_rti: float, candidate_level_v3: str, meaningful_count: int,
) -> PolicyLevel:
    if (
        isinstance(original_rti, bool)
        or not isinstance(original_rti, (int, float))
        or not isfinite(original_rti)
        or not 0 <= original_rti <= 100
    ):
        raise PolicyIndexV3Error("ORIGINAL_RTI_INVALID")
    if candidate_level_v3 not in _BANDS:
        raise PolicyIndexV3Error("CANDIDATE_LEVEL_INVALID")
    if original_rti < 40 and candidate_level_v3 != "DANGER":
        raise PolicyIndexV3Error("ORIGINAL_DANGER_NOT_PRESERVED")
    short_text_penalty(meaningful_count)
    return candidate_level_v3  # type: ignore[return-value]


def _anchor(config: PolicyIndexV3Config, level: PolicyLevel) -> tuple[float, float]:
    return {
        "DANGER": config.danger_anchor,
        "WARN": config.warn_anchor,
        "SAFE": config.safe_anchor,
    }[level]


def policy_index_v3(
    original_rti: float,
    candidate_level_v3: str,
    meaningful_count: int,
    *,
    config: PolicyIndexV3Config = BALANCED_INTERIOR,
) -> PolicyIndexV3Result:
    """Project RTI order into a v3 tier, then deduct a length signal.

    ``base = anchor_low + (anchor_high - anchor_low) * original_rti / 100``
    ``deduction = max_penalty * max(0, 30 - length) / 30``

    The result is clamped only to the already-decided tier.  Thus a short
    photo-backed SAFE can receive a lower SAFE index without becoming WARN.
    """

    _validate_config(config)
    level = _validate_inputs(original_rti, candidate_level_v3, meaningful_count)
    anchor_lower, anchor_upper = _anchor(config, level)
    position = float(original_rti) / 100.0
    base = anchor_lower + (anchor_upper - anchor_lower) * position
    deficit = max(0.0, 30.0 - meaningful_count) / 30.0
    deduction = config.maximum_length_penalty * deficit
    unbounded = base - deduction
    band_lower, band_upper = _BANDS[level]
    score = min(max(unbounded, band_lower), band_upper)
    score = round(score, 6)
    score_level = classify_rti(score).value.upper()
    penalty_band, _ = short_text_penalty(meaningful_count)
    return PolicyIndexV3Result(
        method=config.name,
        original_rti=float(original_rti),
        candidate_level_v3=level,
        policy_index=score,
        score_level=score_level,  # type: ignore[arg-type]
        level_consistent=score_level == level,
        base_index_before_length_penalty=round(base, 6),
        length_penalty=round(deduction, 6),
        length_deficit_ratio=round(deficit, 6),
        short_text_penalty_band=penalty_band.value,
        boundary_clamped=not band_lower <= unbounded <= band_upper,
    )
