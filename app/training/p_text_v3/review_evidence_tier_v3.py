"""Offline review evidence tier v3 with promotional DANGER and length signals.

Versions 1 and 2 remain unchanged.  The optional length-penalty helpers are
strictly offline sensitivity candidates and never mutate the operating RTI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from app.analyzers.network import NetworkEvidenceState
from app.scoring.meta_scorer import classify_rti
from app.training.p_text_v3.review_evidence_tier_v1 import (
    EvidenceValue,
    ReviewEvidenceInput,
    detect_experience_evidence,
    detect_promotional_evidence,
    meaningful_character_count,
)
from app.training.p_text_v3.review_evidence_tier_v2 import ImageEvidenceState


POLICY_NAME = "review_evidence_tier_v3"


class ShortTextPenaltyBand(str, Enum):
    STRONG = "STRONG_0_9"
    GENERAL = "GENERAL_10_29"
    NONE = "NONE_30_PLUS"


@dataclass(frozen=True, slots=True)
class ReviewEvidenceTierV3Result:
    policy_name: str
    original_rti: float
    original_level: str
    candidate_level_v3: str
    candidate_reasons_v3: tuple[str, ...]
    grade_conflict: bool
    meaningful_character_count: int
    short_text_penalty_band: str
    short_text_reason: str
    image_evidence_state: str
    verified_rating_five: bool
    experience_evidence: str
    promotional_evidence: str
    strong_risk_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LengthPenaltyScore:
    original_rti: float
    meaningful_character_count: int
    maximum_penalty: float
    length_deficit_ratio: float
    applied_penalty: float
    policy_score_candidate: float
    score_level: str


def short_text_penalty(count: int) -> tuple[ShortTextPenaltyBand, str]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("meaningful character count must be a non-negative integer")
    if count < 10:
        return ShortTextPenaltyBand.STRONG, "MEANINGFUL_TEXT_0_9_STRONG_PENALTY"
    if count < 30:
        return ShortTextPenaltyBand.GENERAL, "MEANINGFUL_TEXT_10_29_GENERAL_PENALTY"
    return ShortTextPenaltyBand.NONE, "MEANINGFUL_TEXT_30_PLUS_NO_PENALTY"


def apply_length_penalty_candidate(
    original_rti: float, meaningful_count: int, *, maximum_penalty: float,
) -> LengthPenaltyScore:
    if (
        isinstance(original_rti, bool) or not isinstance(original_rti, (int, float))
        or not isfinite(original_rti) or not 0 <= original_rti <= 100
    ):
        raise ValueError("original RTI must be finite and between 0 and 100")
    if (
        isinstance(maximum_penalty, bool)
        or not isinstance(maximum_penalty, (int, float))
        or not isfinite(maximum_penalty)
        or maximum_penalty <= 0
    ):
        raise ValueError("maximum penalty must be positive and finite")
    short_text_penalty(meaningful_count)
    deficit = max(0.0, 30.0 - meaningful_count) / 30.0
    penalty = maximum_penalty * deficit
    score = max(0.0, float(original_rti) - penalty)
    return LengthPenaltyScore(
        original_rti=float(original_rti),
        meaningful_character_count=meaningful_count,
        maximum_penalty=float(maximum_penalty),
        length_deficit_ratio=round(deficit, 6),
        applied_penalty=round(penalty, 6),
        policy_score_candidate=round(score, 6),
        score_level=classify_rti(score).value.upper(),
    )


def _strong_risks(data: ReviewEvidenceInput) -> tuple[str, ...]:
    reasons: list[str] = []
    if data.text_predicted_label == "SUSPICIOUS":
        reasons.append("TEXT_PREDICTION_SUSPICIOUS")
    state = (
        data.network_evidence_state.value
        if isinstance(data.network_evidence_state, NetworkEvidenceState)
        else data.network_evidence_state
    )
    if state in {
        NetworkEvidenceState.STRONG_PAIR.value,
        NetworkEvidenceState.STRONG_CLUSTER.value,
    }:
        reasons.append("STRONG_NETWORK_SIMILARITY")
    return tuple(reasons)


def evaluate_review_evidence_tier_v3(data: ReviewEvidenceInput) -> ReviewEvidenceTierV3Result:
    count = meaningful_character_count(data.content)
    penalty_band, penalty_reason = short_text_penalty(count)
    experience = detect_experience_evidence(data.content)
    promotional = detect_promotional_evidence(data.content)
    strong_risks = _strong_risks(data)
    original_level = classify_rti(data.existing_rti_score).value.upper()
    image_state = (
        ImageEvidenceState.UNKNOWN if data.image_count is None
        else ImageEvidenceState.PRESENT if data.image_count >= 1
        else ImageEvidenceState.ABSENT
    )
    verified_five = data.rating_scale_max == 5 and data.rating == 5
    verified_below_five = (
        data.rating_scale_max == 5 and data.rating is not None and data.rating < 5
    )
    reasons: list[str] = [penalty_reason]
    reasons.append(
        f"IMAGE_{image_state.value}_VERIFIED"
        if image_state is not ImageEvidenceState.UNKNOWN else "IMAGE_COUNT_UNKNOWN"
    )
    if verified_five:
        reasons.append("RATING_FIVE_VERIFIED")
    elif verified_below_five:
        reasons.append("RATING_BELOW_FIVE_VERIFIED")
    elif data.rating is None:
        reasons.append("RATING_UNKNOWN")
    else:
        reasons.append("RATING_SCALE_UNVERIFIED")
    reasons.append(f"EXPERIENCE_EVIDENCE_{experience.value.value}")
    if promotional.value is EvidenceValue.YES:
        reasons.append("PROMOTIONAL_STYLE_DETECTED")
    elif promotional.value is EvidenceValue.UNCERTAIN:
        reasons.append("PROMOTIONAL_STYLE_UNCERTAIN")
    else:
        reasons.append("PROMOTIONAL_STYLE_NOT_DETECTED")
    reasons.extend(strong_risks)

    if data.existing_rti_score < 40:
        candidate = "DANGER"
        reasons.append("ORIGINAL_RTI_DANGER_PRESERVED")
    elif promotional.value is EvidenceValue.YES:
        candidate = "DANGER"
        reasons.append("PROMOTIONAL_STYLE_DANGER_CANDIDATE")
    elif (
        image_state is ImageEvidenceState.ABSENT and verified_five and count < 10
    ):
        candidate = "DANGER"
        reasons.append("VERIFIED_NO_PHOTO_ULTRASHORT_FIVE_STAR")
    elif image_state is ImageEvidenceState.PRESENT:
        if strong_risks:
            candidate = "WARN"
            reasons.append("PHOTO_SAFE_BLOCKED_BY_STRONG_RISK")
        elif promotional.value is EvidenceValue.NO:
            candidate = "SAFE"
            reasons.append("VERIFIED_PHOTO_AND_NO_PROMOTIONAL_STYLE")
        else:
            candidate = "WARN"
            reasons.append("PHOTO_SAFE_BLOCKED_BY_PROMOTIONAL_UNCERTAINTY")
    elif strong_risks:
        candidate = "WARN"
        reasons.append("DETAILED_SAFE_BLOCKED_BY_STRONG_RISK")
    elif (
        image_state is ImageEvidenceState.ABSENT
        and count >= 30
        and verified_below_five
        and experience.value is EvidenceValue.YES
        and promotional.value is EvidenceValue.NO
    ):
        candidate = "SAFE"
        reasons.append("NO_PHOTO_DETAILED_EXPERIENCE_SAFE")
    else:
        candidate = "WARN"
        if promotional.value is EvidenceValue.UNCERTAIN:
            reasons.append("PROMOTIONAL_UNCERTAINTY_FORCES_WARN")
        elif image_state is ImageEvidenceState.UNKNOWN:
            reasons.append("PHOTO_EVIDENCE_UNKNOWN")
        else:
            reasons.append("SAFE_REQUIREMENTS_INCOMPLETE")

    return ReviewEvidenceTierV3Result(
        policy_name=POLICY_NAME,
        original_rti=float(data.existing_rti_score),
        original_level=original_level,
        candidate_level_v3=candidate,
        candidate_reasons_v3=tuple(dict.fromkeys(reasons)),
        grade_conflict=original_level != candidate,
        meaningful_character_count=count,
        short_text_penalty_band=penalty_band.value,
        short_text_reason=penalty_reason,
        image_evidence_state=image_state.value,
        verified_rating_five=verified_five,
        experience_evidence=experience.value.value,
        promotional_evidence=promotional.value.value,
        strong_risk_reasons=strong_risks,
    )
