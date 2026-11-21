"""Corrected offline review evidence tier with verified-photo precedence.

Version 1 remains unchanged for comparison and rollback.  This module records
a candidate tier only; it never mutates the existing RTI or runtime level.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.analyzers.network import NetworkEvidenceState
from app.scoring.meta_scorer import classify_rti
from app.training.p_text_v3.review_evidence_tier_v1 import (
    EvidenceValue,
    ReviewEvidenceInput,
    detect_experience_evidence,
    detect_promotional_evidence,
    meaningful_character_count,
)


POLICY_NAME = "review_evidence_tier_v2"


class ImageEvidenceState(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ReviewEvidenceTierV2Result:
    policy_name: str
    original_rti: float
    original_level: str
    candidate_level_v2: str
    candidate_reasons_v2: tuple[str, ...]
    grade_conflict: bool
    meaningful_character_count: int
    image_evidence_state: str
    verified_rating_five: bool
    experience_evidence: str
    promotional_evidence: str
    strong_risk_reasons: tuple[str, ...]


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


def evaluate_review_evidence_tier_v2(data: ReviewEvidenceInput) -> ReviewEvidenceTierV2Result:
    count = meaningful_character_count(data.content)
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
    reasons: list[str] = [f"IMAGE_{image_state.value}_VERIFIED" if image_state is not ImageEvidenceState.UNKNOWN else "IMAGE_COUNT_UNKNOWN"]
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
    elif image_state is ImageEvidenceState.PRESENT:
        if strong_risks:
            candidate = "WARN"
            reasons.append("PHOTO_SAFE_BLOCKED_BY_STRONG_RISK")
        elif promotional.value is EvidenceValue.NO:
            candidate = "SAFE"
            reasons.append("VERIFIED_PHOTO_AND_NO_PROMOTIONAL_STYLE")
        elif promotional.value is EvidenceValue.UNCERTAIN:
            candidate = "WARN"
            reasons.append("PHOTO_SAFE_BLOCKED_BY_PROMOTIONAL_UNCERTAINTY")
        else:
            candidate = "WARN"
            reasons.append("PHOTO_SAFE_BLOCKED_BY_PROMOTIONAL_STYLE")
    elif (
        image_state is ImageEvidenceState.ABSENT
        and verified_five
        and count < 10
    ):
        candidate = "DANGER"
        reasons.append("VERIFIED_NO_PHOTO_ULTRASHORT_FIVE_STAR")
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
        if image_state is ImageEvidenceState.UNKNOWN:
            reasons.append("PHOTO_EVIDENCE_UNKNOWN")
        elif promotional.value is EvidenceValue.YES:
            reasons.append("PROMOTIONAL_STYLE_FORCES_WARN")
        elif promotional.value is EvidenceValue.UNCERTAIN:
            reasons.append("PROMOTIONAL_UNCERTAINTY_FORCES_WARN")
        else:
            reasons.append("SAFE_REQUIREMENTS_INCOMPLETE")

    return ReviewEvidenceTierV2Result(
        policy_name=POLICY_NAME,
        original_rti=float(data.existing_rti_score),
        original_level=original_level,
        candidate_level_v2=candidate,
        candidate_reasons_v2=tuple(dict.fromkeys(reasons)),
        grade_conflict=original_level != candidate,
        meaningful_character_count=count,
        image_evidence_state=image_state.value,
        verified_rating_five=verified_five,
        experience_evidence=experience.value.value,
        promotional_evidence=promotional.value.value,
        strong_risk_reasons=strong_risks,
    )
