"""Deterministic offline review-evidence tier candidate policy.

This module never changes the existing RTI.  It records a separate candidate
level based on review evidence that is explicitly present in the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
import re
import unicodedata
from typing import Pattern

from app.analyzers.network import NetworkEvidenceState
from app.scoring.meta_scorer import classify_rti


POLICY_NAME = "review_evidence_tier_v1"


class EvidenceValue(str, Enum):
    YES = "YES"
    NO = "NO"
    UNCERTAIN = "UNCERTAIN"


class CandidateLevel(str, Enum):
    SAFE = "SAFE"
    WARN = "WARN"
    DANGER = "DANGER"


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    value: EvidenceValue
    context_hits: tuple[str, ...]
    detail_hits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceRuleConfig:
    experience_context_patterns: tuple[str, ...] = (
        r"(?:직접\s*)?(?:사용|착용|복용|설치|조립)(?:해|하|했|한|해\s*보|해봤|해보|하고|하며|중)",
        r"(?:직접\s*)?(?:써|입어|신어|먹어|마셔|발라|바르고)(?:\s*보|봤|보니|보니까|보았|난\s*후)",
        r"(?:구매|배송|수령)\s*(?:후|하고|받고)",
    )
    experience_detail_patterns: tuple[str, ...] = (
        r"(?:며칠|\d+\s*(?:일|주|개월|달|시간)|한\s*달|두\s*달)\s*(?:동안|째|간)?",
        r"(?:출근|퇴근|운동|여행|외출|회사|학교|집|요리|세탁|산책|잠들기\s*전|아침|오전|오후|저녁)(?:할|하|에서|때|용)",
        r"(?:발림성|지속력|착용감|밀착감|흡수력|보온성|크기|사이즈|무게|향|맛|질감|자극|건조|소음|배터리|세척|조립|마감|두께|핏|색감)",
        r"(?:장점|단점|아쉽|불편|편했|편하|하지만|다만|반면|처음에는|나중에는)",
        r"(?:사용|착용|복용|설치)\s*(?:전|후)|전보다|기존(?:에|의)?|다른\s*제품|전에\s*쓰던|보다\s*(?:좋|나쁘|편|불편)",
    )
    marketing_patterns: tuple[str, ...] = (
        r"혁신적인\s*기술력", r"독자적인\s*포뮬러", r"압도적인\s*성능",
        r"차별화된\s*솔루션", r"프리미엄\s*기술", r"획기적인\s*효과",
    )
    hype_patterns: tuple[str, ...] = (
        r"무조건\s*(?:구매|사야|추천)", r"단연\s*최고", r"절대\s*후회하지",
        r"완벽한\s*효과(?:를)?\s*보장", r"누구에게나\s*반드시\s*추천",
        r"사용하면\s*무조건\s*효과", r"반드시\s*효과를?\s*(?:봅니다|본다)",
    )


DEFAULT_RULE_CONFIG = EvidenceRuleConfig()


@dataclass(frozen=True, slots=True)
class ReviewEvidenceInput:
    content: str
    rating: float | None
    rating_scale_max: float | None
    image_count: int | None
    existing_rti_score: float
    text_predicted_label: str | None
    network_evidence_state: str | NetworkEvidenceState | None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if isinstance(self.existing_rti_score, bool) or not isinstance(
            self.existing_rti_score, (int, float)
        ):
            raise TypeError("existing_rti_score must be numeric")
        if not isfinite(self.existing_rti_score) or not 0 <= self.existing_rti_score <= 100:
            raise ValueError("existing_rti_score must be finite and between 0 and 100")
        for name, value in (("rating", self.rating), ("rating_scale_max", self.rating_scale_max)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
            ):
                raise TypeError(f"{name} must be a finite number or None")
        if self.rating_scale_max is not None and self.rating_scale_max <= 0:
            raise ValueError("rating_scale_max must be greater than zero")
        if self.rating is not None and self.rating < 0:
            raise ValueError("rating must not be negative")
        if self.rating is not None and self.rating_scale_max is not None and self.rating > self.rating_scale_max:
            raise ValueError("rating must not exceed rating_scale_max")
        if self.image_count is not None and (
            isinstance(self.image_count, bool) or not isinstance(self.image_count, int)
            or self.image_count < 0
        ):
            raise ValueError("image_count must be a non-negative integer or None")
        if self.text_predicted_label not in {None, "NORMAL", "SUSPICIOUS"}:
            raise ValueError("text_predicted_label must be NORMAL, SUSPICIOUS, or None")


@dataclass(frozen=True, slots=True)
class ReviewEvidenceTierResult:
    policy_name: str
    existing_rti_score: float
    existing_rti_level: str
    candidate_level: str
    candidate_reasons: tuple[str, ...]
    meaningful_character_count: int
    experience_evidence: str
    experience_context_hits: tuple[str, ...]
    experience_detail_hits: tuple[str, ...]
    promotional_evidence: str
    promotional_marketing_hits: tuple[str, ...]
    promotional_hype_hits: tuple[str, ...]
    strong_risk_reasons: tuple[str, ...]
    rti_level_conflict: bool


def meaningful_character_count(content: str) -> int:
    """Count Unicode letters and numbers, excluding spacing, emoji, and punctuation."""
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    return sum(unicodedata.category(character)[0] in {"L", "N"} for character in content)


def _compiled(patterns: tuple[str, ...]) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


def _hits(content: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(pattern.pattern for pattern in _compiled(patterns) if pattern.search(content))


def detect_experience_evidence(
    content: str, *, config: EvidenceRuleConfig = DEFAULT_RULE_CONFIG
) -> EvidenceMatch:
    context = _hits(content, config.experience_context_patterns)
    details = _hits(content, config.experience_detail_patterns)
    if context and details:
        value = EvidenceValue.YES
    elif context or details:
        value = EvidenceValue.UNCERTAIN
    else:
        value = EvidenceValue.NO
    return EvidenceMatch(value, context, details)


def detect_promotional_evidence(
    content: str, *, config: EvidenceRuleConfig = DEFAULT_RULE_CONFIG
) -> EvidenceMatch:
    marketing = _hits(content, config.marketing_patterns)
    hype = _hits(content, config.hype_patterns)
    if marketing and hype:
        value = EvidenceValue.YES
    elif marketing or hype:
        value = EvidenceValue.UNCERTAIN
    else:
        value = EvidenceValue.NO
    return EvidenceMatch(value, marketing, hype)


def _network_state(value: str | NetworkEvidenceState | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, NetworkEvidenceState):
        return value.value
    if not isinstance(value, str):
        raise TypeError("network_evidence_state must be a string, enum, or None")
    allowed = {state.value for state in NetworkEvidenceState}
    if value not in allowed:
        raise ValueError("unknown network_evidence_state")
    return value


def evaluate_review_evidence_tier(
    data: ReviewEvidenceInput, *, config: EvidenceRuleConfig = DEFAULT_RULE_CONFIG
) -> ReviewEvidenceTierResult:
    count = meaningful_character_count(data.content)
    experience = detect_experience_evidence(data.content, config=config)
    promotional = detect_promotional_evidence(data.content, config=config)
    existing_level = classify_rti(float(data.existing_rti_score)).value.upper()
    network_state = _network_state(data.network_evidence_state)

    reasons: list[str] = []
    strong_risks: list[str] = []
    if data.text_predicted_label == "SUSPICIOUS":
        strong_risks.append("TEXT_PREDICTION_SUSPICIOUS")
    if network_state in {
        NetworkEvidenceState.STRONG_PAIR.value,
        NetworkEvidenceState.STRONG_CLUSTER.value,
    }:
        strong_risks.append("STRONG_NETWORK_SIMILARITY")

    if count == 0:
        reasons.extend(("EMPTY_CONTENT", "MEANINGFUL_CONTENT_0_20"))
    elif count <= 20:
        reasons.append("MEANINGFUL_CONTENT_0_20")
    elif count <= 29:
        reasons.append("MEANINGFUL_CONTENT_21_29")

    verified_five_point_scale = data.rating_scale_max == 5
    rating_is_five = verified_five_point_scale and data.rating == 5
    if data.rating is None:
        reasons.append("RATING_MISSING")
    elif not verified_five_point_scale:
        reasons.append("RATING_SCALE_UNVERIFIED")
    elif rating_is_five:
        reasons.append("RATING_FIVE_FORCES_WARN")
    else:
        reasons.append("RATING_BELOW_FIVE")

    if rating_is_five and count < 30:
        if count == 0:
            reasons.append("EMPTY_CONTENT_RATING_FIVE")
        elif data.image_count is None:
            reasons.append("SHORT_CONTENT_IMAGE_UNKNOWN_RATING_FIVE")
        elif data.image_count > 0:
            reasons.append("SHORT_CONTENT_WITH_PHOTO_RATING_FIVE")
        else:
            reasons.append("SHORT_CONTENT_WITHOUT_PHOTO_RATING_FIVE")

    reasons.append(f"EXPERIENCE_EVIDENCE_{experience.value.value}")
    if promotional.value is EvidenceValue.YES:
        reasons.append("PROMOTIONAL_STYLE_DETECTED")
    elif promotional.value is EvidenceValue.UNCERTAIN:
        reasons.append("PROMOTIONAL_EVIDENCE_UNCERTAIN")
    else:
        reasons.append("PROMOTIONAL_EVIDENCE_NONE")
    reasons.extend(strong_risks)

    if existing_level == CandidateLevel.DANGER.value:
        candidate = CandidateLevel.DANGER
        reasons.insert(0, "EXISTING_RTI_DANGER_PRESERVED")
    else:
        safe = (
            count >= 30
            and verified_five_point_scale
            and data.rating is not None
            and data.rating < 5
            and experience.value is EvidenceValue.YES
            and promotional.value is not EvidenceValue.YES
            and not strong_risks
        )
        if safe:
            candidate = CandidateLevel.SAFE
            reasons.append("SAFE_REQUIRED_EVIDENCE_CONFIRMED")
        else:
            candidate = CandidateLevel.WARN

    return ReviewEvidenceTierResult(
        policy_name=POLICY_NAME,
        existing_rti_score=float(data.existing_rti_score),
        existing_rti_level=existing_level,
        candidate_level=candidate.value,
        candidate_reasons=tuple(dict.fromkeys(reasons)),
        meaningful_character_count=count,
        experience_evidence=experience.value.value,
        experience_context_hits=experience.context_hits,
        experience_detail_hits=experience.detail_hits,
        promotional_evidence=promotional.value.value,
        promotional_marketing_hits=promotional.context_hits,
        promotional_hype_hits=promotional.detail_hits,
        strong_risk_reasons=tuple(strong_risks),
        rti_level_conflict=candidate.value != existing_level,
    )
