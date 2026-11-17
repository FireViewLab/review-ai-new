from __future__ import annotations

import pytest

from app.training.p_text_v3.review_evidence_policy_index_v3 import (
    BALANCED_INTERIOR,
    POLICY_INDEX_CANDIDATES,
    PolicyIndexV3Error,
    policy_index_v3,
)
from app.training.p_text_v3.review_evidence_tier_v1 import ReviewEvidenceInput
from app.training.p_text_v3.review_evidence_tier_v3 import (
    evaluate_review_evidence_tier_v3,
)


def _tier(*, content: str, images=1, rating=4, rti=55, text="NORMAL", network="COMPARISON_SUFFICIENT_NO_PATTERN"):
    return evaluate_review_evidence_tier_v3(ReviewEvidenceInput(
        content=content,
        rating=rating,
        rating_scale_max=5,
        image_count=images,
        existing_rti_score=rti,
        text_predicted_label=text,
        network_evidence_state=network,
    ))


def _index(tier, config=BALANCED_INTERIOR):
    return policy_index_v3(
        tier.original_rti,
        tier.candidate_level_v3,
        tier.meaningful_character_count,
        config=config,
    )


@pytest.mark.parametrize("config", POLICY_INDEX_CANDIDATES)
def test_same_safe_evidence_five_chars_scores_below_thirty(config) -> None:
    short = _index(_tier(content="가" * 5), config)
    long = _index(_tier(content="가" * 30), config)
    assert short.candidate_level_v3 == long.candidate_level_v3 == "SAFE"
    assert 70 <= short.policy_index < long.policy_index <= 100
    assert short.length_penalty > 0
    assert long.length_penalty == 0


def test_photo_five_star_five_chars_stays_safe_with_deduction() -> None:
    tier = _tier(content="가" * 5, images=1, rating=5)
    result = _index(tier)
    assert tier.candidate_level_v3 == "SAFE"
    assert result.score_level == "SAFE"
    assert result.policy_index < result.base_index_before_length_penalty
    assert result.short_text_penalty_band == "STRONG_0_9"


def test_no_photo_five_star_five_chars_is_danger_and_consistent() -> None:
    tier = _tier(content="가" * 5, images=0, rating=5)
    result = _index(tier)
    assert tier.candidate_level_v3 == result.score_level == "DANGER"
    assert 0 <= result.policy_index < 40


def test_promotional_candidate_danger_above_original_danger_boundary_is_supported() -> None:
    result = policy_index_v3(54, "DANGER", 45)
    assert result.original_rti == 54
    assert result.score_level == "DANGER"
    assert result.policy_index < 40


def test_original_rti_danger_cannot_be_relabelled() -> None:
    with pytest.raises(PolicyIndexV3Error, match="ORIGINAL_DANGER_NOT_PRESERVED"):
        policy_index_v3(39.9, "WARN", 30)
    result = policy_index_v3(39.9, "DANGER", 30)
    assert result.score_level == "DANGER"


@pytest.mark.parametrize("risk", [{"network": "STRONG_PAIR"}, {"text": "SUSPICIOUS"}])
def test_strong_risk_blocks_safe_and_index_stays_warn(risk) -> None:
    tier = _tier(content="가" * 40, **risk)
    result = _index(tier)
    assert tier.candidate_level_v3 == result.score_level == "WARN"
    assert 40 <= result.policy_index < 70


@pytest.mark.parametrize("config", POLICY_INDEX_CANDIDATES)
def test_same_length_preserves_original_rti_order_inside_tier(config) -> None:
    low = policy_index_v3(50, "SAFE", 20, config=config)
    high = policy_index_v3(60, "SAFE", 20, config=config)
    assert low.policy_index < high.policy_index


@pytest.mark.parametrize("config", POLICY_INDEX_CANDIDATES)
def test_all_tiers_are_score_consistent_and_never_reach_100(config) -> None:
    for level in ("DANGER", "WARN", "SAFE"):
        result = policy_index_v3(55, level, 30, config=config)
        assert result.score_level == level
        assert result.level_consistent
        assert result.policy_index < 100


def test_shorter_text_never_benefits_for_same_inputs() -> None:
    scores = [policy_index_v3(55, "SAFE", count).policy_index for count in range(31)]
    assert scores == sorted(scores)


def test_result_is_deterministic_and_preserves_original_rti() -> None:
    first = policy_index_v3(54, "DANGER", 5)
    second = policy_index_v3(54, "DANGER", 5)
    assert first == second
    assert first.original_rti == 54
