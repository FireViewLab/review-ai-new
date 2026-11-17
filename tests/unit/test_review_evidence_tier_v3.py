from __future__ import annotations

import pytest

from app.training.p_text_v3.review_evidence_tier_v1 import ReviewEvidenceInput
from app.training.p_text_v3.review_evidence_tier_v3 import (
    apply_length_penalty_candidate,
    evaluate_review_evidence_tier_v3,
)


DETAILED = "2주 동안 직접 사용했고 오후에는 건조했지만 발림성과 지속력은 좋았습니다."
PROMOTIONAL = "혁신적인 기술력으로 완벽한 효과를 보장합니다. 누구에게나 반드시 추천합니다."


def evaluate(
    content=DETAILED, *, rating=4, scale=5, images=0, rti=55,
    text_label="NORMAL", network="COMPARISON_SUFFICIENT_NO_PATTERN",
):
    return evaluate_review_evidence_tier_v3(ReviewEvidenceInput(
        content=content, rating=rating, rating_scale_max=scale, image_count=images,
        existing_rti_score=rti, text_predicted_label=text_label,
        network_evidence_state=network,
    ))


@pytest.mark.parametrize("images", [0, 1])
def test_promotional_yes_is_danger_with_or_without_photo(images) -> None:
    result = evaluate(PROMOTIONAL, images=images)
    assert result.promotional_evidence == "YES"
    assert result.candidate_level_v3 == "DANGER"


def test_photo_promotional_uncertain_is_warn() -> None:
    result = evaluate("혁신적인 기술력", images=1)
    assert result.promotional_evidence == "UNCERTAIN"
    assert result.candidate_level_v3 == "WARN"


def test_photo_five_star_five_char_is_safe_with_strong_penalty_signal() -> None:
    result = evaluate("가" * 5, rating=5, images=1)
    assert result.candidate_level_v3 == "SAFE"
    assert result.short_text_penalty_band == "STRONG_0_9"
    assert result.short_text_reason == "MEANINGFUL_TEXT_0_9_STRONG_PENALTY"


def test_no_photo_five_star_five_char_is_danger() -> None:
    assert evaluate("가" * 5, rating=5, images=0).candidate_level_v3 == "DANGER"


def test_no_photo_three_star_five_char_is_warn_with_penalty() -> None:
    result = evaluate("가" * 5, rating=3, images=0)
    assert result.candidate_level_v3 == "WARN"
    assert result.short_text_penalty_band == "STRONG_0_9"


def test_photo_twenty_char_is_safe_with_general_penalty() -> None:
    result = evaluate("가" * 20, images=1)
    assert result.candidate_level_v3 == "SAFE"
    assert result.short_text_penalty_band == "GENERAL_10_29"


def test_photo_thirty_char_has_no_length_penalty() -> None:
    result = evaluate("가" * 30, images=1)
    assert result.candidate_level_v3 == "SAFE"
    assert result.short_text_penalty_band == "NONE_30_PLUS"


def test_existing_rti_danger_is_preserved_before_other_rules() -> None:
    result = evaluate(PROMOTIONAL, images=1, rti=39.5)
    assert result.candidate_level_v3 == "DANGER"
    assert "ORIGINAL_RTI_DANGER_PRESERVED" in result.candidate_reasons_v3


@pytest.mark.parametrize(
    ("count", "band"), [(9, "STRONG_0_9"), (10, "GENERAL_10_29"),
                         (29, "GENERAL_10_29"), (30, "NONE_30_PLUS")],
)
def test_length_penalty_boundaries(count, band) -> None:
    assert evaluate("가" * count, images=1).short_text_penalty_band == band


@pytest.mark.parametrize("maximum", [10.0, 20.0, 30.0])
def test_penalty_is_monotonic_and_original_rti_is_preserved(maximum) -> None:
    scores = [
        apply_length_penalty_candidate(55, count, maximum_penalty=maximum)
        for count in (0, 5, 9, 10, 20, 29, 30)
    ]
    assert all(result.original_rti == 55 for result in scores)
    assert all(
        left.applied_penalty >= right.applied_penalty
        for left, right in zip(scores, scores[1:])
    )
    assert scores[2].applied_penalty > scores[3].applied_penalty
    assert scores[-1].applied_penalty == 0


def test_strong_risk_blocks_photo_safe() -> None:
    assert evaluate(images=1, network="STRONG_PAIR").candidate_level_v3 == "WARN"


def test_candidate_danger_does_not_mutate_original_rti() -> None:
    result = evaluate(PROMOTIONAL, images=1, rti=55)
    assert result.candidate_level_v3 == "DANGER"
    assert result.original_rti == 55
    assert result.original_level == "WARN"
    assert result.grade_conflict
