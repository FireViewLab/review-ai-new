from __future__ import annotations

import pytest

from app.training.p_text_v3.review_evidence_tier_v1 import ReviewEvidenceInput
from app.training.p_text_v3.review_evidence_tier_v2 import evaluate_review_evidence_tier_v2


DETAILED = "2주 동안 직접 사용했고 오후에는 건조했지만 발림성과 지속력은 좋았습니다."
PROMOTIONAL = "혁신적인 기술력으로 완벽한 효과를 보장합니다. 누구에게나 반드시 추천합니다."


def evaluate(
    content=DETAILED, *, rating=4, scale=5, images=0, rti=55,
    text_label="NORMAL", network="COMPARISON_SUFFICIENT_NO_PATTERN",
):
    return evaluate_review_evidence_tier_v2(ReviewEvidenceInput(
        content=content, rating=rating, rating_scale_max=scale, image_count=images,
        existing_rti_score=rti, text_predicted_label=text_label,
        network_evidence_state=network,
    ))


@pytest.mark.parametrize("content", ["", "가" * 9, "가" * 10, DETAILED])
@pytest.mark.parametrize("rating", [5, 4, None])
def test_verified_photo_is_safe_regardless_of_length_or_rating(content, rating) -> None:
    result = evaluate(content, rating=rating, scale=5 if rating is not None else None, images=1)
    assert result.candidate_level_v2 == "SAFE"
    assert "VERIFIED_PHOTO_AND_NO_PROMOTIONAL_STYLE" in result.candidate_reasons_v2


@pytest.mark.parametrize("images", [1, 2])
def test_photo_never_overrides_existing_danger(images) -> None:
    result = evaluate("", rating=None, scale=None, images=images, rti=39.9)
    assert result.candidate_level_v2 == "DANGER"
    assert result.original_rti == 39.9


@pytest.mark.parametrize(
    ("content", "expected"), [("가" * 9, "DANGER"), ("가" * 10, "WARN")],
)
def test_no_photo_five_star_ultrashort_boundary(content, expected) -> None:
    assert evaluate(content, rating=5, images=0).candidate_level_v2 == expected


@pytest.mark.parametrize("images", [None, 1])
def test_unknown_or_present_photo_is_not_no_photo_ultrashort_danger(images) -> None:
    result = evaluate("가" * 9, rating=5, images=images)
    assert result.candidate_level_v2 != "DANGER"
    assert result.candidate_level_v2 == ("SAFE" if images == 1 else "WARN")


def test_no_photo_detailed_experience_below_five_is_safe() -> None:
    assert evaluate(DETAILED, rating=4, images=0).candidate_level_v2 == "SAFE"


@pytest.mark.parametrize(
    "changes",
    [
        {"rating": 5}, {"rating": None, "scale": None},
        {"content": "좋아요 만족합니다"}, {"network": "STRONG_PAIR"},
        {"text_label": "SUSPICIOUS"},
    ],
)
def test_no_photo_detailed_safe_requires_every_gate(changes) -> None:
    values = {"content": DETAILED, "images": 0, **changes}
    assert evaluate(**values).candidate_level_v2 == "WARN"


@pytest.mark.parametrize("images", [0, 1])
def test_promotional_style_blocks_safe(images) -> None:
    result = evaluate(PROMOTIONAL, rating=4, images=images)
    assert result.candidate_level_v2 == "WARN"
    assert result.promotional_evidence == "YES"


def test_promotional_uncertainty_blocks_photo_safe() -> None:
    result = evaluate("혁신적인 기술력", rating=None, scale=None, images=1)
    assert result.promotional_evidence == "UNCERTAIN"
    assert result.candidate_level_v2 == "WARN"


@pytest.mark.parametrize("images", [0, 1])
@pytest.mark.parametrize("risk", [("SUSPICIOUS", "NO_PEERS"), ("NORMAL", "STRONG_CLUSTER")])
def test_strong_risk_blocks_photo_or_detailed_safe(images, risk) -> None:
    text_label, network = risk
    result = evaluate(images=images, text_label=text_label, network=network)
    assert result.candidate_level_v2 == "WARN"


def test_new_candidate_danger_does_not_mutate_original_rti() -> None:
    result = evaluate("가" * 9, rating=5, images=0, rti=55)
    assert result.candidate_level_v2 == "DANGER"
    assert result.original_rti == 55
    assert result.original_level == "WARN"
    assert result.grade_conflict
