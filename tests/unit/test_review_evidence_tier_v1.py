import pytest

from app.training.p_text_v3.review_evidence_tier_v1 import (
    ReviewEvidenceInput,
    detect_experience_evidence,
    detect_promotional_evidence,
    evaluate_review_evidence_tier,
    meaningful_character_count,
)


SAFE_TEXT = "2주 동안 직접 사용했는데 처음에는 발림성이 좋았지만 오후에는 조금 건조해졌어요. 개인적으로 장단점이 분명했습니다."
CRITICAL_TEXT = "한 달 동안 직접 착용해 보니 출근할 때 발이 편했지만 오래 걸으면 뒤꿈치가 조금 아쉬웠습니다. 사이즈는 기존 제품보다 넉넉합니다."
PROMOTIONAL_TEXT = "혁신적인 기술력과 독자적인 포뮬러로 무조건 구매해야 합니다. 단연 최고의 제품이며 절대 후회하지 않습니다."


def evaluate(
    content: str, *, rating=4, scale=5, images=None, rti=55,
    text_label="NORMAL", network_state=None,
):
    return evaluate_review_evidence_tier(ReviewEvidenceInput(
        content=content, rating=rating, rating_scale_max=scale,
        image_count=images, existing_rti_score=rti,
        text_predicted_label=text_label, network_evidence_state=network_state,
    ))


def test_meaningful_character_count_excludes_spacing_emoji_and_punctuation() -> None:
    assert meaningful_character_count("가 나A1 !! 😀") == 4


@pytest.mark.parametrize(
    ("content", "images", "reason"),
    [
        ("", 0, "EMPTY_CONTENT_RATING_FIVE"),
        ("좋아요최고", 0, "SHORT_CONTENT_WITHOUT_PHOTO_RATING_FIVE"),
        ("정말좋고추천합니다", 2, "SHORT_CONTENT_WITH_PHOTO_RATING_FIVE"),
    ],
)
def test_short_five_star_combinations_are_warn(content, images, reason) -> None:
    result = evaluate(content, rating=5, images=images)
    assert result.candidate_level == "WARN"
    assert "RATING_FIVE_FORCES_WARN" in result.candidate_reasons
    assert reason in result.candidate_reasons


def test_long_specific_five_star_review_is_still_warn() -> None:
    result = evaluate(SAFE_TEXT, rating=5)
    assert result.experience_evidence == "YES"
    assert result.candidate_level == "WARN"


@pytest.mark.parametrize("length", [10, 25])
def test_short_sub_five_reviews_are_warn(length) -> None:
    result = evaluate("가" * length, rating=4)
    assert result.candidate_level == "WARN"


def test_specific_experience_without_risk_is_safe_candidate() -> None:
    result = evaluate(SAFE_TEXT, rating=4)
    assert result.experience_evidence == "YES"
    assert result.promotional_evidence == "NO"
    assert result.candidate_level == "SAFE"


def test_detailed_critical_experience_is_safe_candidate() -> None:
    result = evaluate(CRITICAL_TEXT, rating=3)
    assert result.experience_evidence == "YES"
    assert result.candidate_level == "SAFE"


def test_combined_marketing_and_hype_is_warn() -> None:
    result = evaluate(PROMOTIONAL_TEXT, rating=4)
    assert result.promotional_evidence == "YES"
    assert result.candidate_level == "WARN"
    assert "PROMOTIONAL_STYLE_DETECTED" in result.candidate_reasons


def test_professional_term_alone_does_not_block_specific_experience() -> None:
    content = "독자적인 포뮬러라는 설명을 보고 2주 동안 직접 사용했는데 발림성은 부드럽고 오후에는 약간 건조했습니다."
    result = evaluate(content, rating=4)
    assert result.experience_evidence == "YES"
    assert result.promotional_evidence == "UNCERTAIN"
    assert result.candidate_level == "SAFE"


def test_missing_rating_is_warn() -> None:
    result = evaluate(SAFE_TEXT, rating=None, scale=None)
    assert result.candidate_level == "WARN"
    assert "RATING_MISSING" in result.candidate_reasons


def test_unverified_rating_scale_is_warn() -> None:
    result = evaluate(SAFE_TEXT, rating=4, scale=None)
    assert result.candidate_level == "WARN"
    assert "RATING_SCALE_UNVERIFIED" in result.candidate_reasons


def test_strong_similarity_is_warn() -> None:
    result = evaluate(SAFE_TEXT, network_state="STRONG_PAIR")
    assert result.candidate_level == "WARN"
    assert "STRONG_NETWORK_SIMILARITY" in result.strong_risk_reasons


def test_existing_danger_is_preserved_over_all_other_conditions() -> None:
    result = evaluate(SAFE_TEXT, rating=4, rti=39.99)
    assert result.candidate_level == "DANGER"
    assert result.existing_rti_score == 39.99
    assert "EXISTING_RTI_DANGER_PRESERVED" in result.candidate_reasons


def test_unavailable_network_does_not_block_safe_candidate() -> None:
    result = evaluate(SAFE_TEXT, network_state=None)
    assert result.candidate_level == "SAFE"


def test_uncertain_experience_is_warn() -> None:
    content = "제가 직접 사용해 봤는데 정말 좋아요. 전반적으로 매우 만족하고 추천합니다."
    result = evaluate(content)
    assert result.experience_evidence == "UNCERTAIN"
    assert result.candidate_level == "WARN"


def test_explicit_suspicious_text_prediction_is_warn() -> None:
    result = evaluate(SAFE_TEXT, text_label="SUSPICIOUS")
    assert result.candidate_level == "WARN"
    assert "TEXT_PREDICTION_SUSPICIOUS" in result.strong_risk_reasons


@pytest.mark.parametrize(
    ("length", "expected_reason"),
    [
        (20, "MEANINGFUL_CONTENT_0_20"),
        (21, "MEANINGFUL_CONTENT_21_29"),
        (29, "MEANINGFUL_CONTENT_21_29"),
        (30, None),
    ],
)
def test_meaningful_character_boundaries(length, expected_reason) -> None:
    result = evaluate("가" * length)
    assert result.meaningful_character_count == length
    if expected_reason is None:
        assert not any(reason.startswith("MEANINGFUL_CONTENT_") for reason in result.candidate_reasons)
    else:
        assert expected_reason in result.candidate_reasons


def test_detectors_require_both_evidence_families() -> None:
    assert detect_experience_evidence("직접 사용해 봤는데 정말 좋아요").value.value == "UNCERTAIN"
    assert detect_experience_evidence("2주 동안 배송을 기다렸습니다").value.value == "UNCERTAIN"
    assert detect_experience_evidence(SAFE_TEXT).value.value == "YES"
    assert detect_promotional_evidence("혁신적인 기술력의 제품입니다").value.value == "UNCERTAIN"
    assert detect_promotional_evidence(PROMOTIONAL_TEXT).value.value == "YES"


def test_candidate_level_is_separate_from_existing_rti() -> None:
    result = evaluate(SAFE_TEXT, rti=55)
    assert result.existing_rti_level == "WARN"
    assert result.existing_rti_score == 55
    assert result.candidate_level == "SAFE"
    assert result.rti_level_conflict is True
