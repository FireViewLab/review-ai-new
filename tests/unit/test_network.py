"""Final Hybrid A P_network regression tests."""

import pytest

from app.analyzers.network import (
    NetworkEvidenceState,
    NetworkReview,
    NetworkScoringPolicy,
    _from_similarities,
    analyze_network,
    analyze_network_batch,
)
from app.integrations.similarity import canonical_text, hybrid_similarity_rows


def review(review_id: str, content: str, *, product_id: str = "product-1") -> NetworkReview:
    return NetworkReview(review_id=review_id, product_id=product_id, content=content)


def batch(*texts: str):
    return analyze_network_batch(tuple(review(str(index), text) for index, text in enumerate(texts)))


class FixedSimilarityAdapter:
    def __init__(self, value: float | None):
        self.value = value

    def calculate(self, left: str, right: str) -> float | None:
        return self.value


def test_no_comparison_reviews_is_unavailable() -> None:
    target = review("target", "충분한 길이를 가진 실제 사용 후기 문장입니다.")
    result = analyze_network(target, ())
    assert result.network_score == -1
    assert result.unavailable_reason == "no_comparison_reviews"
    assert result.evidence.evidence_state is NetworkEvidenceState.NO_PEERS
    assert result.evidence.peer_count == 0
    assert result.evidence.informative_peer_count == 0
    assert result.evidence.valid_pair_count == 0


def test_low_similarity_with_valid_comparison_is_unavailable() -> None:
    result = _from_similarities([0.2, 0.5])
    assert result.network_score == -1
    assert result.available is False
    assert result.features.compared_review_count == 2
    assert result.features.similarity_max == 0.5
    assert result.features.similar_review_count == 0
    assert result.reasons == ()
    assert result.evidence.evidence_state is NetworkEvidenceState.COMPARISON_SUFFICIENT_NO_PATTERN
    assert result.evidence.peer_count == 2
    assert result.evidence.informative_peer_count == 2
    assert result.evidence.valid_pair_count == 2
    assert result.evidence.above_score_floor_count == 0
    assert result.evidence.strong_similarity_count == 0
    assert result.evidence.unavailable_reason == "no_meaningful_similarity_evidence"


def test_insufficient_target_text_has_peer_scope_without_valid_pair() -> None:
    results = batch("짧음", "충분한 비교 정보를 포함한 실제 사용 후기 문장입니다")
    short = results[0]
    assert short.available is False
    assert short.evidence.evidence_state is NetworkEvidenceState.INSUFFICIENT_TEXT
    assert short.evidence.peer_count == 1
    assert short.evidence.informative_peer_count == 1
    assert short.evidence.valid_pair_count == 0
    assert short.unavailable_reason == "similarity_unavailable"


def test_weak_above_floor_exposes_rounded_100_boundary_without_policy_change() -> None:
    result = _from_similarities([0.5103])
    assert result.available is True
    assert result.p_network == 100.0
    assert result.reasons == ()
    assert result.evidence.evidence_state is NetworkEvidenceState.WEAK_ABOVE_FLOOR
    assert result.evidence.above_score_floor_count == 1
    assert result.evidence.strong_similarity_count == 0
    assert result.evidence.similarity_max == pytest.approx(0.5103)


@pytest.mark.parametrize(
    ("similarity", "baseline_available", "strong_available"),
    [
        (0.49, False, False),
        (0.50, False, False),
        (0.5103, True, False),
        (0.8499, True, False),
        (0.85, True, True),
        (1.00, True, True),
    ],
)
def test_strong_evidence_policy_boundary(
    similarity: float,
    baseline_available: bool,
    strong_available: bool,
) -> None:
    baseline = _from_similarities([similarity])
    experimental = _from_similarities(
        [similarity], policy=NetworkScoringPolicy.STRONG_EVIDENCE_V1
    )

    assert baseline.available is baseline_available
    assert experimental.available is strong_available
    if 0.50 < similarity < 0.85:
        assert baseline.evidence.evidence_state is NetworkEvidenceState.WEAK_ABOVE_FLOOR
        assert experimental.evidence.evidence_state is NetworkEvidenceState.WEAK_ABOVE_FLOOR
        assert experimental.p_network is None
        assert experimental.unavailable_reason == "strong_similarity_evidence_required"
    if similarity >= 0.85:
        assert experimental.p_network == baseline.p_network
        assert experimental.reasons == baseline.reasons


def test_strong_evidence_policy_preserves_no_peer_and_insufficient_text() -> None:
    target = review("target", "충분한 길이의 실제 사용 후기 문장입니다.")
    no_peer = analyze_network(
        target, (), policy=NetworkScoringPolicy.STRONG_EVIDENCE_V1
    )
    insufficient = analyze_network_batch(
        (review("short", "짧음"), target),
        policy=NetworkScoringPolicy.STRONG_EVIDENCE_V1,
    )[0]

    assert no_peer.evidence.evidence_state is NetworkEvidenceState.NO_PEERS
    assert no_peer.unavailable_reason == "no_comparison_reviews"
    assert insufficient.evidence.evidence_state is NetworkEvidenceState.INSUFFICIENT_TEXT
    assert insufficient.available is False


def test_strong_evidence_policy_keeps_multiple_strong_peers_and_reason() -> None:
    similarities = [0.85, 0.91, 1.0]
    baseline = _from_similarities(similarities)
    experimental = _from_similarities(
        similarities, policy=NetworkScoringPolicy.STRONG_EVIDENCE_V1
    )

    assert experimental.available is True
    assert experimental.p_network == baseline.p_network
    assert experimental.evidence.strong_similarity_count == 3
    assert experimental.evidence.evidence_state is NetworkEvidenceState.STRONG_PAIR
    assert experimental.reasons == baseline.reasons


def test_unknown_network_policy_fails_closed() -> None:
    with pytest.raises(TypeError, match="NetworkScoringPolicy"):
        _from_similarities([0.9], policy="network_strong_evidence_v1")


def test_strong_pair_and_cluster_evidence_priority() -> None:
    pair = _from_similarities([0.51, 0.85, 0.91])
    cluster = _from_similarities([0.85] * 5 + [0.51])
    assert pair.evidence.evidence_state is NetworkEvidenceState.STRONG_PAIR
    assert pair.evidence.above_score_floor_count == 3
    assert pair.evidence.strong_similarity_count == 2
    assert cluster.evidence.evidence_state is NetworkEvidenceState.STRONG_CLUSTER
    assert cluster.evidence.above_score_floor_count == 6
    assert cluster.evidence.strong_similarity_count == 5


def test_similarity_above_score_floor_keeps_continuous_score() -> None:
    result = _from_similarities([0.5001])
    assert result.available is True
    assert 0 <= result.network_score <= 100
    assert result.reasons == ()


def test_canonical_normalization_does_not_mutate_source() -> None:
    source = " 배송  빠르고 제품도 좋아요!!! "
    assert canonical_text(source) == "배송빠르고제품도좋아요"
    assert source == " 배송  빠르고 제품도 좋아요!!! "


@pytest.mark.parametrize(("left", "right"), [
    ("2주 사용해보니 피부가 덜 건조해서 만족합니다.", "2주 사용 해보니 피부가 덜 건조해서 만족합니다!"),
    ("배송 빠르고 포장도 깔끔해요. 만족합니다.", "배송 빠르고 포장도 깔끔해요!!! 만족합니다"),
])
def test_spacing_and_punctuation_differences_are_strong(left: str, right: str) -> None:
    results = batch(left, right)
    assert results[0].features.similarity_max == 1.0
    assert results[0].features.similar_review_count == 1
    assert results[0].p_network == 9.1
    assert results[0].reasons[0].code == "SIMILAR_REVIEW_PATTERN"


def test_near_duplicate_just_below_raw_threshold_is_not_strong() -> None:
    results = batch("배송 빠르고 제품도 좋아요. 재구매할게요.", "배송도 빠르고 제품도 좋아요! 재구매 할게요.")
    assert 0.84 < results[0].features.similarity_max < 0.85
    assert results[0].p_network < 100
    assert results[0].reasons == ()


@pytest.mark.parametrize(("raw_similarity", "is_strong"), [
    (0.8499, False),
    (0.8500, True),
    (0.8501, True),
])
def test_strong_threshold_uses_unrounded_raw_similarity(
    raw_similarity: float,
    is_strong: bool,
) -> None:
    result = _from_similarities([raw_similarity])
    assert (result.features.similar_review_count == 1) is is_strong
    assert bool(result.reasons) is is_strong


@pytest.mark.parametrize(("left", "right"), [
    ("배송이 빨라서 좋았고 포장도 깔끔했습니다.", "한 달 사용해보니 피부가 조금 덜 건조해졌어요."),
    ("가격 대비 만족합니다. 배송도 빨라요.", "가격은 조금 비싸지만 사용감은 좋습니다."),
])
def test_different_reviews_do_not_create_false_positive(left: str, right: str) -> None:
    results = batch(left, right)
    assert results[0].features.similarity_max < 0.85
    assert results[0].p_network is None
    assert results[0].unavailable_reason == "no_meaningful_similarity_evidence"
    assert results[0].reasons == ()


def test_short_duplicates_are_not_cluster_evidence() -> None:
    results = batch(*(["좋아요"] * 5))
    assert all(result.network_score == -1 for result in results)
    assert all(result.reasons == () for result in results)


def test_short_review_does_not_make_long_reviews_unavailable() -> None:
    results = batch("좋아요", "배송 빠르고 제품도 좋아요. 재구매할게요.", "배송도 빠르고 제품도 좋아요! 재구매 할게요.")
    assert results[0].network_score == -1
    assert results[1].available is True
    assert results[1].features.compared_review_count == 1
    assert results[1].features.similar_review_count == 1


def test_six_long_duplicates_form_five_peer_cluster() -> None:
    text = "한 달 동안 사용했고 피부 당김이 줄어서 만족한 제품입니다."
    results = batch(*([text] * 6))
    assert all(result.features.similar_review_count == 5 for result in results)
    assert all(result.p_network == 1.2 for result in results)
    assert all(result.reasons[0].code == "SIMILAR_REVIEW_CLUSTER" for result in results)


def test_hybrid_similarity_changes_continuously() -> None:
    rows = hybrid_similarity_rows([
        "배송 빠르고 제품도 좋아요. 재구매할게요.",
        "배송도 빠르고 제품도 좋아요! 재구매 할게요.",
        "배송이 늦었지만 제품 색상은 화면과 비슷합니다.",
    ])
    assert 0 < rows[0][1] < rows[0][0] < 1


def test_custom_adapter_remains_supported_for_informative_text() -> None:
    target = review("target", "충분한 길이를 가진 실제 사용 후기 문장입니다.")
    peer = review("peer", "비교에 사용할 충분한 길이의 다른 후기 문장입니다.")
    result = analyze_network(target, (peer,), similarity_adapter=FixedSimilarityAdapter(0.9))
    assert result.available is True
    assert result.features.similar_review_count == 1


def test_mixed_product_ids_raise_contract_error() -> None:
    target = review("target", "충분한 길이를 가진 실제 사용 후기 문장입니다.")
    with pytest.raises(ValueError, match="target product_id"):
        analyze_network(target, (review("peer", "충분한 비교 후기입니다.", product_id="other"),))
