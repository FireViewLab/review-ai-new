"""동일 상품 리뷰 집합의 관계와 유사도를 분석해 P_network를 계산하는 모듈.

역할:
- target 리뷰를 같은 상품의 다른 리뷰와 비교한다.
- 중복·유사 리뷰 feature와 현재 v0 P_network를 계산한다.
- 유사도 계산은 adapter로 분리해 향후 한국어 embedding 모델로 교체할 수 있다.

수정 범위:
- [AI CORE]
- Spring/크롤링 연동을 위해 scoring 로직을 직접 수정하지 않는다.
- similarity 판단 또는 scoring 정책 변경은 AI 담당자와 협의한다.

주의:
- exact/normalized duplicate 감점은 Ground Truth로 검증되지 않은 v0 heuristic이다.
- similarity 결과가 없으면 임의 점수를 만들지 않으며 crawler를 직접 호출하지 않는다.
"""

from dataclasses import dataclass
from enum import Enum
from math import exp
from statistics import fmean
from typing import Protocol, Sequence

from app.integrations.similarity import (
    NormalizedTextSimilarityAdapter,
    hybrid_similarity_rows,
    is_informative_text,
)

SIMILARITY_THRESHOLD = 0.85
CLUSTER_SIZE = 5
SIMILARITY_SCORE_FLOOR = 0.50


class NetworkScoringPolicy(str, Enum):
    """Explicitly selected P_network scoring behavior.

    ``BASELINE`` remains the public default. ``STRONG_EVIDENCE_V1`` is an
    offline experiment that keeps weak evidence diagnostics but withholds the
    score unless at least one raw similarity reaches the existing 0.85 strong
    threshold.
    """

    BASELINE = "baseline"
    STRONG_EVIDENCE_V1 = "network_strong_evidence_v1"


class NetworkEvidenceState(str, Enum):
    """Non-scoring evidence status for one review inside its supplied batch.

    ``COMPARISON_SUFFICIENT`` means only that at least one informative pair was
    computed exhaustively inside the supplied batch.  It is not a statistical
    or operational claim that the product's review population is sufficiently
    covered; no unvalidated minimum-peer gate is applied to scoring.
    """

    NO_PEERS = "NO_PEERS"
    INSUFFICIENT_TEXT = "INSUFFICIENT_TEXT"
    COMPARISON_SUFFICIENT_NO_PATTERN = "COMPARISON_SUFFICIENT_NO_PATTERN"
    WEAK_ABOVE_FLOOR = "WEAK_ABOVE_FLOOR"
    STRONG_PAIR = "STRONG_PAIR"
    STRONG_CLUSTER = "STRONG_CLUSTER"


class SimilarityAdapter(Protocol):
    """두 리뷰 본문의 유사도를 계산하는 adapter 인터페이스."""

    def calculate(self, left: str, right: str) -> float | None:
        """0~1 유사도 또는 계산 불가 시 None을 반환한다."""


@dataclass(frozen=True, slots=True)
class NetworkReview:
    """Spring이 전달하는 network 분석용 최소 리뷰 정보."""

    review_id: str
    product_id: str
    content: str


@dataclass(frozen=True, slots=True)
class NetworkFeatures:
    """실제로 계산된 리뷰 관계 feature."""

    similar_review_count: int | None
    similarity_max: float | None
    similarity_mean: float | None
    top3_mean_similarity: float | None
    compared_review_count: int


@dataclass(frozen=True, slots=True)
class NetworkReason:
    """P_network 점수에 영향을 준 판단 근거."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class NetworkEvidenceDiagnostics:
    """Identifier-free observability for Network evidence sufficiency."""

    peer_count: int
    informative_peer_count: int
    valid_pair_count: int
    similarity_max: float | None
    similarity_mean: float | None
    top3_mean_similarity: float | None
    above_score_floor_count: int
    strong_similarity_count: int
    evidence_state: NetworkEvidenceState
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class NetworkAnalysisResult:
    """P_network 점수 또는 명시적인 unavailable 상태."""

    available: bool
    p_network: float | None
    features: NetworkFeatures
    reasons: tuple[NetworkReason, ...]
    unavailable_reason: str | None
    evidence: NetworkEvidenceDiagnostics

    @property
    def network_score(self) -> float:
        return self.p_network if self.p_network is not None else -1


def analyze_network(
    target: NetworkReview,
    reviews: Sequence[NetworkReview],
    *,
    similarity_adapter: SimilarityAdapter | None = None,
    policy: NetworkScoringPolicy = NetworkScoringPolicy.BASELINE,
) -> NetworkAnalysisResult:
    """target을 같은 상품의 다른 리뷰들과 비교해 P_network를 반환한다.

    product_id가 다른 리뷰는 Spring → AI 계약 오류이므로 ``ValueError``를
    발생시킨다. review_id가 target과 같으면 자기 자신으로 보고 비교에서 제외한다.
    """

    _validate_review(target, field_name="target")
    for index, review in enumerate(reviews):
        _validate_review(review, field_name=f"reviews[{index}]")
        if review.product_id != target.product_id:
            raise ValueError(
                "all comparison reviews must have the target product_id: "
                f"expected {target.product_id!r}, got {review.product_id!r}"
            )

    candidates = [review for review in reviews if review.review_id != target.review_id]
    if not candidates:
        return _unavailable_result(
            "no_comparison_reviews", peer_count=0, informative_peer_count=0,
            target_informative=is_informative_text(target.content),
        )

    if similarity_adapter is None:
        return analyze_network_batch((target, *candidates), policy=policy)[0]

    adapter = similarity_adapter or NormalizedTextSimilarityAdapter()
    if not is_informative_text(target.content):
        return _unavailable_result(
            "insufficient_text_information", peer_count=len(candidates),
            informative_peer_count=sum(is_informative_text(review.content) for review in candidates),
            target_informative=False,
        )
    similarities: list[float] = []
    for review in candidates:
        if not is_informative_text(review.content):
            continue
        similarity = adapter.calculate(target.content, review.content)
        if similarity is None:
            continue
        if not 0.0 <= similarity <= 1.0:
            raise ValueError("similarity adapter must return a value between 0 and 1")
        similarities.append(similarity)

    if not similarities:
        return _unavailable_result(
            "similarity_unavailable", peer_count=len(candidates),
            informative_peer_count=sum(is_informative_text(review.content) for review in candidates),
            target_informative=True,
        )

    return _from_similarities(
        similarities, peer_count=len(candidates),
        informative_peer_count=sum(is_informative_text(review.content) for review in candidates),
        target_informative=True,
        policy=policy,
    )


def _from_similarities(
    similarities, *, peer_count: int | None = None,
    informative_peer_count: int | None = None, target_informative: bool = True,
    policy: NetworkScoringPolicy = NetworkScoringPolicy.BASELINE,
):
    if not isinstance(policy, NetworkScoringPolicy):
        raise TypeError("policy must be a NetworkScoringPolicy")
    peer_count = len(similarities) if peer_count is None else peer_count
    informative_peer_count = len(similarities) if informative_peer_count is None else informative_peer_count
    if not similarities:
        return _unavailable_result(
            "similarity_unavailable", peer_count=peer_count,
            informative_peer_count=informative_peer_count,
            target_informative=target_informative,
        )
    similar_review_count = sum(similarity >= SIMILARITY_THRESHOLD for similarity in similarities)
    above_score_floor_count = sum(similarity > SIMILARITY_SCORE_FLOOR for similarity in similarities)
    top3_mean_similarity = fmean(sorted(similarities, reverse=True)[:3])
    features = NetworkFeatures(
        similar_review_count=similar_review_count,
        similarity_max=max(similarities),
        similarity_mean=fmean(similarities),
        top3_mean_similarity=top3_mean_similarity,
        compared_review_count=len(similarities),
    )

    if similar_review_count == 0 and features.similarity_max <= SIMILARITY_SCORE_FLOOR:
        unavailable_reason = "no_meaningful_similarity_evidence"
        return NetworkAnalysisResult(
            available=False,
            p_network=None,
            features=features,
            reasons=(),
            unavailable_reason=unavailable_reason,
            evidence=_evidence_diagnostics(
                features, peer_count=peer_count, informative_peer_count=informative_peer_count,
                above_score_floor_count=above_score_floor_count,
                evidence_state=NetworkEvidenceState.COMPARISON_SUFFICIENT_NO_PATTERN,
                unavailable_reason=unavailable_reason,
            ),
        )

    if (
        policy is NetworkScoringPolicy.STRONG_EVIDENCE_V1
        and similar_review_count == 0
    ):
        unavailable_reason = "strong_similarity_evidence_required"
        return NetworkAnalysisResult(
            available=False,
            p_network=None,
            features=features,
            reasons=(),
            unavailable_reason=unavailable_reason,
            evidence=_evidence_diagnostics(
                features,
                peer_count=peer_count,
                informative_peer_count=informative_peer_count,
                above_score_floor_count=above_score_floor_count,
                evidence_state=NetworkEvidenceState.WEAK_ABOVE_FLOOR,
                unavailable_reason=unavailable_reason,
            ),
        )

    scale = lambda value: min(1.0, max(0.0, (value - SIMILARITY_SCORE_FLOOR) / 0.50))
    cluster_scale = 1.0 - exp(-similar_review_count / 2.0)
    score = (
        100.0
        - 55.0 * scale(features.similarity_max) ** 2
        - 30.0 * scale(top3_mean_similarity) ** 2
        - 15.0 * cluster_scale
    )
    reasons: list[NetworkReason] = []
    if similar_review_count >= CLUSTER_SIZE:
        reasons.append(
            NetworkReason(
                code="SIMILAR_REVIEW_CLUSTER",
                message="높은 본문 유사도의 리뷰 군집 탐지",
            )
        )
    elif similar_review_count >= 1:
        reasons.append(
            NetworkReason(
                code="SIMILAR_REVIEW_PATTERN",
                message="높은 본문 유사도의 리뷰 탐지",
            )
        )

    evidence_state = (
        NetworkEvidenceState.STRONG_CLUSTER
        if similar_review_count >= CLUSTER_SIZE
        else NetworkEvidenceState.STRONG_PAIR
        if similar_review_count >= 1
        else NetworkEvidenceState.WEAK_ABOVE_FLOOR
    )
    return NetworkAnalysisResult(
        available=True,
        p_network=round(min(100.0, max(score, 0.0)), 1),
        features=features,
        reasons=tuple(reasons),
        unavailable_reason=None,
        evidence=_evidence_diagnostics(
            features, peer_count=peer_count, informative_peer_count=informative_peer_count,
            above_score_floor_count=above_score_floor_count,
            evidence_state=evidence_state, unavailable_reason=None,
        ),
    )


def analyze_network_batch(
    reviews: Sequence[NetworkReview],
    *,
    policy: NetworkScoringPolicy = NetworkScoringPolicy.BASELINE,
) -> tuple[NetworkAnalysisResult, ...]:
    """Fit once, reuse sparse vectors, and avoid retaining a dense pair matrix.

    Identity is review_id only; author is deliberately not part of this API.
    """
    ids = set()
    for review in reviews:
        _validate_review(review, field_name="review")
        if review.review_id in ids:
            raise ValueError("duplicate review_id")
        ids.add(review.review_id)
    if len({review.product_id for review in reviews}) > 1:
        raise ValueError("all reviews must have the same product_id")
    informative = [is_informative_text(review.content) for review in reviews]
    informative_total = sum(informative)
    rows = hybrid_similarity_rows([review.content for review in reviews])
    return tuple(
        _from_similarities(
            similarities,
            peer_count=max(0, len(reviews) - 1),
            informative_peer_count=informative_total - int(informative[index]),
            target_informative=informative[index],
            policy=policy,
        )
        for index, similarities in enumerate(rows)
    )


def _validate_review(review: NetworkReview, *, field_name: str) -> None:
    if not review.review_id.strip():
        raise ValueError(f"{field_name}.review_id must not be blank")
    if not review.product_id.strip():
        raise ValueError(f"{field_name}.product_id must not be blank")
    if not isinstance(review.content, str):
        raise TypeError(f"{field_name}.content must be a string")


def _evidence_diagnostics(
    features: NetworkFeatures, *, peer_count: int, informative_peer_count: int,
    above_score_floor_count: int, evidence_state: NetworkEvidenceState,
    unavailable_reason: str | None,
) -> NetworkEvidenceDiagnostics:
    return NetworkEvidenceDiagnostics(
        peer_count=peer_count,
        informative_peer_count=informative_peer_count,
        valid_pair_count=features.compared_review_count,
        similarity_max=features.similarity_max,
        similarity_mean=features.similarity_mean,
        top3_mean_similarity=features.top3_mean_similarity,
        above_score_floor_count=above_score_floor_count,
        strong_similarity_count=features.similar_review_count or 0,
        evidence_state=evidence_state,
        unavailable_reason=unavailable_reason,
    )


def _unavailable_result(
    reason: str, *, peer_count: int = 0, informative_peer_count: int = 0,
    target_informative: bool = False,
) -> NetworkAnalysisResult:
    evidence_state = (
        NetworkEvidenceState.NO_PEERS
        if peer_count == 0
        else NetworkEvidenceState.INSUFFICIENT_TEXT
    )
    features = NetworkFeatures(
        similar_review_count=None,
        similarity_max=None,
        similarity_mean=None,
        top3_mean_similarity=None,
        compared_review_count=0,
    )
    return NetworkAnalysisResult(
        available=False,
        p_network=None,
        features=features,
        reasons=(),
        unavailable_reason=reason,
        evidence=_evidence_diagnostics(
            features, peer_count=peer_count, informative_peer_count=informative_peer_count,
            above_score_floor_count=0, evidence_state=evidence_state,
            unavailable_reason=reason,
        ),
    )
