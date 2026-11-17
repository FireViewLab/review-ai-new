"""AI-only integration adapter for the offline Re:view v3 candidate.

This module does not alter API/runtime defaults.  It reuses the existing text,
behavior, network, and Meta-Scorer implementations, then adds the offline
``review_evidence_tier_v3`` and ``BALANCED_INTERIOR_V1`` policy index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.analyzers.behavior import BehaviorInput, analyze_behavior
from app.analyzers.network import (
    NetworkReview,
    NetworkScoringPolicy,
    analyze_network,
    analyze_network_batch,
)
from app.analyzers.p_text import TextScorePredictor
from app.scoring.meta_scorer import (
    MetaScoreInput,
    ScoreSignal,
    calculate_meta_score,
)
from app.training.p_text_v3.review_evidence_policy_index_v3 import (
    BALANCED_INTERIOR,
    INTERPRETATION,
    policy_index_v3,
)
from app.training.p_text_v3.review_evidence_tier_v1 import ReviewEvidenceInput
from app.training.p_text_v3.review_evidence_tier_v3 import (
    evaluate_review_evidence_tier_v3,
)


EXPECTED_MODEL_SHA256 = "012b65174cdbc2e1fb36d75519438043bebd4317a69ea9997325b66bb436474b"
POLICY_INDEX_TYPE = "ORDINAL_V3_POLICY_INDEX"
PACKAGE_STATUS = "OFFLINE_TEAM_REVIEW_CANDIDATE_NOT_RUNTIME_DEFAULT"


class Predictor(Protocol):
    def predict_text_score(self, content: str) -> dict[str, Any]: ...


class IntegratedCandidateError(ValueError):
    """Fail-closed adapter error without review content."""


@dataclass(frozen=True, slots=True)
class IntegratedReviewInput:
    platform: str
    product_id: str
    review_id: str
    content: str
    rating: float | None = None
    rating_scale_max: float | None = None
    rating_source: str | None = None
    rating_source_verified: bool = False
    image_count: int | None = None
    image_count_source: str | None = None
    image_count_source_verified: bool = False
    behavior: BehaviorInput | None = None

    def __post_init__(self) -> None:
        for name in ("platform", "product_id", "review_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise IntegratedCandidateError(f"{name.upper()}_REQUIRED")
        if not isinstance(self.content, str) or not self.content.strip():
            raise IntegratedCandidateError("CONTENT_REQUIRED")
        _validate_rating(self)
        _validate_image_count(self)


@dataclass(frozen=True, slots=True)
class IntegratedAIResult:
    package_status: str
    platform: str
    product_id: str
    review_id: str
    original_rti: float
    original_level: str
    text_score: float
    behavior_score: float | None
    network_score: float | None
    text_predicted_label: str
    text_suspicious_probability: float
    evidence_level_v3: str
    evidence_reasons_v3: tuple[str, ...]
    policy_index_v3: float
    policy_index_type: str
    policy_index_interpretation: str
    short_text_penalty: float
    short_text_penalty_band: str
    experience_evidence: str
    promotional_evidence: str
    available_signals: tuple[str, ...]
    unavailable_signals: tuple[str, ...]
    missing_information: tuple[str, ...]
    signal_reasons: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_rating(data: IntegratedReviewInput) -> None:
    if data.rating is None:
        if data.rating_source_verified:
            raise IntegratedCandidateError("RATING_VERIFIED_WITHOUT_VALUE")
        return
    if isinstance(data.rating, bool) or not isinstance(data.rating, (int, float)):
        raise IntegratedCandidateError("RATING_INVALID")
    if data.rating_scale_max != 5:
        raise IntegratedCandidateError("RATING_SCALE_MUST_BE_VERIFIED_FIVE_POINT")
    if not 0 <= float(data.rating) <= 5:
        raise IntegratedCandidateError("RATING_OUT_OF_RANGE")
    if not data.rating_source_verified or not (data.rating_source or "").strip():
        raise IntegratedCandidateError("RATING_PROVENANCE_NOT_VERIFIED")


def _validate_image_count(data: IntegratedReviewInput) -> None:
    if data.image_count is None:
        if data.image_count_source_verified:
            raise IntegratedCandidateError("IMAGE_COUNT_VERIFIED_WITHOUT_VALUE")
        return
    if isinstance(data.image_count, bool) or not isinstance(data.image_count, int):
        raise IntegratedCandidateError("IMAGE_COUNT_INVALID")
    if data.image_count < 0:
        raise IntegratedCandidateError("IMAGE_COUNT_NEGATIVE")
    if not data.image_count_source_verified or not (data.image_count_source or "").strip():
        raise IntegratedCandidateError("IMAGE_COUNT_PROVENANCE_NOT_VERIFIED")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IntegratedAICandidate:
    """Explicit, non-runtime v3.1 B integration candidate."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_model_sha256: str = EXPECTED_MODEL_SHA256,
        device: str = "cpu",
        predictor_factory=TextScorePredictor,
    ) -> None:
        model_path = Path(model_path).resolve()
        weights = model_path / "model.safetensors"
        if not weights.is_file():
            raise IntegratedCandidateError("MODEL_WEIGHTS_NOT_FOUND")
        actual_sha = _file_sha256(weights)
        if actual_sha != expected_model_sha256:
            raise IntegratedCandidateError("MODEL_SHA256_MISMATCH")
        self.model_path = model_path
        self.model_sha256 = actual_sha
        self.predictor: Predictor = predictor_factory(str(model_path), device=device)

    @classmethod
    def from_predictor(cls, predictor: Predictor) -> "IntegratedAICandidate":
        """Test/offline dependency injection without claiming model verification."""

        instance = object.__new__(cls)
        instance.model_path = None
        instance.model_sha256 = None
        instance.predictor = predictor
        return instance

    def analyze_one(
        self,
        review: IntegratedReviewInput,
        *,
        comparison_reviews: Sequence[NetworkReview] = (),
    ) -> IntegratedAIResult:
        target = NetworkReview(review.review_id, review.product_id, review.content)
        network = analyze_network(
            target,
            tuple(comparison_reviews),
            policy=NetworkScoringPolicy.BASELINE,
        )
        return self._combine(review, network)

    def analyze_product_batch(
        self, reviews: Sequence[IntegratedReviewInput],
    ) -> tuple[IntegratedAIResult, ...]:
        if not reviews:
            return ()
        if len({review.product_id for review in reviews}) != 1:
            raise IntegratedCandidateError("MIXED_PRODUCT_BATCH")
        if len({review.review_id for review in reviews}) != len(reviews):
            raise IntegratedCandidateError("DUPLICATE_REVIEW_ID")
        network_reviews = tuple(
            NetworkReview(review.review_id, review.product_id, review.content)
            for review in reviews
        )
        network_results = analyze_network_batch(
            network_reviews,
            policy=NetworkScoringPolicy.BASELINE,
        )
        return tuple(
            self._combine(review, network)
            for review, network in zip(reviews, network_results, strict=True)
        )

    def _combine(self, review: IntegratedReviewInput, network) -> IntegratedAIResult:
        text = self.predictor.predict_text_score(review.content)
        if (
            text.get("text_score") == -1
            or text.get("predicted_label") not in {"NORMAL", "SUSPICIOUS"}
            or text.get("suspicious_probability") is None
        ):
            raise IntegratedCandidateError("PTEXT_INFERENCE_UNAVAILABLE")
        behavior = analyze_behavior(review.behavior or BehaviorInput())
        text_signal = ScoreSignal(True, float(text["text_score"]))
        behavior_signal = ScoreSignal(behavior.available, behavior.p_behavior)
        network_signal = ScoreSignal(network.available, network.p_network)
        meta = calculate_meta_score(MetaScoreInput(
            text=text_signal,
            behavior=behavior_signal,
            network=network_signal,
        ))
        if not meta.available or meta.rti is None or meta.level is None:
            raise IntegratedCandidateError("META_SCORER_UNAVAILABLE")

        tier = evaluate_review_evidence_tier_v3(ReviewEvidenceInput(
            content=review.content,
            rating=review.rating,
            rating_scale_max=review.rating_scale_max,
            image_count=review.image_count,
            existing_rti_score=meta.rti,
            text_predicted_label=text["predicted_label"],
            network_evidence_state=network.evidence.evidence_state,
        ))
        index = policy_index_v3(
            tier.original_rti,
            tier.candidate_level_v3,
            tier.meaningful_character_count,
            config=BALANCED_INTERIOR,
        )
        if not index.level_consistent:
            raise IntegratedCandidateError("POLICY_INDEX_LEVEL_CONFLICT")

        missing: list[str] = []
        if review.rating is None:
            missing.append("rating")
        if review.image_count is None:
            missing.append("image_count")
        if not behavior.available:
            missing.append("behavior_evidence")
        if not network.available:
            missing.append("network_evidence")
        signal_reasons = {
            "behavior": [reason.code for reason in behavior.reasons],
            "behavior_unavailable": list(behavior.unavailable_reasons),
            "network": [reason.code for reason in network.reasons],
            "network_unavailable": network.unavailable_reason,
            "network_evidence_state": network.evidence.evidence_state.value,
        }
        return IntegratedAIResult(
            package_status=PACKAGE_STATUS,
            platform=review.platform,
            product_id=review.product_id,
            review_id=review.review_id,
            original_rti=float(meta.rti),
            original_level=meta.level.value.upper(),
            text_score=float(text["text_score"]),
            behavior_score=behavior.p_behavior,
            network_score=network.p_network,
            text_predicted_label=text["predicted_label"],
            text_suspicious_probability=float(text["suspicious_probability"]),
            evidence_level_v3=tier.candidate_level_v3,
            evidence_reasons_v3=tier.candidate_reasons_v3,
            policy_index_v3=index.policy_index,
            policy_index_type=POLICY_INDEX_TYPE,
            policy_index_interpretation=INTERPRETATION,
            short_text_penalty=index.length_penalty,
            short_text_penalty_band=index.short_text_penalty_band,
            experience_evidence=tier.experience_evidence,
            promotional_evidence=tier.promotional_evidence,
            available_signals=meta.used_signals,
            unavailable_signals=meta.unavailable_signals,
            missing_information=tuple(missing),
            signal_reasons=signal_reasons,
            provenance={
                "rating": {
                    "source": review.rating_source,
                    "verified": review.rating_source_verified,
                },
                "image_count": {
                    "source": review.image_count_source,
                    "verified": review.image_count_source_verified,
                },
                "model_sha256": self.model_sha256,
            },
        )
