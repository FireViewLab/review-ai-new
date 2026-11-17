from __future__ import annotations

from hashlib import sha256

import pytest

from app.training.p_text_v3.integrated_ai_candidate import (
    IntegratedAICandidate,
    IntegratedCandidateError,
    IntegratedReviewInput,
    POLICY_INDEX_TYPE,
)


class StubPredictor:
    def __init__(self, score=54, probability=0.42, label="NORMAL"):
        self.result = {
            "text_score": score,
            "suspicious_probability": probability,
            "predicted_label": label,
        }

    def predict_text_score(self, content):
        return dict(self.result)


def _review(**overrides):
    values = dict(
        platform="synthetic", product_id="p1", review_id="r1",
        content="가" * 30,
        rating=4, rating_scale_max=5, rating_source="fixture",
        rating_source_verified=True,
        image_count=1, image_count_source="fixture",
        image_count_source_verified=True,
    )
    values.update(overrides)
    return IntegratedReviewInput(**values)


def test_single_review_keeps_missing_network_behavior_and_adds_policy_index() -> None:
    analyzer = IntegratedAICandidate.from_predictor(StubPredictor())
    result = analyzer.analyze_one(_review(content="가" * 5, rating=5))
    assert result.original_rti == result.text_score == 54
    assert result.original_level == "WARN"
    assert result.behavior_score is None
    assert result.network_score is None
    assert result.available_signals == ("text",)
    assert result.unavailable_signals == ("behavior", "network")
    assert result.evidence_level_v3 == "SAFE"
    assert 70 <= result.policy_index_v3 <= 100
    assert result.short_text_penalty > 0
    assert result.policy_index_type == POLICY_INDEX_TYPE
    assert "behavior_evidence" in result.missing_information
    assert "network_evidence" in result.missing_information


def test_nullable_rating_and_image_are_not_defaulted() -> None:
    analyzer = IntegratedAICandidate.from_predictor(StubPredictor())
    result = analyzer.analyze_one(_review(
        rating=None, rating_scale_max=None, rating_source=None,
        rating_source_verified=False,
        image_count=None, image_count_source=None,
        image_count_source_verified=False,
    ))
    assert {"rating", "image_count"}.issubset(result.missing_information)
    assert result.provenance["rating"]["verified"] is False
    assert result.provenance["image_count"]["verified"] is False


def test_verified_no_photo_five_star_ultrashort_is_policy_danger() -> None:
    analyzer = IntegratedAICandidate.from_predictor(StubPredictor())
    result = analyzer.analyze_one(_review(content="가" * 5, rating=5, image_count=0))
    assert result.original_rti == 54
    assert result.evidence_level_v3 == "DANGER"
    assert 0 <= result.policy_index_v3 < 40


def test_batch_reuses_network_analyzer_and_detects_strong_similarity() -> None:
    analyzer = IntegratedAICandidate.from_predictor(StubPredictor())
    reviews = (
        _review(review_id="r1", content="동일한 사용 경험 문장이 충분히 길게 반복됩니다"),
        _review(review_id="r2", content="동일한 사용 경험 문장이 충분히 길게 반복됩니다"),
    )
    results = analyzer.analyze_product_batch(reviews)
    assert len(results) == 2
    assert all(result.network_score is not None for result in results)
    assert all(result.signal_reasons["network_evidence_state"] == "STRONG_PAIR" for result in results)
    assert all(result.evidence_level_v3 == "WARN" for result in results)


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"rating_source_verified": False}, "RATING_PROVENANCE_NOT_VERIFIED"),
        ({"rating_scale_max": 10}, "RATING_SCALE_MUST_BE_VERIFIED_FIVE_POINT"),
        ({"image_count_source_verified": False}, "IMAGE_COUNT_PROVENANCE_NOT_VERIFIED"),
        ({"image_count": -1}, "IMAGE_COUNT_NEGATIVE"),
    ],
)
def test_observed_fields_fail_closed_without_verified_contract(changes, code) -> None:
    with pytest.raises(IntegratedCandidateError, match=code):
        _review(**changes)


def test_batch_rejects_mixed_products_and_duplicate_ids() -> None:
    analyzer = IntegratedAICandidate.from_predictor(StubPredictor())
    with pytest.raises(IntegratedCandidateError, match="MIXED_PRODUCT_BATCH"):
        analyzer.analyze_product_batch((_review(), _review(product_id="p2", review_id="r2")))
    with pytest.raises(IntegratedCandidateError, match="DUPLICATE_REVIEW_ID"):
        analyzer.analyze_product_batch((_review(), _review()))


def test_model_sha_is_verified_once_before_predictor_creation(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    weights = b"synthetic weights"
    (model / "model.safetensors").write_bytes(weights)
    calls = []

    def factory(path, *, device):
        calls.append((path, device))
        return StubPredictor()

    analyzer = IntegratedAICandidate(
        model, expected_model_sha256=sha256(weights).hexdigest(),
        predictor_factory=factory,
    )
    assert analyzer.model_sha256 == sha256(weights).hexdigest()
    assert len(calls) == 1


def test_model_path_and_sha_fail_closed(tmp_path) -> None:
    with pytest.raises(IntegratedCandidateError, match="MODEL_WEIGHTS_NOT_FOUND"):
        IntegratedAICandidate(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"wrong")
    with pytest.raises(IntegratedCandidateError, match="MODEL_SHA256_MISMATCH"):
        IntegratedAICandidate(model)
