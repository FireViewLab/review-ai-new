from random import Random

import pytest

from scripts.analyze_ptext_low_information import (
    DeterministicSampler,
    calculate_statistics,
    length_bucket,
    normalize_content,
)


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (1, "1~5"),
        (5, "1~5"),
        (6, "6~10"),
        (10, "6~10"),
        (11, "11~20"),
        (20, "11~20"),
        (21, "21~40"),
        (40, "21~40"),
        (41, "41~80"),
        (80, "41~80"),
        (81, "81+"),
    ],
)
def test_length_bucket_boundaries(length, expected):
    assert length_bucket(length) == expected


def test_normalize_content_changes_whitespace_only():
    assert normalize_content("  좋아요\n\n잘 씀  ") == "좋아요 잘 씀"
    assert normalize_content(None) == ""
    assert normalize_content("AbC!?  123 🍃") == "AbC!? 123 🍃"


def _sample_rows(order):
    sampler = DeterministicSampler(per_bucket=3)
    for index in order:
        sampler.consider(
            platform="kurly",
            product_id=f"product-{index % 2}",
            review_id=f"review-{index}",
            normalized_content=f"내용{index:02d}",
        )
    return [sample.sample_hash for sample in sampler.samples()]


def test_deterministic_sampling_is_repeatable_and_order_invariant():
    ordered = list(range(20))
    shuffled = ordered[:]
    Random(20261018).shuffle(shuffled)
    assert _sample_rows(ordered) == _sample_rows(ordered)
    assert _sample_rows(ordered) == _sample_rows(shuffled)


def test_duplicate_normalized_content_is_sampled_once():
    sampler = DeterministicSampler(per_bucket=10)
    for review_id in ("z", "a", "m"):
        sampler.consider(
            platform="naver",
            product_id="p",
            review_id=review_id,
            normalized_content="같은 내용",
        )
    samples = sampler.samples()
    assert len(samples) == 1
    assert samples[0].normalized_content == "같은 내용"


def test_statistics_and_unavailable_are_aggregated_separately():
    records = [
        {"text_score": 80, "suspicious_probability": 0.2, "predicted_label": "NORMAL"},
        {"text_score": 90, "suspicious_probability": 0.1, "predicted_label": "NORMAL"},
        {"text_score": 100, "suspicious_probability": 0.0, "predicted_label": "NORMAL"},
        {"text_score": -1, "suspicious_probability": None, "predicted_label": None},
    ]
    result = calculate_statistics(records)
    assert result["inferred_sample_count"] == 4
    assert result["available_count"] == 3
    assert result["text_score"] == {"mean": 90.0, "median": 90.0, "min": 80.0, "max": 100.0}
    assert result["suspicious_probability"]["mean"] == pytest.approx(0.1)
    assert result["suspicious_probability"]["median"] == pytest.approx(0.1)
    assert result["text_score_gte_90"] == {"count": 2, "ratio": 0.5}
    assert result["text_score_gte_95"] == {"count": 1, "ratio": 0.25}
    assert result["predicted_label"]["NORMAL"] == {"count": 3, "ratio": 0.75}
    assert result["predicted_label"]["SUSPICIOUS"] == {"count": 0, "ratio": 0.0}
    assert result["unavailable"] == {"count": 1, "ratio": 0.25}
