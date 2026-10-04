from copy import deepcopy

from app.training.p_text.build_synthetic_expansion_v1 import (
    KEYWORD_BALANCE_REVISED_PAIRS,
    PAIRS,
    build_rows,
)
from app.training.p_text.contrastive_pilot import ForbiddenRecord
from app.training.p_text.synthetic_expansion import validate_synthetic_expansion


def test_expansion_has_30_pairs_and_category_quota() -> None:
    report = validate_synthetic_expansion(build_rows())
    assert report.passed, report.errors
    assert report.counts["pairs"] == 30
    assert report.counts["rows"] == 60
    assert set(report.counts["category_pairs"].values()) == {5}


def test_category_quota_error_fails() -> None:
    rows = build_rows()
    for row in rows[:2]:
        row["category"] = "UNUSED_RECOMMENDATION"
    assert any("EXPERIENCE_FREE_CLAIM" in error for error in validate_synthetic_expansion(rows).errors)


def test_health_domain_quota_error_fails() -> None:
    rows = build_rows()
    changed = set()
    for row in rows:
        if row["domain"] == "health_supplement" and len(changed) < 3:
            changed.add(row["parent_pair_id"])
        if row["parent_pair_id"] in changed:
            row["domain"] = "household"
    assert any("health_supplement requires" in error for error in validate_synthetic_expansion(rows).errors)


def test_provenance_error_fails() -> None:
    rows = build_rows()
    rows[0]["source_type"] = "human_written"
    assert any("source_type must be synthetic" in error for error in validate_synthetic_expansion(rows).errors)


def test_human_reviewed_true_is_rejected() -> None:
    rows = build_rows()
    rows[0]["human_reviewed"] = True
    assert any("human_reviewed must be false" in error for error in validate_synthetic_expansion(rows).errors)


def test_extreme_praise_with_sufficient_evidence_normal_is_allowed() -> None:
    rows = build_rows()
    normal = next(
        row for row in rows
        if row["category"] == "EXAGGERATED_PRAISE_WITHOUT_EVIDENCE" and row["label"] == "NORMAL"
    )
    assert normal["praise_intensity"] == "extreme"
    assert normal["evidence_level"] == "sufficient"
    assert validate_synthetic_expansion(rows).passed


def test_extreme_praise_with_insufficient_evidence_suspicious_is_allowed() -> None:
    rows = build_rows()
    suspicious = next(
        row for row in rows
        if row["category"] == "EXAGGERATED_PRAISE_WITHOUT_EVIDENCE" and row["label"] == "SUSPICIOUS"
    )
    assert suspicious["praise_intensity"] == "extreme"
    assert suspicious["evidence_level"] in {"none", "limited"}
    assert validate_synthetic_expansion(rows).passed


def test_duplicate_detection_fails() -> None:
    rows = build_rows()
    rows[2]["content"] = rows[0]["content"]
    assert any("exact duplicate groups" in error for error in validate_synthetic_expansion(rows).errors)


def test_cross_family_near_template_detection_fails() -> None:
    rows = build_rows()
    rows[2]["content"] = rows[0]["content"] + " 정말입니다."
    assert any("cross-family near-template" in error for error in validate_synthetic_expansion(rows).errors)


def test_forbidden_overlap_detection_fails() -> None:
    rows = build_rows()
    forbidden = [ForbiddenRecord("blind.xlsx", "blind_id:B1", rows[0]["content"], "blind")]
    report = validate_synthetic_expansion(rows, forbidden_records=forbidden)
    assert report.counts["forbidden_exact_overlap"] == 1
    assert not report.passed


def test_builder_does_not_depend_on_or_mutate_pilot_rows() -> None:
    before = deepcopy(PAIRS)
    first = build_rows()
    first[0]["content"] = "changed only in returned value"
    assert PAIRS == before
    assert build_rows()[0]["content"] != first[0]["content"]


def test_keyword_balance_terms_are_not_label_exclusive() -> None:
    rows = build_rows()
    for keyword in ("건강", "성분", "추천", "완벽", "만족"):
        labels = {row["label"] for row in rows if keyword in row["content"]}
        assert labels == {"NORMAL", "SUSPICIOUS"}, keyword


def test_keyword_balance_revision_marker_is_limited_to_approved_pairs() -> None:
    rows = build_rows()
    expected = {f"synthetic-expansion-v1-p{number:03d}" for number in KEYWORD_BALANCE_REVISED_PAIRS}
    marked = {
        row["parent_pair_id"]
        for row in rows
        if "revision_reason=keyword_balance" in row["notes"]
    }
    assert marked == expected
    assert all(
        "revision_reason=keyword_balance" not in row["notes"]
        for row in rows
        if row["parent_pair_id"] == "synthetic-expansion-v1-p018"
    )
