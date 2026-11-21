from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from app.training.p_text_v3.data_contract import file_sha256
from app.training.p_text_v3.human_review_validator import (
    HumanReviewValidationError,
    validate_human_review,
)


ANNOTATION_FIELDS = (
    "annotation_id", "annotator_id", "text_status", "signal_decision", "signal_types",
    "other_review_needed", "rationale_codes", "rationale_note", "guideline_version",
    "annotation_round", "annotated_at",
)
FINAL_FIELDS = (
    "annotation_id", "adjudication_status", "adjudicator_id", "final_text_status",
    "final_signal_decision", "final_signal_types", "final_other_review_needed",
    "adjudication_codes", "adjudication_note", "adjudicated_at",
)
SIGNALS = (
    "PRODUCT_DESCRIPTION_LIKE", "NO_USE_ASSERTIVE_CLAIM",
    "PROMOTIONAL_OR_SALES_LANGUAGE", "HIGH_CERTAINTY_EFFICACY_CLAIM",
    "WITHIN_REVIEW_REPETITION",
)
RATIONALES = (
    "PROPOSED_BOUNDARY_EFFICACY_CLAIM", "PROPOSED_REPEAT_WITHIN_TEXT",
    "PROPOSED_BOUNDARY_PROMOTIONAL_LANGUAGE", "PROPOSED_LOW_INFO_NOT_ASSESSED",
    "PROPOSED_BOUNDARY_CONTEXT_UNCLEAR", "PROPOSED_PRODUCT_INFO_DOMINANT",
    "PROPOSED_BOUNDARY_PROMO_EFFICACY", "PROPOSED_BOUNDARY_PRODUCT_DESCRIPTION",
)


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def review_row(annotation_id: str, reviewer: str, *, present: bool) -> dict[str, str]:
    return {
        "annotation_id": annotation_id,
        "annotator_id": reviewer,
        "text_status": "SUFFICIENT",
        "signal_decision": "PRESENT" if present else "ABSENT",
        "signal_types": "HIGH_CERTAINTY_EFFICACY_CLAIM" if present else "",
        "other_review_needed": "NO",
        "rationale_codes": "PROPOSED_BOUNDARY_EFFICACY_CLAIM",
        "rationale_note": "",
        "guideline_version": "ptext-labeling-guide-v0.3",
        "annotation_round": "1",
        "annotated_at": "2026-11-13T10:00:00+09:00",
    }


def final_row(annotation_id: str, *, present: bool) -> dict[str, str]:
    return {
        "annotation_id": annotation_id,
        "adjudication_status": "not_required",
        "adjudicator_id": "",
        "final_text_status": "SUFFICIENT",
        "final_signal_decision": "PRESENT" if present else "ABSENT",
        "final_signal_types": "HIGH_CERTAINTY_EFFICACY_CLAIM" if present else "",
        "final_other_review_needed": "NO",
        "adjudication_codes": "",
        "adjudication_note": "",
        "adjudicated_at": "",
    }


def make_bundle(tmp_path: Path) -> dict:
    masters = [
        {"annotation_id": "a1", "row_uid": "r1", "platform": "p", "product_id": "p1", "review_id": "v1", "content": "fixture one"},
        {"annotation_id": "a2", "row_uid": "r2", "platform": "p", "product_id": "p2", "review_id": "v2", "content": "fixture two"},
    ]
    roles = [{"annotation_id": "a1", "split": "train"}, {"annotation_id": "a2", "split": "test"}]
    reviewer_a = [review_row("a1", "reviewer_a", present=False), review_row("a2", "reviewer_a", present=True)]
    reviewer_b = [review_row("a1", "reviewer_b", present=False), review_row("a2", "reviewer_b", present=True)]
    finals = [final_row("a1", present=False), final_row("a2", present=True)]
    guideline = tmp_path / "guide.md"
    guideline.write_text("official synthetic guideline", encoding="utf-8")
    source_packet = tmp_path / "source_contract.json"
    write_json(source_packet, {
        "reviewer_ids": ["reviewer_a", "reviewer_b"], "annotation_count": 2,
        "packet_fields": ["annotation_id", "display_order", "content"], "sampling_seed": 7,
    })
    contract = tmp_path / "contract.json"
    write_json(contract, {
        "contract_version": "ptext-human-review-contract-v0.3",
        "guideline_version": "ptext-labeling-guide-v0.3",
        "guideline_sha256": file_sha256(guideline),
        "status": "OFFICIAL_FOR_HUMAN_REVIEW",
        "source_packet_contract": {"sha256": file_sha256(source_packet), "annotation_count": 2},
        "review_mode": "TWO_INDEPENDENT_BLIND_REVIEWERS",
        "reviewer_ids": ["reviewer_a", "reviewer_b"],
        "packet_fields": ["annotation_id", "display_order", "content"],
        "annotation_fields": list(ANNOTATION_FIELDS), "adjudication_fields": list(FINAL_FIELDS),
        "vocabulary": {
            "text_status": ["SUFFICIENT", "INSUFFICIENT", "INVALID"],
            "signal_decision": ["PRESENT", "ABSENT", "UNCERTAIN", "NOT_ASSESSED"],
            "signal_types": list(SIGNALS), "rationale_codes": list(RATIONALES),
        },
        "allowed_combinations": [
            {"text_status": "SUFFICIENT", "signal_decision": "PRESENT", "signal_types": "ONE_OR_MORE_REQUIRED"},
            {"text_status": "SUFFICIENT", "signal_decision": "ABSENT", "signal_types": "MUST_BE_EMPTY"},
            {"text_status": "SUFFICIENT", "signal_decision": "UNCERTAIN", "signal_types": "ZERO_OR_MORE_CANDIDATES"},
            {"text_status": "INSUFFICIENT", "signal_decision": "NOT_ASSESSED", "signal_types": "MUST_BE_EMPTY"},
            {"text_status": "INVALID", "signal_decision": "NOT_ASSESSED", "signal_types": "MUST_BE_EMPTY"},
        ],
        "rationale_policy": {"proposed_codes_are_ground_truth": False, "human_review_required": True},
        "independence_policy": {"ai_assisted_staging_is_independent_human_review": False},
    })
    reviewer_a_path, reviewer_b_path, final_path = (
        tmp_path / "reviewer_a.csv", tmp_path / "reviewer_b.csv", tmp_path / "final.csv"
    )
    write_csv(reviewer_a_path, ANNOTATION_FIELDS, reviewer_a)
    write_csv(reviewer_b_path, ANNOTATION_FIELDS, reviewer_b)
    write_csv(final_path, FINAL_FIELDS, finals)
    return {
        "masters": masters, "roles": roles, "guideline": guideline, "source_packet": source_packet,
        "contract": contract, "reviewer_a_path": reviewer_a_path, "reviewer_b_path": reviewer_b_path,
        "final_path": final_path, "reviewer_a": reviewer_a, "reviewer_b": reviewer_b, "finals": finals,
    }


def validate(bundle: dict, **overrides):
    values = {
        "master_rows": bundle["masters"], "role_rows": bundle["roles"],
        "guideline_path": bundle["guideline"], "review_contract_path": bundle["contract"],
        "source_packet_contract_path": bundle["source_packet"],
        "reviewer_a_path": bundle["reviewer_a_path"], "reviewer_b_path": bundle["reviewer_b_path"],
        "final_labels_path": bundle["final_path"],
    }
    values.update(overrides)
    return validate_human_review(**values)


def test_complete_independent_review_passes_and_reports_content_free_counts(tmp_path):
    bundle = make_bundle(tmp_path)
    report = validate(bundle)
    assert report["status"] == "PASSED"
    assert (report["total_review_count"], report["reviewer_a_completed"], report["reviewer_b_completed"]) == (2, 2, 2)
    assert (report["agreement_count"], report["disagreement_count"]) == (2, 0)
    assert report["adjudication_not_required_count"] == 2
    assert report["binary_label_distribution"] == {"NORMAL": 1, "SUSPICIOUS": 1}
    assert report["split_binary_labels"] == {
        "train": {"NORMAL": 1}, "validation": {}, "test": {"SUSPICIOUS": 1}
    }
    assert "content" not in json.dumps(report)


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_reviewer_ids_must_be_complete_and_unique(tmp_path, mode):
    bundle = make_bundle(tmp_path)
    rows = deepcopy(bundle["reviewer_a"])
    rows = rows[:1] if mode == "missing" else rows + [deepcopy(rows[0])]
    write_csv(bundle["reviewer_a_path"], ANNOTATION_FIELDS, rows)
    with pytest.raises(HumanReviewValidationError, match="REVIEWER_"):
        validate(bundle)


def test_wrong_guideline_sha_is_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    contract = json.loads(bundle["contract"].read_text(encoding="utf-8"))
    contract["guideline_sha256"] = "0" * 64
    write_json(bundle["contract"], contract)
    with pytest.raises(HumanReviewValidationError, match="GUIDELINE_SHA256_MISMATCH"):
        validate(bundle)


def test_invalid_label_combination_is_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    rows = deepcopy(bundle["reviewer_a"])
    rows[0].update(text_status="INSUFFICIENT", signal_decision="PRESENT", signal_types="PRODUCT_DESCRIPTION_LIKE")
    write_csv(bundle["reviewer_a_path"], ANNOTATION_FIELDS, rows)
    with pytest.raises(HumanReviewValidationError, match="LABEL_COMBINATION_INVALID"):
        validate(bundle)


def test_disagreement_requires_completed_adjudication(tmp_path):
    bundle = make_bundle(tmp_path)
    rows = deepcopy(bundle["reviewer_b"])
    rows[0].update(signal_decision="PRESENT", signal_types="PRODUCT_DESCRIPTION_LIKE")
    write_csv(bundle["reviewer_b_path"], ANNOTATION_FIELDS, rows)
    with pytest.raises(HumanReviewValidationError, match="DISAGREEMENT_NOT_ADJUDICATED"):
        validate(bundle)


def test_resolved_disagreement_passes_with_adjudicator_evidence(tmp_path):
    bundle = make_bundle(tmp_path)
    rows = deepcopy(bundle["reviewer_b"])
    rows[0].update(signal_decision="PRESENT", signal_types="PRODUCT_DESCRIPTION_LIKE")
    write_csv(bundle["reviewer_b_path"], ANNOTATION_FIELDS, rows)
    finals = deepcopy(bundle["finals"])
    finals[0].update(
        adjudication_status="resolved", adjudicator_id="adjudicator_1",
        adjudication_codes="REVIEWER_DISAGREEMENT", adjudication_note="",
        adjudicated_at="2026-11-13T12:00:00+09:00",
    )
    write_csv(bundle["final_path"], FINAL_FIELDS, finals)
    report = validate(bundle)
    assert report["disagreement_count"] == 1
    assert report["adjudication_resolved_count"] == 1


def test_ai_assisted_staging_cannot_be_submitted_as_official_review(tmp_path):
    bundle = make_bundle(tmp_path)
    ai_path = tmp_path / "reviewer_a_ai_assisted_staging_400.csv"
    write_csv(ai_path, ANNOTATION_FIELDS, bundle["reviewer_a"])
    with pytest.raises(HumanReviewValidationError, match="AI_ASSISTED_SOURCE_FORBIDDEN"):
        validate(bundle, reviewer_a_path=ai_path)


def test_incorrect_not_required_final_label_is_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    finals = deepcopy(bundle["finals"])
    finals[0].update(final_signal_decision="PRESENT", final_signal_types="PRODUCT_DESCRIPTION_LIKE")
    write_csv(bundle["final_path"], FINAL_FIELDS, finals)
    with pytest.raises(HumanReviewValidationError, match="NOT_REQUIRED_FINAL_MISMATCH"):
        validate(bundle)


@pytest.mark.parametrize(
    "codes,note,error",
    [("UNKNOWN_CODE", "", "RATIONALE_CODES_UNKNOWN"), ("", "", "REVIEWER_RATIONALE_REQUIRED")],
)
def test_rationale_vocabulary_and_required_evidence(tmp_path, codes, note, error):
    bundle = make_bundle(tmp_path)
    rows = deepcopy(bundle["reviewer_a"])
    rows[0].update(rationale_codes=codes, rationale_note=note)
    write_csv(bundle["reviewer_a_path"], ANNOTATION_FIELDS, rows)
    with pytest.raises(HumanReviewValidationError, match=error):
        validate(bundle)
