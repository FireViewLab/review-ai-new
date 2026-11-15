from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from app.training.p_text_v3.row_review_cli import (
    BLIND_FIELDS,
    CONFIRMED_FIELDS,
    REVIEW_MODE,
    RowReviewError,
    load_sources,
    run_review,
)
from app.training.p_text_v3.validation_extension_confirmation import STAGING_FIELDS


SEOUL_TIME = datetime(2026, 11, 15, 14, 30, 5, tzinfo=timezone(timedelta(hours=9)))


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def staging_row(annotation_id: str, decision: str = "PRESENT") -> dict[str, str]:
    return {
        "annotation_id": annotation_id, "annotator_id": "", "text_status": "SUFFICIENT",
        "signal_decision": decision,
        "signal_types": "PRODUCT_DESCRIPTION_LIKE" if decision == "PRESENT" else "",
        "other_review_needed": "NO",
        "rationale_codes": "PROPOSED_BOUNDARY_PRODUCT_DESCRIPTION" if decision == "PRESENT" else "",
        "rationale_note": f"AI rationale for {annotation_id}",
        "guideline_version": "ptext-labeling-guide-v0.3", "annotation_round": "1",
        "annotated_at": "", "display_order": "1" if annotation_id == "a1" else "2",
        "proposal_origin": "AI_ASSISTED_DRAFT",
        "batch_confirmation_status": "USER_REPORTED_AGREEMENT_WITH_AI_DRAFT",
        "human_per_row_review_evidence": "NOT_RECORDED", "official_human_ground_truth": "NO",
        "eligible_for_training": "NO", "eligible_for_threshold_selection": "NO",
    }


def make_sources(tmp_path: Path):
    packet = tmp_path / "packet.csv"
    staging = tmp_path / "staging.csv"
    write_csv(packet, BLIND_FIELDS, [
        {"annotation_id": "a1", "display_order": "1", "content": "private review one"},
        {"annotation_id": "a2", "display_order": "2", "content": "private review two"},
    ])
    write_csv(staging, STAGING_FIELDS, [staging_row("a1"), staging_row("a2", "ABSENT")])
    return packet, staging


def inputs(*values):
    iterator = iter(values)
    return lambda prompt: next(iterator)


def test_agree_records_only_confirmed_row_with_real_seoul_timestamp(tmp_path):
    packet, staging = make_sources(tmp_path)
    logs = []
    result = run_review(
        packet_path=packet, staging_path=staging, output_dir=tmp_path / "private" / "session",
        reviewer_id="reviewer_local", expected_count=2, input_fn=inputs("a", "q"),
        output_fn=logs.append, now_fn=lambda: SEOUL_TIME, private_root=tmp_path / "private",
    )
    rows = read_csv(tmp_path / "private" / "session" / "confirmed_row_reviews.csv")
    assert result["status"] == "STOPPED_BY_USER"
    assert len(rows) == 1
    assert rows[0]["reviewer_id"] == "reviewer_local"
    assert rows[0]["confirmation_action"] == "AGREE"
    assert rows[0]["final_signal_decision"] == "PRESENT"
    assert rows[0]["reviewed_at"] == "2026-11-15T14:30:05+09:00"
    assert rows[0]["review_mode"] == REVIEW_MODE
    assert rows[0]["official_human_ground_truth"] == "NO"
    assert any("private review one" in line for line in logs)
    assert any("AI rationale for a1" in line for line in logs)


def test_modify_records_actual_user_label_and_rationale(tmp_path):
    packet, staging = make_sources(tmp_path)
    response = inputs(
        "m", "SUFFICIENT", "ABSENT", "", "NO", "", "직접 확인한 정상 표현", "q"
    )
    run_review(
        packet_path=packet, staging_path=staging, output_dir=tmp_path / "private" / "session",
        reviewer_id="human1", expected_count=2, input_fn=response, output_fn=lambda value: None,
        now_fn=lambda: SEOUL_TIME, private_root=tmp_path / "private",
    )
    row = read_csv(tmp_path / "private" / "session" / "confirmed_row_reviews.csv")[0]
    assert row["confirmation_action"] == "MODIFY"
    assert row["final_signal_decision"] == "ABSENT"
    assert row["final_signal_types"] == ""
    assert row["final_rationale_note"] == "직접 확인한 정상 표현"


def test_hold_creates_no_confirmation_and_can_be_resumed(tmp_path):
    packet, staging = make_sources(tmp_path)
    output = tmp_path / "private" / "session"
    first = run_review(
        packet_path=packet, staging_path=staging, output_dir=output,
        reviewer_id="human1", expected_count=2, input_fn=inputs("h", "q"),
        output_fn=lambda value: None, now_fn=lambda: SEOUL_TIME, private_root=tmp_path / "private",
    )
    assert first["confirmed_count"] == 0
    assert first["deferred_count"] == 1
    assert read_csv(output / "confirmed_row_reviews.csv") == []
    assert len(read_csv(output / "deferred_rows.csv")) == 1

    second = run_review(
        packet_path=packet, staging_path=staging, output_dir=output,
        reviewer_id="human1", expected_count=2, input_fn=inputs("a", "q"),
        output_fn=lambda value: None, now_fn=lambda: SEOUL_TIME, private_root=tmp_path / "private",
    )
    assert second["confirmed_count"] == 1
    assert second["deferred_count"] == 0


def test_immediate_quit_does_not_auto_approve_or_create_review_timestamp(tmp_path):
    packet, staging = make_sources(tmp_path)
    output = tmp_path / "private" / "session"
    result = run_review(
        packet_path=packet, staging_path=staging, output_dir=output,
        reviewer_id="human1", expected_count=2, input_fn=inputs("q"),
        output_fn=lambda value: None, now_fn=lambda: SEOUL_TIME, private_root=tmp_path / "private",
    )
    assert result["confirmed_count"] == 0
    assert read_csv(output / "confirmed_row_reviews.csv") == []
    manifest = json.loads((output / "review_session_manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_status"] == "IN_PROGRESS"
    assert manifest["official_human_ground_truth"] is False


def test_duplicate_or_mismatched_source_ids_are_rejected(tmp_path):
    packet, staging = make_sources(tmp_path)
    rows = [staging_row("a1"), staging_row("a1")]
    write_csv(staging, STAGING_FIELDS, rows)
    with pytest.raises(RowReviewError, match="STAGING_ANNOTATION_ID_DUPLICATE"):
        load_sources(packet, staging, expected_count=2)


def test_invalid_modified_combination_reprompts_until_valid(tmp_path):
    packet, staging = make_sources(tmp_path)
    logs = []
    response = inputs(
        "m",
        "INSUFFICIENT", "PRESENT", "PRODUCT_DESCRIPTION_LIKE", "NO", "", "bad",
        "SUFFICIENT", "ABSENT", "", "NO", "", "valid human reason",
        "q",
    )
    run_review(
        packet_path=packet, staging_path=staging, output_dir=tmp_path / "private" / "session",
        reviewer_id="human1", expected_count=2, input_fn=response, output_fn=logs.append,
        now_fn=lambda: SEOUL_TIME, private_root=tmp_path / "private",
    )
    assert any("LABEL_COMBINATION_INVALID" in line for line in logs)
    row = read_csv(tmp_path / "private" / "session" / "confirmed_row_reviews.csv")[0]
    assert row["final_signal_decision"] == "ABSENT"


def test_output_must_be_under_private_ignored_root(tmp_path):
    packet, staging = make_sources(tmp_path)
    with pytest.raises(RowReviewError, match="OUTPUT_NOT_IN_PRIVATE_EXTENSION_ROOT"):
        run_review(
            packet_path=packet, staging_path=staging, output_dir=tmp_path / "outside",
            reviewer_id="human1", expected_count=2, input_fn=inputs("q"),
            output_fn=lambda value: None, private_root=tmp_path / "private",
        )


def test_resume_rejects_source_mutation_and_naive_review_timestamp(tmp_path):
    packet, staging = make_sources(tmp_path)
    output = tmp_path / "private" / "session"
    run_review(
        packet_path=packet, staging_path=staging, output_dir=output,
        reviewer_id="human1", expected_count=2, input_fn=inputs("a", "q"),
        output_fn=lambda value: None, now_fn=lambda: SEOUL_TIME, private_root=tmp_path / "private",
    )
    rows = read_csv(output / "confirmed_row_reviews.csv")
    rows[0]["reviewed_at"] = "2026-11-15T14:30:05"
    write_csv(output / "confirmed_row_reviews.csv", CONFIRMED_FIELDS, rows)
    with pytest.raises(RowReviewError, match="CONFIRMED_TIMESTAMP_TIMEZONE_REQUIRED"):
        run_review(
            packet_path=packet, staging_path=staging, output_dir=output,
            reviewer_id="human1", expected_count=2, input_fn=inputs("q"),
            output_fn=lambda value: None, private_root=tmp_path / "private",
        )


def test_resume_rejects_aware_timestamp_outside_asia_seoul(tmp_path):
    packet, staging = make_sources(tmp_path)
    output = tmp_path / "private" / "session"
    run_review(
        packet_path=packet, staging_path=staging, output_dir=output,
        reviewer_id="human1", expected_count=2, input_fn=inputs("a", "q"),
        output_fn=lambda value: None, now_fn=lambda: SEOUL_TIME, private_root=tmp_path / "private",
    )
    rows = read_csv(output / "confirmed_row_reviews.csv")
    rows[0]["reviewed_at"] = "2026-11-15T05:30:05+00:00"
    write_csv(output / "confirmed_row_reviews.csv", CONFIRMED_FIELDS, rows)
    with pytest.raises(RowReviewError, match="CONFIRMED_TIMESTAMP_NOT_ASIA_SEOUL"):
        run_review(
            packet_path=packet, staging_path=staging, output_dir=output,
            reviewer_id="human1", expected_count=2, input_fn=inputs("q"),
            output_fn=lambda value: None, private_root=tmp_path / "private",
        )
