"""Fail-closed validation for official P_text v3 human-review evidence.

This module reads review artifacts but never writes or edits them.  Stable
error codes intentionally omit annotation identifiers and review content.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from .data_contract import SPLITS, file_sha256


CONTRACT_VERSION = "ptext-human-review-contract-v0.3"
GUIDELINE_VERSION = "ptext-labeling-guide-v0.3"
TEXT_STATUSES = frozenset({"SUFFICIENT", "INSUFFICIENT", "INVALID"})
SIGNAL_DECISIONS = frozenset({"PRESENT", "ABSENT", "UNCERTAIN", "NOT_ASSESSED"})
BOOLEAN_VALUES = frozenset({"YES", "NO"})
AI_PATH_MARKERS = ("ai_assisted", "ai-assisted", "ai_staging", "ai-staging")


class HumanReviewValidationError(ValueError):
    """A content-free validation error safe to show in logs."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise HumanReviewValidationError(code)


def _required_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(code)
    return value


def _load_json(path: str | Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HumanReviewValidationError(code) from exc
    if not isinstance(value, dict):
        _fail(code)
    return value


def _read_csv(path: str | Path, expected_fields: Sequence[str], code: str) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_fields):
                _fail(code + "_SCHEMA")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise HumanReviewValidationError(code + "_READ") from exc


def _unique(rows: Iterable[Mapping[str, str]], code: str) -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        annotation_id = str(row.get("annotation_id", "")).strip()
        if not annotation_id:
            _fail(code + "_ID_MISSING")
        if annotation_id in result:
            _fail(code + "_ID_DUPLICATE")
        result[annotation_id] = row
    return result


def _parse_values(value: object, allowed: frozenset[str], code: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split("|") if item.strip())
    if len(items) != len(set(items)):
        _fail(code + "_DUPLICATE")
    if set(items).difference(allowed):
        _fail(code + "_UNKNOWN")
    return tuple(sorted(items))


def _aware_timestamp(value: object, code: str) -> str:
    text = _required_text(value, code)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HumanReviewValidationError(code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(code)
    return text


def _positive_round(value: object) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise HumanReviewValidationError("ANNOTATION_ROUND_INVALID") from exc
    if parsed < 1:
        _fail("ANNOTATION_ROUND_INVALID")
    return parsed


def _validate_label(
    text_status: str,
    signal_decision: str,
    signal_types: tuple[str, ...],
    *,
    prefix: str,
) -> None:
    if text_status not in TEXT_STATUSES:
        _fail(prefix + "_TEXT_STATUS_INVALID")
    if signal_decision not in SIGNAL_DECISIONS:
        _fail(prefix + "_SIGNAL_DECISION_INVALID")
    allowed = {
        ("SUFFICIENT", "PRESENT"),
        ("SUFFICIENT", "ABSENT"),
        ("SUFFICIENT", "UNCERTAIN"),
        ("INSUFFICIENT", "NOT_ASSESSED"),
        ("INVALID", "NOT_ASSESSED"),
    }
    if (text_status, signal_decision) not in allowed:
        _fail(prefix + "_LABEL_COMBINATION_INVALID")
    if signal_decision == "PRESENT" and not signal_types:
        _fail(prefix + "_PRESENT_SIGNAL_TYPES_REQUIRED")
    if signal_decision in {"ABSENT", "NOT_ASSESSED"} and signal_types:
        _fail(prefix + "_SIGNAL_TYPES_MUST_BE_EMPTY")


def _label_tuple(row: Mapping[str, str], signal_vocabulary: frozenset[str]) -> tuple[Any, ...]:
    signal_types = _parse_values(row.get("signal_types"), signal_vocabulary, "SIGNAL_TYPES")
    text_status = str(row.get("text_status", ""))
    signal_decision = str(row.get("signal_decision", ""))
    _validate_label(text_status, signal_decision, signal_types, prefix="REVIEWER")
    other = str(row.get("other_review_needed", ""))
    if other not in BOOLEAN_VALUES:
        _fail("OTHER_REVIEW_NEEDED_INVALID")
    return text_status, signal_decision, signal_types, other


def _final_tuple(row: Mapping[str, str], signal_vocabulary: frozenset[str]) -> tuple[Any, ...]:
    signal_types = _parse_values(row.get("final_signal_types"), signal_vocabulary, "FINAL_SIGNAL_TYPES")
    text_status = str(row.get("final_text_status", ""))
    signal_decision = str(row.get("final_signal_decision", ""))
    _validate_label(text_status, signal_decision, signal_types, prefix="FINAL")
    other = str(row.get("final_other_review_needed", ""))
    if other not in BOOLEAN_VALUES:
        _fail("FINAL_OTHER_REVIEW_NEEDED_INVALID")
    return text_status, signal_decision, signal_types, other


def _validate_contract(
    contract: Mapping[str, Any],
    guideline_path: str | Path,
    source_packet_contract_path: str | Path,
) -> tuple[tuple[str, str], tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    if contract.get("contract_version") != CONTRACT_VERSION:
        _fail("REVIEW_CONTRACT_VERSION_INVALID")
    if contract.get("guideline_version") != GUIDELINE_VERSION:
        _fail("GUIDELINE_VERSION_INVALID")
    if contract.get("status") != "OFFICIAL_FOR_HUMAN_REVIEW":
        _fail("REVIEW_CONTRACT_NOT_OFFICIAL")
    if contract.get("guideline_sha256") != file_sha256(guideline_path):
        _fail("GUIDELINE_SHA256_MISMATCH")
    source = contract.get("source_packet_contract")
    if not isinstance(source, dict) or source.get("sha256") != file_sha256(source_packet_contract_path):
        _fail("SOURCE_PACKET_CONTRACT_SHA256_MISMATCH")
    packet = _load_json(source_packet_contract_path, "SOURCE_PACKET_CONTRACT_INVALID")
    reviewer_ids = tuple(contract.get("reviewer_ids", ()))
    if len(reviewer_ids) != 2 or len(set(reviewer_ids)) != 2 or any(not isinstance(value, str) or not value for value in reviewer_ids):
        _fail("REVIEWER_IDS_INVALID")
    if packet.get("reviewer_ids") != list(reviewer_ids):
        _fail("SOURCE_PACKET_REVIEWER_IDS_MISMATCH")
    if packet.get("annotation_count") != source.get("annotation_count"):
        _fail("SOURCE_PACKET_ANNOTATION_COUNT_MISMATCH")
    if packet.get("packet_fields") != contract.get("packet_fields"):
        _fail("SOURCE_PACKET_FIELDS_MISMATCH")
    if contract.get("review_mode") != "TWO_INDEPENDENT_BLIND_REVIEWERS":
        _fail("REVIEW_MODE_INVALID")
    combinations = contract.get("allowed_combinations")
    if not isinstance(combinations, list):
        _fail("ALLOWED_COMBINATIONS_INVALID")
    actual_combinations = {
        (item.get("text_status"), item.get("signal_decision"), item.get("signal_types"))
        for item in combinations if isinstance(item, dict)
    }
    expected_combinations = {
        ("SUFFICIENT", "PRESENT", "ONE_OR_MORE_REQUIRED"),
        ("SUFFICIENT", "ABSENT", "MUST_BE_EMPTY"),
        ("SUFFICIENT", "UNCERTAIN", "ZERO_OR_MORE_CANDIDATES"),
        ("INSUFFICIENT", "NOT_ASSESSED", "MUST_BE_EMPTY"),
        ("INVALID", "NOT_ASSESSED", "MUST_BE_EMPTY"),
    }
    if actual_combinations != expected_combinations or len(combinations) != len(expected_combinations):
        _fail("ALLOWED_COMBINATIONS_INVALID")
    independence = contract.get("independence_policy", {})
    if independence.get("ai_assisted_staging_is_independent_human_review") is not False:
        _fail("AI_INDEPENDENCE_POLICY_INVALID")
    rationale_policy = contract.get("rationale_policy", {})
    if rationale_policy.get("proposed_codes_are_ground_truth") is not False or rationale_policy.get("human_review_required") is not True:
        _fail("RATIONALE_POLICY_INVALID")
    vocabulary = contract.get("vocabulary")
    if not isinstance(vocabulary, dict):
        _fail("REVIEW_VOCABULARY_INVALID")
    if frozenset(vocabulary.get("text_status", ())) != TEXT_STATUSES:
        _fail("TEXT_STATUS_VOCABULARY_INVALID")
    if frozenset(vocabulary.get("signal_decision", ())) != SIGNAL_DECISIONS:
        _fail("SIGNAL_DECISION_VOCABULARY_INVALID")
    signals = tuple(vocabulary.get("signal_types", ()))
    rationales = tuple(vocabulary.get("rationale_codes", ()))
    if len(signals) != 5 or len(signals) != len(set(signals)) or any(not value for value in signals):
        _fail("SIGNAL_TYPES_VOCABULARY_INVALID")
    if len(rationales) != 8 or len(rationales) != len(set(rationales)) or any(not str(value).startswith("PROPOSED_") for value in rationales):
        _fail("RATIONALE_CODES_VOCABULARY_INVALID")
    return (str(reviewer_ids[0]), str(reviewer_ids[1])), signals, rationales, packet


def _reject_ai_source(path: str | Path) -> None:
    normalized = str(Path(path)).casefold().replace(" ", "_")
    if any(marker in normalized for marker in AI_PATH_MARKERS):
        _fail("AI_ASSISTED_SOURCE_FORBIDDEN")


def _validate_reviewer(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_reviewer: str,
    expected_ids: frozenset[str],
    signal_vocabulary: frozenset[str],
    rationale_vocabulary: frozenset[str],
) -> tuple[dict[str, tuple[Any, ...]], int]:
    by_id = _unique(rows, "REVIEWER")
    if frozenset(by_id) != expected_ids:
        _fail("REVIEWER_ANNOTATION_IDS_MISMATCH")
    labels: dict[str, tuple[Any, ...]] = {}
    for annotation_id, row in by_id.items():
        if row.get("annotator_id") != expected_reviewer:
            _fail("ANNOTATOR_ID_INVALID")
        if row.get("guideline_version") != GUIDELINE_VERSION:
            _fail("ANNOTATION_GUIDELINE_VERSION_INVALID")
        _positive_round(row.get("annotation_round"))
        _aware_timestamp(row.get("annotated_at"), "ANNOTATED_AT_INVALID")
        label = _label_tuple(row, signal_vocabulary)
        rationale_codes = _parse_values(row.get("rationale_codes"), rationale_vocabulary, "RATIONALE_CODES")
        rationale_note = str(row.get("rationale_note", "")).strip()
        if not rationale_codes and not rationale_note:
            _fail("REVIEWER_RATIONALE_REQUIRED")
        labels[annotation_id] = label
    return labels, len(by_id)


def validate_human_review(
    *,
    master_rows: Sequence[Mapping[str, str]],
    role_rows: Sequence[Mapping[str, str]],
    guideline_path: str | Path,
    review_contract_path: str | Path,
    source_packet_contract_path: str | Path,
    reviewer_a_path: str | Path,
    reviewer_b_path: str | Path,
    final_labels_path: str | Path,
) -> dict[str, Any]:
    """Validate official review evidence and return content-free counts/hashes."""
    _reject_ai_source(reviewer_a_path)
    _reject_ai_source(reviewer_b_path)
    if Path(reviewer_a_path).resolve() == Path(reviewer_b_path).resolve():
        _fail("REVIEWER_FILES_MUST_BE_DISTINCT")
    contract = _load_json(review_contract_path, "REVIEW_CONTRACT_INVALID")
    reviewer_ids, signal_values, rationale_values, packet_contract = _validate_contract(
        contract, guideline_path, source_packet_contract_path
    )
    annotation_fields = tuple(contract.get("annotation_fields", ()))
    final_fields = tuple(contract.get("adjudication_fields", ()))
    if not annotation_fields or not final_fields:
        _fail("REVIEW_CONTRACT_FIELDS_INVALID")
    masters = _unique(master_rows, "MASTER")
    roles = _unique(role_rows, "ROLE")
    expected_ids = frozenset(masters)
    expected_count = contract["source_packet_contract"].get("annotation_count")
    if not isinstance(expected_count, int) or expected_count < 1 or len(masters) != expected_count:
        _fail("MASTER_ANNOTATION_COUNT_MISMATCH")
    if frozenset(roles) != expected_ids:
        _fail("ROLE_ANNOTATION_IDS_MISMATCH")
    for row in roles.values():
        if row.get("split") not in SPLITS:
            _fail("ROLE_SPLIT_INVALID")
    reviewer_a_rows = _read_csv(reviewer_a_path, annotation_fields, "REVIEWER_A")
    reviewer_b_rows = _read_csv(reviewer_b_path, annotation_fields, "REVIEWER_B")
    signal_vocabulary = frozenset(signal_values)
    rationale_vocabulary = frozenset(rationale_values)
    labels_a, completed_a = _validate_reviewer(
        reviewer_a_rows,
        expected_reviewer=reviewer_ids[0],
        expected_ids=expected_ids,
        signal_vocabulary=signal_vocabulary,
        rationale_vocabulary=rationale_vocabulary,
    )
    labels_b, completed_b = _validate_reviewer(
        reviewer_b_rows,
        expected_reviewer=reviewer_ids[1],
        expected_ids=expected_ids,
        signal_vocabulary=signal_vocabulary,
        rationale_vocabulary=rationale_vocabulary,
    )
    final_rows = _read_csv(final_labels_path, final_fields, "FINAL_LABELS")
    finals = _unique(final_rows, "FINAL")
    if frozenset(finals) != expected_ids:
        _fail("FINAL_ANNOTATION_IDS_MISMATCH")

    agreements = 0
    disagreements = 0
    resolved = 0
    not_required = 0
    binary = Counter()
    final_distribution = Counter()
    split_binary = {split: Counter() for split in SPLITS}
    split_excluded = Counter()
    for annotation_id in sorted(expected_ids):
        left, right = labels_a[annotation_id], labels_b[annotation_id]
        agrees = left == right
        agreements += int(agrees)
        disagreements += int(not agrees)
        final_row = finals[annotation_id]
        status = str(final_row.get("adjudication_status", ""))
        final = _final_tuple(final_row, signal_vocabulary)
        if agrees and status == "not_required":
            not_required += 1
            if final != left:
                _fail("NOT_REQUIRED_FINAL_MISMATCH")
            metadata = (
                final_row.get("adjudicator_id"), final_row.get("adjudication_codes"),
                final_row.get("adjudication_note"), final_row.get("adjudicated_at"),
            )
            if any(str(value).strip() for value in metadata):
                _fail("NOT_REQUIRED_ADJUDICATION_METADATA_FORBIDDEN")
        elif status == "resolved":
            resolved += 1
            _required_text(final_row.get("adjudicator_id"), "ADJUDICATOR_ID_REQUIRED")
            _aware_timestamp(final_row.get("adjudicated_at"), "ADJUDICATED_AT_INVALID")
            codes = str(final_row.get("adjudication_codes", "")).strip()
            note = str(final_row.get("adjudication_note", "")).strip()
            if not codes and not note:
                _fail("ADJUDICATION_RATIONALE_REQUIRED")
        elif not agrees:
            _fail("DISAGREEMENT_NOT_ADJUDICATED")
        else:
            _fail("ADJUDICATION_STATUS_INVALID")
        status_name, decision, _, _ = final
        final_distribution[f"{status_name}/{decision}"] += 1
        split = str(roles[annotation_id]["split"])
        if status_name == "SUFFICIENT" and decision in {"PRESENT", "ABSENT"}:
            label = "SUSPICIOUS" if decision == "PRESENT" else "NORMAL"
            binary[label] += 1
            split_binary[split][label] += 1
        else:
            split_excluded[split] += 1

    return {
        "status": "PASSED",
        "contract_version": CONTRACT_VERSION,
        "guideline_version": GUIDELINE_VERSION,
        "total_review_count": len(expected_ids),
        "reviewer_a_completed": completed_a,
        "reviewer_b_completed": completed_b,
        "agreement_count": agreements,
        "disagreement_count": disagreements,
        "adjudication_resolved_count": resolved,
        "adjudication_not_required_count": not_required,
        "final_label_distribution": dict(sorted(final_distribution.items())),
        "binary_label_distribution": dict(sorted(binary.items())),
        "split_binary_labels": {
            split: dict(sorted(split_binary[split].items())) for split in SPLITS
        },
        "split_excluded_non_binary": dict(sorted(split_excluded.items())),
        "binary_eligible_count": sum(binary.values()),
        "binary_excluded_count": len(expected_ids) - sum(binary.values()),
        "input_sha256": {
            "guideline": file_sha256(guideline_path),
            "review_contract": file_sha256(review_contract_path),
            "source_packet_contract": file_sha256(source_packet_contract_path),
            "reviewer_a": file_sha256(reviewer_a_path),
            "reviewer_b": file_sha256(reviewer_b_path),
            "final_labels": file_sha256(final_labels_path),
        },
        "source_packet_sampling_seed": packet_contract.get("sampling_seed"),
        "operational_confirmation_required": [
            "confirm Reviewer A and B worked independently before result exchange",
            "confirm reviewer credentials and timestamps originate from the approved review system",
            "confirm AI-assisted staging was not copied into official human annotation files",
            "confirm raw Reviewer A/B files were frozen before adjudication and retained unchanged",
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--guideline", required=True)
    parser.add_argument("--review-contract", required=True)
    parser.add_argument("--source-packet-contract", required=True)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    parser.add_argument("--final-labels", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        master_fields = (
            "pool_id", "row_uid", "sampling_group", "sampling_reason", "label_status",
            "platform", "source_row", "product_id", "review_id", "product_category_raw",
            "normalized_category", "content", "normalized_content_hash", "content_length",
            "duplicate_cluster_id", "candidate_sources", "original_candidate_member",
            "original_control_eligible", "seed", "policy_version", "annotation_id",
            "split_role", "selection_category", "retrieval_evidence",
            "near_duplicate_review_required",
        )
        master_rows = _read_csv(args.master, master_fields, "MASTER")
        role_rows = _read_csv(args.roles, ("annotation_id", "split", "role_name"), "ROLES")
        report = validate_human_review(
            master_rows=master_rows,
            role_rows=role_rows,
            guideline_path=args.guideline,
            review_contract_path=args.review_contract,
            source_packet_contract_path=args.source_packet_contract,
            reviewer_a_path=args.reviewer_a,
            reviewer_b_path=args.reviewer_b,
            final_labels_path=args.final_labels,
        )
    except HumanReviewValidationError as exc:
        print(json.dumps({"status": "FAILED", "failures": [exc.code]}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
