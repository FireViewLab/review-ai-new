"""Build a clearly non-official P_text v3 dataset from AI-assisted draft labels.

This path is intentionally separate from ``build_dataset.py`` and its official
two-reviewer gate.  It accepts only explicitly confirmed draft provenance,
preserves the preassigned split, and never represents its output as Human
Ground Truth.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from .build_dataset import write_dataset_workbook
from .data_contract import (
    ContractError,
    SPLITS,
    VerifiedRelations,
    exclusion_matches,
    file_sha256,
    load_pilot_exclusions,
    load_verified_relations,
    normalized_content_hash,
    validate_rows,
)


EXPERIMENT_STATUS = "SINGLE_REVIEWER_AI_ASSISTED_EXPERIMENT"
PROVENANCE_STATUS = "AI_ASSISTED_STAGING_NOT_FORMAL_REVIEW"
CONFIRMATION_STATUS = "CONFIRMED_AS_DRAFT"
EXPECTED_SPLIT_COUNTS = {"train": 220, "validation": 80, "test": 100}
EXPECTED_TOTAL = sum(EXPECTED_SPLIT_COUNTS.values())
ROLE_NAMES = {"train": "TRAIN", "validation": "VALIDATION", "test": "FROZEN_TEST"}
TEXT_STATUSES = frozenset({"SUFFICIENT", "INSUFFICIENT", "INVALID"})
SIGNAL_DECISIONS = frozenset({"PRESENT", "ABSENT", "UNCERTAIN", "NOT_ASSESSED"})
SIGNAL_TYPES = frozenset(
    {
        "PRODUCT_DESCRIPTION_LIKE",
        "NO_USE_ASSERTIVE_CLAIM",
        "PROMOTIONAL_OR_SALES_LANGUAGE",
        "HIGH_CERTAINTY_EFFICACY_CLAIM",
        "WITHIN_REVIEW_REPETITION",
    }
)
RATIONALE_CODES = frozenset(
    {
        "PROPOSED_BOUNDARY_EFFICACY_CLAIM",
        "PROPOSED_REPEAT_WITHIN_TEXT",
        "PROPOSED_BOUNDARY_PROMOTIONAL_LANGUAGE",
        "PROPOSED_LOW_INFO_NOT_ASSESSED",
        "PROPOSED_BOUNDARY_CONTEXT_UNCLEAR",
        "PROPOSED_PRODUCT_INFO_DOMINANT",
        "PROPOSED_BOUNDARY_PROMO_EFFICACY",
        "PROPOSED_BOUNDARY_PRODUCT_DESCRIPTION",
    }
)
STAGING_FIELDS = (
    "annotation_id",
    "annotator_id",
    "text_status",
    "signal_decision",
    "signal_types",
    "other_review_needed",
    "rationale_codes",
    "rationale_note",
    "guideline_version",
    "annotation_round",
    "annotated_at",
)
MASTER_REQUIRED = frozenset(
    {"annotation_id", "row_uid", "platform", "product_id", "review_id", "content"}
)
ROLE_REQUIRED = frozenset({"annotation_id", "split"})


class ExperimentValidationError(ValueError):
    """Stable content-free error safe to include in experiment logs."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise ExperimentValidationError(code)


def _required_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(code)
    return value


def _read_csv(
    path: str | Path,
    *,
    exact_fields: Sequence[str] | None = None,
    required_fields: frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if exact_fields is not None and fields != tuple(exact_fields):
                _fail("CSV_SCHEMA_INVALID")
            if required_fields.difference(fields):
                _fail("CSV_COLUMNS_MISSING")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ExperimentValidationError("CSV_READ_FAILED") from exc


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentValidationError("PROVENANCE_READ_FAILED") from exc
    if not isinstance(value, dict):
        _fail("PROVENANCE_INVALID")
    return value


def _unique(
    rows: Iterable[Mapping[str, Any]], field: str, prefix: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get(field, "")).strip()
        if not key:
            _fail(prefix + "_ID_MISSING")
        if key in result:
            _fail(prefix + "_ID_DUPLICATE")
        result[key] = row
    return result


def _values(value: object, allowed: frozenset[str], code: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split("|") if item.strip())
    if len(items) != len(set(items)):
        _fail(code + "_DUPLICATE")
    if set(items).difference(allowed):
        _fail(code + "_UNKNOWN")
    return tuple(sorted(items))


def _validate_label(row: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    text_status = str(row.get("text_status", ""))
    decision = str(row.get("signal_decision", ""))
    signals = _values(row.get("signal_types"), SIGNAL_TYPES, "SIGNAL_TYPES")
    if text_status not in TEXT_STATUSES:
        _fail("TEXT_STATUS_INVALID")
    if decision not in SIGNAL_DECISIONS:
        _fail("SIGNAL_DECISION_INVALID")
    allowed = {
        ("SUFFICIENT", "PRESENT"),
        ("SUFFICIENT", "ABSENT"),
        ("SUFFICIENT", "UNCERTAIN"),
        ("INSUFFICIENT", "NOT_ASSESSED"),
        ("INVALID", "NOT_ASSESSED"),
    }
    if (text_status, decision) not in allowed:
        _fail("LABEL_COMBINATION_INVALID")
    if decision == "PRESENT" and not signals:
        _fail("PRESENT_SIGNAL_TYPES_REQUIRED")
    if decision in {"ABSENT", "NOT_ASSESSED"} and signals:
        _fail("SIGNAL_TYPES_MUST_BE_EMPTY")
    _values(row.get("rationale_codes"), RATIONALE_CODES, "RATIONALE_CODES")
    if str(row.get("other_review_needed", "")) not in {"YES", "NO"}:
        _fail("OTHER_REVIEW_NEEDED_INVALID")
    if str(row.get("guideline_version", "")) not in {
        "ptext-labeling-guide-v0.2",
        "ptext-labeling-guide-v0.3",
    }:
        _fail("GUIDELINE_VERSION_INVALID")
    return text_status, decision, signals


def _binary_label(row: Mapping[str, Any]) -> str | None:
    text_status, decision, _ = _validate_label(row)
    if text_status == "SUFFICIENT" and decision == "PRESENT":
        return "SUSPICIOUS"
    if text_status == "SUFFICIENT" and decision == "ABSENT":
        return "NORMAL"
    return None


def _validate_provenance(provenance: Mapping[str, Any], expected_total: int) -> str:
    if provenance.get("artifact_status") != PROVENANCE_STATUS:
        _fail("PROVENANCE_ARTIFACT_STATUS_INVALID")
    confirmation = provenance.get("user_confirmation_status")
    if confirmation not in {None, CONFIRMATION_STATUS}:
        _fail("USER_CONFIRMATION_STATUS_INVALID")
    if provenance.get("experiment_status") not in {None, EXPERIMENT_STATUS}:
        _fail("PROVENANCE_EXPERIMENT_STATUS_INVALID")
    if provenance.get("review_mode") not in {None, "ONE_PERSON_AI_ASSISTED_CONFIRMED_DRAFT"}:
        _fail("PROVENANCE_REVIEW_MODE_INVALID")
    if provenance.get("reviewer_count") not in {None, 1}:
        _fail("PROVENANCE_REVIEWER_COUNT_INVALID")
    if provenance.get("ai_assistance_used") not in {None, True}:
        _fail("PROVENANCE_AI_ASSISTANCE_INVALID")
    if provenance.get("confirmed_rows") != expected_total:
        _fail("PROVENANCE_CONFIRMED_ROWS_INVALID")
    _required_text(provenance.get("candidate_pack"), "PROVENANCE_CANDIDATE_PACK_MISSING")
    _required_text(provenance.get("confirmed_source"), "PROVENANCE_CONFIRMED_SOURCE_MISSING")
    if provenance.get("formal_reviewer_a_submission_completed") is not False:
        _fail("FORMAL_REVIEWER_A_STATUS_INVALID")
    if provenance.get("reviewer_b_independent_review_completed") is not False:
        _fail("REVIEWER_B_STATUS_INVALID")
    # The original 2026-11-13 provenance predates the explicit confirmation
    # field. Its exact non-formal status, confirmed source/count, and both
    # formal-review=false flags are accepted only for this non-official path.
    return "explicit_status" if confirmation == CONFIRMATION_STATUS else "legacy_confirmed_source_and_count"


def _validate_leakage(
    masters: Mapping[str, Mapping[str, Any]],
    roles: Mapping[str, Mapping[str, Any]],
    relations: VerifiedRelations,
) -> dict[str, int]:
    scopes: dict[str, dict[Any, set[str]]] = {
        name: defaultdict(set)
        for name in ("product", "normalized_content_hash", "duplicate_cluster", "verified_relation")
    }
    for annotation_id, master in masters.items():
        split = str(roles[annotation_id]["split"])
        scopes["product"][(str(master["platform"]), str(master["product_id"]))].add(split)
        scopes["normalized_content_hash"][normalized_content_hash(str(master["content"]))].add(split)
        cluster = str(master.get("duplicate_cluster_id", "")).strip()
        if cluster:
            scopes["duplicate_cluster"][cluster].add(split)
        row_uid = str(master.get("row_uid", "")).strip()
        for relation_type, group_id in relations.groups_by_row.get(row_uid, ()):
            scopes["verified_relation"][(relation_type, group_id)].add(split)
    counts = {
        scope: sum(len(split_values) > 1 for split_values in values.values())
        for scope, values in scopes.items()
    }
    leaking = sorted(scope for scope, count in counts.items() if count)
    if leaking:
        _fail("CROSS_SPLIT_LEAKAGE:" + ",".join(leaking))
    return counts


def validate_experiment(
    *,
    master_rows: Sequence[Mapping[str, Any]],
    role_rows: Sequence[Mapping[str, Any]],
    staging_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    exclusions=None,
    relations: VerifiedRelations | None = None,
    expected_split_counts: Mapping[str, int] = EXPECTED_SPLIT_COUNTS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate draft evidence and return a content-free report plus binary rows."""
    expected_counts = dict(expected_split_counts)
    if set(expected_counts) != set(SPLITS) or any(type(value) is not int or value < 1 for value in expected_counts.values()):
        _fail("EXPECTED_SPLIT_COUNTS_INVALID")
    expected_total = sum(expected_counts.values())
    confirmation_evidence = _validate_provenance(provenance, expected_total)
    masters = _unique(master_rows, "annotation_id", "MASTER")
    roles = _unique(role_rows, "annotation_id", "ROLE")
    staging = _unique(staging_rows, "annotation_id", "STAGING")
    expected_ids = frozenset(masters)
    if len(expected_ids) != expected_total:
        _fail("MASTER_ANNOTATION_COUNT_INVALID")
    if frozenset(roles) != expected_ids:
        _fail("ROLE_ANNOTATION_IDS_MISMATCH")
    if frozenset(staging) != expected_ids:
        _fail("STAGING_ANNOTATION_IDS_MISMATCH")

    role_counts = Counter()
    for annotation_id, role in roles.items():
        split = str(role.get("split", ""))
        if split not in SPLITS:
            _fail("ROLE_SPLIT_INVALID")
        role_counts[split] += 1
        master_role = str(masters[annotation_id].get("split_role", "")).strip()
        role_name = str(role.get("role_name", "")).strip()
        if master_role and master_role != ROLE_NAMES[split]:
            _fail("MASTER_ROLE_MISMATCH")
        if role_name and role_name != ROLE_NAMES[split]:
            _fail("ROLE_NAME_MISMATCH")
    if dict(role_counts) != expected_counts:
        _fail("FIXED_SPLIT_QUOTA_MISMATCH")

    exclusions = exclusions or load_pilot_exclusions(None)
    relations = relations or load_verified_relations(None)
    leakage = _validate_leakage(masters, roles, relations)
    binary_rows: list[dict[str, Any]] = []
    binary_distribution = Counter()
    split_binary = {split: Counter() for split in SPLITS}
    split_excluded = Counter()
    label_distribution = Counter()
    for annotation_id in sorted(expected_ids):
        staged = staging[annotation_id]
        # Blank annotator/timestamp are deliberate: this is a confirmed AI
        # draft, not evidence of an independent human review submission.
        if str(staged.get("annotator_id", "")).strip():
            _fail("OFFICIAL_REVIEW_REPRESENTATION_FORBIDDEN")
        if str(staged.get("annotated_at", "")).strip():
            _fail("OFFICIAL_REVIEW_TIMESTAMP_FORBIDDEN")
        label = _binary_label(staged)
        label_distribution[f"{staged.get('text_status')}/{staged.get('signal_decision')}"] += 1
        master = masters[annotation_id]
        base_row = {
            "platform": str(master["platform"]).strip(),
            "product_id": str(master["product_id"]).strip(),
            "review_id": str(master["review_id"]).strip(),
            "content": master["content"],
            "group_id": str(master.get("group_id") or f"review:{master['row_uid']}").strip(),
            "split": str(roles[annotation_id]["split"]),
            "row_uid": str(master["row_uid"]).strip(),
            "duplicate_cluster_id": str(master.get("duplicate_cluster_id", "")).strip(),
            "annotation_id": annotation_id,
            "source_set": str(master.get("sampling_group", "")).strip(),
        }
        if exclusion_matches(base_row, exclusions):
            _fail("PILOT_EXCLUSION_MATCH")
        if label is None:
            split_excluded[base_row["split"]] += 1
            continue
        row = base_row | {
            "label": label,
            "label_origin": "single_reviewer_ai_assisted_confirmed_draft",
        }
        binary_rows.append(row)
        binary_distribution[label] += 1
        split_binary[base_row["split"]][label] += 1

    # Reuse the canonical final-row contract in addition to the all-candidate
    # leakage audit above.  Frozen Test remains present only with split=test.
    canonical_validation = validate_rows(
        binary_rows,
        exclusions=exclusions,
        relations=relations,
        strict_product_isolation=True,
        require_both_labels_per_split=True,
    )
    report = {
        "status": "PASSED",
        "experiment_status": EXPERIMENT_STATUS,
        "official_human_ground_truth": False,
        "review_mode": "ONE_PERSON_AI_ASSISTED_CONFIRMED_DRAFT",
        "user_confirmation_status": CONFIRMATION_STATUS,
        "confirmation_evidence": confirmation_evidence,
        "total_candidate_count": expected_total,
        "fixed_split_counts": {split: role_counts[split] for split in SPLITS},
        "label_distribution": dict(sorted(label_distribution.items())),
        "binary_label_distribution": dict(sorted(binary_distribution.items())),
        "split_binary_labels": {
            split: dict(sorted(split_binary[split].items())) for split in SPLITS
        },
        "split_excluded_non_binary": {split: split_excluded[split] for split in SPLITS},
        "binary_eligible_count": len(binary_rows),
        "binary_excluded_count": expected_total - len(binary_rows),
        "all_candidate_leakage": leakage,
        "canonical_binary_validation": canonical_validation,
        "frozen_test_policy": {
            "preassigned_before_labels": True,
            "may_train": False,
            "may_select_model_or_threshold": False,
            "prior_ai_assisted_label_exposure": True,
            "independent_blind_evaluation_claim_allowed": False,
        },
    }
    return report, binary_rows


def run(
    *,
    master_path: str | Path,
    roles_path: str | Path,
    staging_path: str | Path,
    provenance_path: str | Path,
    output_path: str | Path,
    exclusion_manifest: str | Path | None = None,
    verified_relations: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")
    if "experiment" not in str(output).casefold():
        _fail("EXPERIMENT_OUTPUT_PATH_REQUIRED")
    masters = _read_csv(master_path, required_fields=MASTER_REQUIRED)
    roles = _read_csv(roles_path, required_fields=ROLE_REQUIRED)
    staging = _read_csv(staging_path, exact_fields=STAGING_FIELDS)
    provenance = _load_json(provenance_path)
    exclusions = load_pilot_exclusions(exclusion_manifest)
    relations = load_verified_relations(verified_relations)
    validation, rows = validate_experiment(
        master_rows=masters,
        role_rows=roles,
        staging_rows=staging,
        provenance=provenance,
        exclusions=exclusions,
        relations=relations,
    )
    summary = {
        "contract_version": "ptext-v3-presplit-v1",
        "experiment_status": EXPERIMENT_STATUS,
        "official_human_review_validated": False,
        "official_human_ground_truth": False,
        "automatic_labels_assigned": False,
        "split_generated": False,
        "frozen_test_evaluated": False,
        "validation": validation,
        "inputs": {
            "master_sha256": file_sha256(master_path),
            "roles_sha256": file_sha256(roles_path),
            "staging_sha256": file_sha256(staging_path),
            "provenance_sha256": file_sha256(provenance_path),
            "pilot_exclusions_sha256": file_sha256(exclusion_manifest) if exclusion_manifest else None,
            "verified_relations_sha256": file_sha256(verified_relations) if verified_relations else None,
        },
    }
    write_dataset_workbook(output, rows, summary)
    return summary | {"output_sha256": file_sha256(output)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-exclusions")
    parser.add_argument("--verified-relations")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        result = run(
            master_path=args.master,
            roles_path=args.roles,
            staging_path=args.staging,
            provenance_path=args.provenance,
            output_path=args.output,
            exclusion_manifest=args.pilot_exclusions,
            verified_relations=args.verified_relations,
        )
    except (ExperimentValidationError, ContractError, FileExistsError) as exc:
        code = exc.code if isinstance(exc, ExperimentValidationError) else str(exc)
        print(json.dumps({"status": "FAILED", "failures": [code]}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
