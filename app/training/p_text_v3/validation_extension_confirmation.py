"""Read-only validation of AI-assisted batch confirmation for 194 candidates.

Passing this validator proves record integrity only. It never promotes the
records to Human Ground Truth, a training dataset, or a threshold-selection
dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from .candidate_generation import (
    IdentityExclusions,
    PoolRow,
    _SelectionState,
    _identity_reason,
    load_preliminary_pilot,
    load_relation_audit,
    load_v2_exclusions,
)
from .data_contract import exclusion_matches, file_sha256, load_pilot_exclusions
from .single_reviewer_experiment import RATIONALE_CODES, SIGNAL_TYPES
from .validation_extension import ROLE


VERIFIED_STATUS = "AI_ASSISTED_BATCH_CONFIRMATION_VERIFIED_NOT_PUBLISHABLE"
FAILED_STATUS = "AI_ASSISTED_BATCH_CONFIRMATION_VALIDATION_FAILED"
EXPECTED_COUNT = 194
EXPECTED_DISTRIBUTION = {"PRESENT": 42, "ABSENT": 136, "UNCERTAIN": 11, "NOT_ASSESSED": 5}
STAGING_FIELDS = (
    "annotation_id", "annotator_id", "text_status", "signal_decision", "signal_types",
    "other_review_needed", "rationale_codes", "rationale_note", "guideline_version",
    "annotation_round", "annotated_at", "display_order", "proposal_origin",
    "batch_confirmation_status", "human_per_row_review_evidence",
    "official_human_ground_truth", "eligible_for_training", "eligible_for_threshold_selection",
)


def _read_csv(path: str | Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return tuple(reader.fieldnames or ()), [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError("CSV_READ_FAILED") from exc


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("PROVENANCE_READ_FAILED") from exc
    if not isinstance(value, dict):
        raise ValueError("PROVENANCE_INVALID")
    return value


def _unique(rows: Iterable[Mapping[str, Any]], prefix: str, failures: list[str]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        annotation_id = str(row.get("annotation_id", "")).strip()
        if not annotation_id:
            failures.append(prefix + "_ANNOTATION_ID_MISSING")
            continue
        if annotation_id in result:
            failures.append(prefix + "_ANNOTATION_ID_DUPLICATE")
            continue
        result[annotation_id] = row
    return result


def _pipe_values(value: object, allowed: frozenset[str], prefix: str, failures: list[str]) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(value or "").split("|") if item.strip())
    if len(values) != len(set(values)):
        failures.append(prefix + "_DUPLICATE")
    if set(values).difference(allowed):
        failures.append(prefix + "_UNKNOWN")
    return values


def _validate_label(row: Mapping[str, Any], failures: list[str]) -> None:
    text_status = str(row.get("text_status", ""))
    decision = str(row.get("signal_decision", ""))
    signals = _pipe_values(row.get("signal_types"), SIGNAL_TYPES, "SIGNAL_TYPES", failures)
    allowed = {
        ("SUFFICIENT", "PRESENT"),
        ("SUFFICIENT", "ABSENT"),
        ("SUFFICIENT", "UNCERTAIN"),
        ("INSUFFICIENT", "NOT_ASSESSED"),
        ("INVALID", "NOT_ASSESSED"),
    }
    if (text_status, decision) not in allowed:
        failures.append("LABEL_COMBINATION_INVALID")
    if decision == "PRESENT" and not signals:
        failures.append("PRESENT_SIGNAL_TYPES_REQUIRED")
    if decision in {"ABSENT", "NOT_ASSESSED"} and signals:
        failures.append("SIGNAL_TYPES_MUST_BE_EMPTY")
    _pipe_values(row.get("rationale_codes"), RATIONALE_CODES, "RATIONALE_CODES", failures)
    if not str(row.get("rationale_note", "")).strip():
        failures.append("RATIONALE_NOTE_REQUIRED")
    if row.get("other_review_needed") not in {"YES", "NO"}:
        failures.append("OTHER_REVIEW_NEEDED_INVALID")


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    staging_path: str | Path,
    available_source_artifacts: Sequence[str | Path],
    failures: list[str],
) -> dict[str, Any]:
    expected_values = {
        "record_type": "P_TEXT_V3_VALIDATION_EXTENSION_BATCH_CONFIRMATION",
        "schema_version": "review-extension-ai-assisted-batch-record-v0.1",
        "experiment_status": "SINGLE_REVIEWER_AI_ASSISTED_EXPERIMENT",
        "candidate_role": ROLE,
        "source_dataset_status": "UNLABELED_HUMAN_REVIEW_REQUIRED",
        "source_row_count": EXPECTED_COUNT,
        "selected_candidate_count": EXPECTED_COUNT,
    }
    for field, expected in expected_values.items():
        if provenance.get(field) != expected:
            failures.append("PROVENANCE_" + field.upper() + "_INVALID")
    batch = provenance.get("batch_confirmation")
    if not isinstance(batch, dict):
        failures.append("BATCH_CONFIRMATION_INVALID")
        batch = {}
    expected_batch = {
        "status": "USER_REPORTED_AGREEMENT_WITH_AI_DRAFT",
        "recorded_annotation_id_count": EXPECTED_COUNT,
        "human_independent_review_verified": False,
        "per_row_human_annotation_evidence_present": False,
        "independent_reviewers_completed": 0,
        "official_human_ground_truth": False,
    }
    for field, expected in expected_batch.items():
        if batch.get(field) != expected:
            failures.append("BATCH_CONFIRMATION_" + field.upper() + "_INVALID")
    ai = provenance.get("ai_assisted")
    if not isinstance(ai, dict):
        failures.append("AI_ASSISTED_PROVENANCE_INVALID")
        ai = {}
    if ai.get("used") is not True or ai.get("present_is_actual_fraud_count") is not False:
        failures.append("AI_ASSISTED_POLICY_INVALID")
    if ai.get("label_distribution") != EXPECTED_DISTRIBUTION:
        failures.append("PROVENANCE_LABEL_DISTRIBUTION_INVALID")
    staging = provenance.get("staging")
    if not isinstance(staging, dict):
        failures.append("STAGING_PROVENANCE_INVALID")
        staging = {}
    actual_staging_sha = file_sha256(staging_path)
    if str(staging.get("sha256", "")).lower() != actual_staging_sha:
        failures.append("STAGING_SHA256_MISMATCH")
    expected_false = (
        "eligible_for_training", "eligible_for_threshold_selection", "official_dataset_publication_ready"
    )
    if any(staging.get(field) is not False for field in expected_false):
        failures.append("STAGING_ELIGIBILITY_POLICY_INVALID")

    available_hashes = {file_sha256(path): str(Path(path).resolve()) for path in available_source_artifacts}
    declared = provenance.get("input_artifacts")
    resolved = 0
    if not isinstance(declared, list) or not declared:
        failures.append("PROVENANCE_INPUT_ARTIFACTS_INVALID")
        declared = []
    declared_hashes: set[str] = set()
    for artifact in declared:
        digest = str(artifact.get("sha256", "")).lower() if isinstance(artifact, dict) else ""
        declared_hashes.add(digest)
        if len(digest) != 64:
            failures.append("PROVENANCE_INPUT_SHA256_INVALID")
        elif digest not in available_hashes:
            failures.append("PROVENANCE_INPUT_SHA256_UNRESOLVED")
        else:
            resolved += 1
    source_record_sha = file_sha256(available_source_artifacts[1]) if len(available_source_artifacts) > 1 else ""
    if source_record_sha not in declared_hashes:
        failures.append("SOURCE_RECORD_SHA256_NOT_DECLARED")
    return {
        "staging_sha256": actual_staging_sha,
        "declared_input_artifact_count": len(declared),
        "resolved_input_artifact_count": resolved,
    }


def _validate_generation_contract(
    *,
    metadata: Mapping[str, Any],
    master_rows: Sequence[Mapping[str, str]],
    role_rows: Sequence[Mapping[str, str]],
    failures: list[str],
) -> None:
    expected = {
        "selected_count": EXPECTED_COUNT,
        "labeling_packet_ready": False,
        "dataset_publication_ready": False,
        "existing_400_immutable": True,
        "existing_splits_modified": False,
        "automatic_labels_assigned": False,
        "model_predictions_used": False,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            failures.append("GENERATION_" + field.upper() + "_INVALID")
    if metadata.get("targets") and sum(metadata["targets"].values()) != 200:
        failures.append("GENERATION_TARGET_TOTAL_INVALID")
    if metadata.get("shortfalls") != {"PROMOTIONAL_OR_SALES_LANGUAGE": 6}:
        failures.append("GENERATION_SHORTFALL_INVALID")
    if len(master_rows) != EXPECTED_COUNT or len(role_rows) != EXPECTED_COUNT:
        failures.append("GENERATION_ROW_COUNT_INVALID")
    for row in master_rows:
        if row.get("label_status") != "UNLABELED" or row.get("extension_status") != "UNLABELED_HUMAN_REVIEW_REQUIRED":
            failures.append("MASTER_LABEL_STATUS_INVALID")
        if row.get("split_role") != ROLE:
            failures.append("MASTER_ROLE_INVALID")
    for row in role_rows:
        if row.get("candidate_role") != ROLE:
            failures.append("CANDIDATE_ROLE_INVALID")
        if row.get("eligible_for_training") != "NO" or row.get("eligible_for_threshold_selection") != "NO":
            failures.append("CANDIDATE_ROLE_ELIGIBILITY_INVALID")


def _validate_leakage(
    *,
    master_rows: Sequence[Mapping[str, str]],
    existing_master_path: str | Path,
    existing_roles_path: str | Path,
    v2_path: str | Path,
    pilot_path: str | Path,
    relations_path: str | Path,
    metadata: Mapping[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    declared_hashes = metadata.get("input_sha256") if isinstance(metadata.get("input_sha256"), dict) else {}
    actual_hashes = {
        "existing_400_master": file_sha256(existing_master_path),
        "v2": file_sha256(v2_path),
        "published_pilot": file_sha256(pilot_path),
        "verified_relations": file_sha256(relations_path),
    }
    for name, digest in actual_hashes.items():
        if str(declared_hashes.get(name, "")).lower() != digest:
            failures.append("GENERATION_INPUT_SHA256_MISMATCH:" + name)
    existing = load_preliminary_pilot(existing_master_path)
    v2 = load_v2_exclusions(v2_path)
    pilot = load_pilot_exclusions(pilot_path)
    relations = load_relation_audit(relations_path)
    _, existing_role_rows = _read_csv(existing_roles_path)
    existing_roles = Counter(row.get("role_name") for row in existing_role_rows)
    if existing_roles != Counter({"TRAIN": 220, "VALIDATION": 80, "FROZEN_TEST": 100}):
        failures.append("EXISTING_400_SPLIT_COUNTS_INVALID")
    if {row.get("annotation_id") for row in existing_role_rows} != {
        row.get("annotation_id") for row in _read_csv(existing_master_path)[1]
    }:
        failures.append("EXISTING_400_ROLE_IDS_MISMATCH")
    existing_verified = {
        group for uid in existing.row_uids for group in relations.verified_by_row.get(uid, ())
    }
    state = _SelectionState(relations)
    leakage_counts: Counter[str] = Counter()
    for raw in master_rows:
        pool_values = {field: raw.get(field, "") for field in raw if field in raw}
        row = PoolRow(pool_values, (), ())
        reason = _identity_reason(row, existing, "EXISTING_400")
        reason = reason or _identity_reason(row, v2, "V2")
        if not reason and exclusion_matches(row.values, pilot):
            reason = "PILOT_PUBLISHED_MANIFEST"
        if not reason and any(group in existing_verified for group in relations.verified_by_row.get(row.row_uid, ())):
            reason = "EXISTING_400_VERIFIED_RELATION"
        if reason:
            leakage_counts[reason] += 1
        cross = state.reason(row)
        if cross:
            leakage_counts[cross] += 1
        else:
            state.add(row)
    if leakage_counts:
        failures.append("LEAKAGE_DETECTED")
    return {
        "existing_400_split_counts": dict(sorted(existing_roles.items())),
        "extension_cross_scope_leakage": dict(sorted(leakage_counts.items())),
        "generation_input_sha256": actual_hashes,
    }


def validate_bundle(
    *,
    extension_root: str | Path,
    staging_path: str | Path,
    provenance_path: str | Path,
    source_record_path: str | Path,
    existing_master_path: str | Path,
    existing_roles_path: str | Path,
    v2_path: str | Path,
    pilot_path: str | Path,
    relations_path: str | Path,
    provenance_source_artifacts: Sequence[str | Path] = (),
) -> dict[str, Any]:
    root = Path(extension_root)
    paths = {
        "master": root / "master_validation_extension_v1.csv",
        "roles": root / "candidate_role_manifest.csv",
        "blind": root / "blind_review_packet.csv",
        "metadata": root / "candidate_generation_metadata.json",
        "working": root / "review_working_194.csv",
        "staging": Path(staging_path),
        "provenance": Path(provenance_path),
        "source_record": Path(source_record_path),
        "existing_master": Path(existing_master_path),
        "existing_roles": Path(existing_roles_path),
        "v2": Path(v2_path),
        "pilot": Path(pilot_path),
        "relations": Path(relations_path),
        **{f"provenance_source_{index}": Path(path) for index, path in enumerate(provenance_source_artifacts)},
    }
    before = {name: file_sha256(path) for name, path in paths.items()}
    failures: list[str] = []
    master_fields, master_rows = _read_csv(paths["master"])
    _, role_rows = _read_csv(paths["roles"])
    _, blind_rows = _read_csv(paths["blind"])
    _, working_rows = _read_csv(paths["working"])
    staging_fields, staging_rows = _read_csv(paths["staging"])
    metadata = _load_json(paths["metadata"])
    provenance = _load_json(paths["provenance"])
    if staging_fields != STAGING_FIELDS:
        failures.append("STAGING_SCHEMA_INVALID")
    masters = _unique(master_rows, "MASTER", failures)
    roles = _unique(role_rows, "ROLE", failures)
    blinds = _unique(blind_rows, "BLIND", failures)
    workings = _unique(working_rows, "WORKING", failures)
    stagings = _unique(staging_rows, "STAGING", failures)
    expected_ids = set(masters)
    if len(expected_ids) != EXPECTED_COUNT:
        failures.append("MASTER_COUNT_INVALID")
    for name, values in (("ROLE", roles), ("BLIND", blinds), ("WORKING", workings), ("STAGING", stagings)):
        if set(values) != expected_ids:
            failures.append(name + "_ANNOTATION_IDS_MISMATCH")
    distribution = Counter()
    for row in staging_rows:
        _validate_label(row, failures)
        distribution[str(row.get("signal_decision", ""))] += 1
        expected_row_values = {
            "annotator_id": "",
            "annotated_at": "",
            "proposal_origin": "AI_ASSISTED_DRAFT",
            "batch_confirmation_status": "USER_REPORTED_AGREEMENT_WITH_AI_DRAFT",
            "human_per_row_review_evidence": "NOT_RECORDED",
            "official_human_ground_truth": "NO",
            "eligible_for_training": "NO",
            "eligible_for_threshold_selection": "NO",
            "guideline_version": "ptext-labeling-guide-v0.3",
            "annotation_round": "1",
        }
        for field, expected in expected_row_values.items():
            if row.get(field) != expected:
                failures.append("STAGING_" + field.upper() + "_INVALID")
    if dict(distribution) != EXPECTED_DISTRIBUTION:
        failures.append("STAGING_LABEL_DISTRIBUTION_INVALID")
    _validate_generation_contract(
        metadata=metadata, master_rows=master_rows, role_rows=role_rows, failures=failures
    )
    provenance_report = _validate_provenance(
        provenance,
        staging_path=paths["staging"],
        available_source_artifacts=(paths["working"], paths["source_record"], *provenance_source_artifacts),
        failures=failures,
    )
    leakage_report = _validate_leakage(
        master_rows=master_rows,
        existing_master_path=paths["existing_master"],
        existing_roles_path=paths["existing_roles"],
        v2_path=paths["v2"],
        pilot_path=paths["pilot"],
        relations_path=paths["relations"],
        metadata=metadata,
        failures=failures,
    )
    after = {name: file_sha256(path) for name, path in paths.items()}
    if before != after:
        failures.append("INPUT_MUTATED_DURING_VALIDATION")
    unique_failures = sorted(set(failures))
    return {
        "status": VERIFIED_STATUS if not unique_failures else FAILED_STATUS,
        "failures": unique_failures,
        "record_count": len(stagings),
        "annotation_id_sets_match": not any("ANNOTATION_IDS_MISMATCH" in item for item in unique_failures),
        "label_distribution": {key: distribution[key] for key in EXPECTED_DISTRIBUTION},
        "official_human_ground_truth": False,
        "human_independent_review_verified": False,
        "human_per_row_review_evidence_present": False,
        "eligible_for_training": False,
        "eligible_for_threshold_selection": False,
        "dataset_publication_ready": False,
        "quota": {"target": 200, "selected": 194, "shortfall": 6},
        "labeling_packet_ready": False,
        "shortfall_interpretation": "PROMOTIONAL_OR_SALES_LANGUAGE quota shortfall; not evidence of incomplete IDs within the 194-record packet",
        "provenance": provenance_report,
        "leakage": leakage_report,
        "source_record_sha256": before["source_record"],
        "inputs_unchanged": before == after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-root", required=True)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--source-record", required=True)
    parser.add_argument("--existing-400-master", required=True)
    parser.add_argument("--existing-400-roles", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--pilot-exclusions", required=True)
    parser.add_argument("--verified-relations", required=True)
    parser.add_argument("--provenance-source-artifact", action="append", default=[])
    args = parser.parse_args()
    try:
        report = validate_bundle(
            extension_root=args.extension_root,
            staging_path=args.staging,
            provenance_path=args.provenance,
            source_record_path=args.source_record,
            existing_master_path=args.existing_400_master,
            existing_roles_path=args.existing_400_roles,
            v2_path=args.v2,
            pilot_path=args.pilot_exclusions,
            relations_path=args.verified_relations,
            provenance_source_artifacts=args.provenance_source_artifact,
        )
    except Exception as exc:  # CLI must emit content-free codes, never private rows.
        report = {
            "status": FAILED_STATUS,
            "failures": ["VALIDATION_EXCEPTION:" + type(exc).__name__],
            "official_human_ground_truth": False,
            "human_independent_review_verified": False,
            "eligible_for_training": False,
            "eligible_for_threshold_selection": False,
            "dataset_publication_ready": False,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != VERIFIED_STATUS:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
