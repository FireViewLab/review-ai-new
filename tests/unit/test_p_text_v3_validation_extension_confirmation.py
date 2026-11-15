from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

from openpyxl import Workbook

from app.training.p_text_v3.candidate_generation import MASTER_FIELDS, POOL_FIELDS, pool_content_hash
from app.training.p_text_v3.data_contract import (
    PILOT_EXCLUDED_FROM,
    PILOT_MANIFEST_FIELDS,
    PILOT_MANIFEST_VERSION,
    file_sha256,
    pilot_entity_hash,
)
from app.training.p_text_v3.validation_extension import EXTENSION_MASTER_FIELDS, ROLE_FIELDS, ROLE
from app.training.p_text_v3.validation_extension_confirmation import (
    EXPECTED_DISTRIBUTION,
    FAILED_STATUS,
    STAGING_FIELDS,
    VERIFIED_STATUS,
    validate_bundle,
)


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pool_values(index: int, prefix: str) -> dict[str, str]:
    content = f"synthetic {prefix} private content {index}"
    return {
        "pool_id": f"{prefix}-pool-{index}", "row_uid": f"{prefix}-row-{index}",
        "sampling_group": "CANDIDATE", "sampling_reason": "fixture", "label_status": "UNLABELED",
        "platform": "fixture", "source_row": str(index), "product_id": f"{prefix}-product-{index}",
        "review_id": f"{prefix}-review-{index}", "product_category_raw": "fixture",
        "normalized_category": "FIXTURE", "content": content,
        "normalized_content_hash": pool_content_hash(content), "content_length": str(len(content)),
        "duplicate_cluster_id": "", "candidate_sources": "", "original_candidate_member": "0",
        "original_control_eligible": "0", "seed": "1", "policy_version": "fixture",
    }


def make_bundle(tmp_path: Path) -> dict:
    root = tmp_path / "extension"
    root.mkdir()
    confirmation = root / "confirmation"
    confirmation.mkdir()
    extension_rows = []
    role_rows = []
    blind_rows = []
    working_rows = []
    staging_rows = []
    decisions = (
        ["PRESENT"] * EXPECTED_DISTRIBUTION["PRESENT"]
        + ["ABSENT"] * EXPECTED_DISTRIBUTION["ABSENT"]
        + ["UNCERTAIN"] * EXPECTED_DISTRIBUTION["UNCERTAIN"]
        + ["NOT_ASSESSED"] * EXPECTED_DISTRIBUTION["NOT_ASSESSED"]
    )
    for index, decision in enumerate(decisions):
        annotation_id = f"annotation-{index:03d}"
        values = pool_values(index, "extension")
        extension_rows.append(
            values
            | {
                "annotation_id": annotation_id, "split_role": ROLE,
                "selection_category": "PRODUCT_DESCRIPTION_LIKE", "retrieval_evidence": "fixture",
                "near_duplicate_review_required": "NO", "candidate_band": "ENRICHED_RETRIEVAL",
                "extension_status": "UNLABELED_HUMAN_REVIEW_REQUIRED",
            }
        )
        role_rows.append(
            {"annotation_id": annotation_id, "candidate_role": ROLE,
             "eligible_for_training": "NO", "eligible_for_threshold_selection": "NO"}
        )
        blind_rows.append({"annotation_id": annotation_id, "display_order": str(index + 1), "content": values["content"]})
        working_rows.append({"annotation_id": annotation_id})
        staging_rows.append(
            {
                "annotation_id": annotation_id, "annotator_id": "",
                "text_status": "INSUFFICIENT" if decision == "NOT_ASSESSED" else "SUFFICIENT",
                "signal_decision": decision,
                "signal_types": "PRODUCT_DESCRIPTION_LIKE" if decision == "PRESENT" else "",
                "other_review_needed": "YES" if decision in {"UNCERTAIN", "NOT_ASSESSED"} else "NO",
                "rationale_codes": "PROPOSED_BOUNDARY_PRODUCT_DESCRIPTION" if decision != "ABSENT" else "",
                "rationale_note": "synthetic rationale", "guideline_version": "ptext-labeling-guide-v0.3",
                "annotation_round": "1", "annotated_at": "", "display_order": str(index + 1),
                "proposal_origin": "AI_ASSISTED_DRAFT",
                "batch_confirmation_status": "USER_REPORTED_AGREEMENT_WITH_AI_DRAFT",
                "human_per_row_review_evidence": "NOT_RECORDED", "official_human_ground_truth": "NO",
                "eligible_for_training": "NO", "eligible_for_threshold_selection": "NO",
            }
        )
    write_csv(root / "master_validation_extension_v1.csv", EXTENSION_MASTER_FIELDS, extension_rows)
    write_csv(root / "candidate_role_manifest.csv", ROLE_FIELDS, role_rows)
    write_csv(root / "blind_review_packet.csv", ("annotation_id", "display_order", "content"), blind_rows)
    write_csv(root / "review_working_194.csv", ("annotation_id",), working_rows)
    staging = confirmation / "staging.csv"
    write_csv(staging, STAGING_FIELDS, staging_rows)

    existing_master = tmp_path / "existing_master.csv"
    existing_rows = []
    existing_roles = []
    for index in range(400):
        role = "TRAIN" if index < 220 else "VALIDATION" if index < 300 else "FROZEN_TEST"
        existing_rows.append(pool_values(index, "existing") | {"annotation_id": f"existing-{index}"})
        existing_roles.append(
            {"annotation_id": f"existing-{index}", "split": {"TRAIN": "train", "VALIDATION": "validation", "FROZEN_TEST": "test"}[role], "role_name": role}
        )
    write_csv(existing_master, POOL_FIELDS + ("annotation_id",), existing_rows)
    existing_roles_path = tmp_path / "existing_roles.csv"
    write_csv(existing_roles_path, ("annotation_id", "split", "role_name"), existing_roles)

    v2 = tmp_path / "v2.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "dataset"
    sheet.append(["platform", "product_id", "review_id", "content", "label", "label_origin", "group_id", "split"])
    values = pool_values(1, "v2")
    sheet.append([values["platform"], values["product_id"], values["review_id"], values["content"], "NORMAL", "fixture", "v2-group", "train"])
    workbook.save(v2)

    pilot = tmp_path / "pilot.csv"
    write_csv(
        pilot,
        PILOT_MANIFEST_FIELDS,
        [{
            "manifest_version": PILOT_MANIFEST_VERSION, "pilot_version": "fixture-v1",
            "sampling_seed": "1", "policy_version": "fixture-v1", "annotation_id": "pilot-1",
            "entity_type": "ROW_UID", "entity_key_hash": pilot_entity_hash("ROW_UID", "pilot-row"),
            "exclusion_reason": "fixture", "joint_leakage_group_id": "a" * 24,
            "excluded_from": PILOT_EXCLUDED_FROM,
        }],
    )
    relations = tmp_path / "relations.csv"
    write_csv(relations, ("row_uid", "relation_type", "group_id", "validation_status"), [])

    metadata = {
        "selected_count": 194, "labeling_packet_ready": False, "dataset_publication_ready": False,
        "existing_400_immutable": True, "existing_splits_modified": False,
        "automatic_labels_assigned": False, "model_predictions_used": False,
        "targets": {"PROMOTIONAL_OR_SALES_LANGUAGE": 200},
        "shortfalls": {"PROMOTIONAL_OR_SALES_LANGUAGE": 6},
        "input_sha256": {
            "existing_400_master": file_sha256(existing_master), "v2": file_sha256(v2),
            "published_pilot": file_sha256(pilot), "verified_relations": file_sha256(relations),
        },
    }
    (root / "candidate_generation_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    source_record = confirmation / "source_record.xlsx"
    workbook = Workbook()
    workbook.active.append(["synthetic source record; no private content"])
    workbook.save(source_record)
    provenance = {
        "record_type": "P_TEXT_V3_VALIDATION_EXTENSION_BATCH_CONFIRMATION",
        "schema_version": "review-extension-ai-assisted-batch-record-v0.1",
        "experiment_status": "SINGLE_REVIEWER_AI_ASSISTED_EXPERIMENT", "candidate_role": ROLE,
        "source_dataset_status": "UNLABELED_HUMAN_REVIEW_REQUIRED", "source_row_count": 194,
        "selected_candidate_count": 194,
        "batch_confirmation": {
            "status": "USER_REPORTED_AGREEMENT_WITH_AI_DRAFT", "recorded_annotation_id_count": 194,
            "human_independent_review_verified": False, "per_row_human_annotation_evidence_present": False,
            "independent_reviewers_completed": 0, "official_human_ground_truth": False,
        },
        "ai_assisted": {"used": True, "present_is_actual_fraud_count": False, "label_distribution": EXPECTED_DISTRIBUTION},
        "staging": {
            "sha256": file_sha256(staging), "eligible_for_training": False,
            "eligible_for_threshold_selection": False, "official_dataset_publication_ready": False,
        },
        "input_artifacts": [
            {"file": "review_working_194.csv", "sha256": file_sha256(root / "review_working_194.csv")},
            {"file": "source_record.xlsx", "sha256": file_sha256(source_record)},
        ],
    }
    provenance_path = confirmation / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return {
        "root": root, "staging": staging, "staging_rows": staging_rows,
        "provenance": provenance_path, "source_record": source_record,
        "existing_master": existing_master, "existing_roles": existing_roles_path,
        "v2": v2, "pilot": pilot, "relations": relations,
    }


def validate(bundle):
    return validate_bundle(
        extension_root=bundle["root"], staging_path=bundle["staging"],
        provenance_path=bundle["provenance"], source_record_path=bundle["source_record"],
        existing_master_path=bundle["existing_master"], existing_roles_path=bundle["existing_roles"],
        v2_path=bundle["v2"], pilot_path=bundle["pilot"], relations_path=bundle["relations"],
    )


def rewrite_staging(bundle, rows) -> None:
    write_csv(bundle["staging"], STAGING_FIELDS, rows)
    provenance = json.loads(bundle["provenance"].read_text(encoding="utf-8"))
    provenance["staging"]["sha256"] = file_sha256(bundle["staging"])
    bundle["provenance"].write_text(json.dumps(provenance), encoding="utf-8")


def test_valid_batch_confirmation_is_verified_but_not_publishable(tmp_path):
    report = validate(make_bundle(tmp_path))
    assert report["status"] == VERIFIED_STATUS
    assert report["failures"] == []
    assert report["record_count"] == 194
    assert report["label_distribution"] == EXPECTED_DISTRIBUTION
    assert report["official_human_ground_truth"] is False
    assert report["human_independent_review_verified"] is False
    assert report["eligible_for_training"] is False
    assert report["eligible_for_threshold_selection"] is False
    assert report["dataset_publication_ready"] is False
    assert report["labeling_packet_ready"] is False
    assert report["quota"] == {"target": 200, "selected": 194, "shortfall": 6}
    assert report["leakage"]["extension_cross_scope_leakage"] == {}
    assert "synthetic private content" not in json.dumps(report)


def test_duplicate_or_missing_annotation_ids_fail_closed(tmp_path):
    bundle = make_bundle(tmp_path)
    rows = deepcopy(bundle["staging_rows"][:-1])
    rows.append(deepcopy(rows[0]))
    rewrite_staging(bundle, rows)
    report = validate(bundle)
    assert report["status"] == FAILED_STATUS
    assert "STAGING_ANNOTATION_ID_DUPLICATE" in report["failures"]
    assert "STAGING_ANNOTATION_IDS_MISMATCH" in report["failures"]


def test_wrong_staging_sha_is_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    provenance = json.loads(bundle["provenance"].read_text(encoding="utf-8"))
    provenance["staging"]["sha256"] = "0" * 64
    bundle["provenance"].write_text(json.dumps(provenance), encoding="utf-8")
    report = validate(bundle)
    assert "STAGING_SHA256_MISMATCH" in report["failures"]


def test_unresolved_provenance_source_sha_is_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    provenance = json.loads(bundle["provenance"].read_text(encoding="utf-8"))
    provenance["input_artifacts"][1]["sha256"] = "1" * 64
    bundle["provenance"].write_text(json.dumps(provenance), encoding="utf-8")
    report = validate(bundle)
    assert "PROVENANCE_INPUT_SHA256_UNRESOLVED" in report["failures"]
    assert "SOURCE_RECORD_SHA256_NOT_DECLARED" in report["failures"]


def test_invalid_label_combination_and_distribution_are_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    rows = deepcopy(bundle["staging_rows"])
    rows[0].update(text_status="INSUFFICIENT", signal_decision="PRESENT")
    rewrite_staging(bundle, rows)
    report = validate(bundle)
    assert "LABEL_COMBINATION_INVALID" in report["failures"]


def test_independent_human_or_training_claim_is_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    rows = deepcopy(bundle["staging_rows"])
    rows[0].update(annotator_id="reviewer_a", eligible_for_training="YES")
    rewrite_staging(bundle, rows)
    report = validate(bundle)
    assert "STAGING_ANNOTATOR_ID_INVALID" in report["failures"]
    assert "STAGING_ELIGIBLE_FOR_TRAINING_INVALID" in report["failures"]


def test_cross_existing_product_leakage_is_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    fields, rows = None, None
    with bundle["root"].joinpath("master_validation_extension_v1.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); fields, rows = tuple(reader.fieldnames or ()), list(reader)
    rows[0]["product_id"] = "existing-product-0"
    write_csv(bundle["root"] / "master_validation_extension_v1.csv", fields, rows)
    report = validate(bundle)
    assert "LEAKAGE_DETECTED" in report["failures"]
    assert report["leakage"]["extension_cross_scope_leakage"]["EXISTING_400_PRODUCT"] == 1


def test_quota_shortfall_cannot_be_represented_as_labeling_ready(tmp_path):
    bundle = make_bundle(tmp_path)
    metadata_path = bundle["root"] / "candidate_generation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["labeling_packet_ready"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report = validate(bundle)
    assert "GENERATION_LABELING_PACKET_READY_INVALID" in report["failures"]
