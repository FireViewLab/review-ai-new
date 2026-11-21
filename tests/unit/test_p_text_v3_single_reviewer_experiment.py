from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from app.training.p_text_v3.data_contract import VerifiedRelations, load_dataset
from app.training.p_text_v3.single_reviewer_experiment import (
    CONFIRMATION_STATUS,
    EXPERIMENT_STATUS,
    STAGING_FIELDS,
    ExperimentValidationError,
    run,
    validate_experiment,
)


SPLIT_COUNTS = {"train": 220, "validation": 80, "test": 100}


def make_bundle() -> dict:
    masters = []
    roles = []
    staging = []
    position = 0
    for split, count in SPLIT_COUNTS.items():
        for index in range(count):
            annotation_id = f"a{position:03d}"
            decision = ("PRESENT", "ABSENT", "UNCERTAIN")[index % 3]
            masters.append(
                {
                    "annotation_id": annotation_id,
                    "row_uid": f"row-{position}",
                    "platform": "fixture",
                    "product_id": f"product-{position}",
                    "review_id": f"review-{position}",
                    "content": f"synthetic content {position}",
                    "duplicate_cluster_id": "",
                    "sampling_group": "CONTROL",
                    "split_role": {"train": "TRAIN", "validation": "VALIDATION", "test": "FROZEN_TEST"}[split],
                }
            )
            roles.append(
                {
                    "annotation_id": annotation_id,
                    "split": split,
                    "role_name": {"train": "TRAIN", "validation": "VALIDATION", "test": "FROZEN_TEST"}[split],
                }
            )
            staging.append(
                {
                    "annotation_id": annotation_id,
                    "annotator_id": "",
                    "text_status": "SUFFICIENT",
                    "signal_decision": decision,
                    "signal_types": "PRODUCT_DESCRIPTION_LIKE" if decision == "PRESENT" else "",
                    "other_review_needed": "NO",
                    "rationale_codes": "PROPOSED_BOUNDARY_PRODUCT_DESCRIPTION",
                    "rationale_note": "",
                    "guideline_version": "ptext-labeling-guide-v0.2",
                    "annotation_round": "1",
                    "annotated_at": "",
                }
            )
            position += 1
    provenance = {
        "artifact_status": "AI_ASSISTED_STAGING_NOT_FORMAL_REVIEW",
        "experiment_status": EXPERIMENT_STATUS,
        "user_confirmation_status": CONFIRMATION_STATUS,
        "review_mode": "ONE_PERSON_AI_ASSISTED_CONFIRMED_DRAFT",
        "reviewer_count": 1,
        "ai_assistance_used": True,
        "candidate_pack": "synthetic_candidate_400",
        "confirmed_source": "synthetic_confirmed.xlsx",
        "confirmed_rows": 400,
        "formal_reviewer_a_submission_completed": False,
        "reviewer_b_independent_review_completed": False,
    }
    return {"masters": masters, "roles": roles, "staging": staging, "provenance": provenance}


def validate(bundle: dict, **overrides):
    values = dict(
        master_rows=bundle["masters"],
        role_rows=bundle["roles"],
        staging_rows=bundle["staging"],
        provenance=bundle["provenance"],
    )
    values.update(overrides)
    return validate_experiment(**values)


def write_csv(path: Path, rows, fields=None) -> None:
    fields = tuple(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_valid_ai_assisted_draft_passes_with_fixed_roles_and_exclusions():
    report, rows = validate(make_bundle())
    assert report["status"] == "PASSED"
    assert report["experiment_status"] == EXPERIMENT_STATUS
    assert report["official_human_ground_truth"] is False
    assert report["fixed_split_counts"] == SPLIT_COUNTS
    assert report["binary_eligible_count"] + report["binary_excluded_count"] == 400
    assert report["binary_excluded_count"] > 0
    assert all(row["label_origin"] == "single_reviewer_ai_assisted_confirmed_draft" for row in rows)
    assert report["frozen_test_policy"]["may_train"] is False
    assert report["frozen_test_policy"]["prior_ai_assisted_label_exposure"] is True
    assert "synthetic content" not in json.dumps(report)


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_annotation_ids_must_be_complete_and_unique(mode):
    bundle = make_bundle()
    if mode == "missing":
        bundle["staging"] = bundle["staging"][:-1]
    else:
        bundle["staging"].append(deepcopy(bundle["staging"][0]))
    with pytest.raises(ExperimentValidationError, match="STAGING_"):
        validate(bundle)


def test_invalid_label_combination_is_rejected():
    bundle = make_bundle()
    bundle["staging"][0].update(
        text_status="INSUFFICIENT",
        signal_decision="PRESENT",
        signal_types="PRODUCT_DESCRIPTION_LIKE",
    )
    with pytest.raises(ExperimentValidationError, match="LABEL_COMBINATION_INVALID"):
        validate(bundle)


def test_missing_provenance_is_rejected():
    bundle = make_bundle()
    bundle["provenance"] = {}
    with pytest.raises(ExperimentValidationError, match="PROVENANCE_ARTIFACT_STATUS_INVALID"):
        validate(bundle)


def test_wrong_explicit_confirmation_status_is_rejected():
    bundle = make_bundle()
    bundle["provenance"]["user_confirmation_status"] = "OFFICIAL"
    with pytest.raises(ExperimentValidationError, match="USER_CONFIRMATION_STATUS_INVALID"):
        validate(bundle)


def test_fixed_split_quota_cannot_be_changed():
    bundle = make_bundle()
    bundle["roles"][-1]["split"] = "train"
    bundle["masters"][-1]["split_role"] = "TRAIN"
    bundle["roles"][-1]["role_name"] = "TRAIN"
    with pytest.raises(ExperimentValidationError, match="FIXED_SPLIT_QUOTA_MISMATCH"):
        validate(bundle)


@pytest.mark.parametrize("scope", ["product", "normalized_content_hash", "duplicate_cluster"])
def test_cross_split_identity_leakage_is_rejected(scope):
    bundle = make_bundle()
    if scope == "product":
        bundle["masters"][-1]["product_id"] = bundle["masters"][0]["product_id"]
    elif scope == "normalized_content_hash":
        bundle["masters"][-1]["content"] = bundle["masters"][0]["content"]
    else:
        bundle["masters"][0]["duplicate_cluster_id"] = "cluster-shared"
        bundle["masters"][-1]["duplicate_cluster_id"] = "cluster-shared"
    with pytest.raises(ExperimentValidationError, match=f"CROSS_SPLIT_LEAKAGE:{scope}"):
        validate(bundle)


def test_cross_split_verified_near_duplicate_is_rejected():
    bundle = make_bundle()
    relations = VerifiedRelations(
        {
            bundle["masters"][0]["row_uid"]: (("NEAR_DUPLICATE", "shared"),),
            bundle["masters"][-1]["row_uid"]: (("NEAR_DUPLICATE", "shared"),),
        },
        "fixture-sha",
        2,
    )
    with pytest.raises(ExperimentValidationError, match="CROSS_SPLIT_LEAKAGE:verified_relation"):
        validate(bundle, relations=relations)


def test_draft_cannot_be_represented_as_official_reviewer_submission():
    bundle = make_bundle()
    bundle["staging"][0]["annotator_id"] = "reviewer_a"
    with pytest.raises(ExperimentValidationError, match="OFFICIAL_REVIEW_REPRESENTATION_FORBIDDEN"):
        validate(bundle)


def test_frozen_test_role_cannot_be_moved_into_train_even_if_quotas_stay_equal():
    bundle = make_bundle()
    train_index = next(index for index, row in enumerate(bundle["roles"]) if row["split"] == "train")
    test_index = next(index for index, row in enumerate(bundle["roles"]) if row["split"] == "test")
    bundle["roles"][train_index]["split"] = "test"
    bundle["roles"][test_index]["split"] = "train"
    with pytest.raises(ExperimentValidationError, match="MASTER_ROLE_MISMATCH"):
        validate(bundle)


def test_separate_builder_writes_non_official_experiment_workbook_without_resplitting(tmp_path):
    bundle = make_bundle()
    master = tmp_path / "master.csv"
    roles = tmp_path / "roles.csv"
    staging = tmp_path / "staging.csv"
    provenance = tmp_path / "provenance.json"
    write_csv(master, bundle["masters"])
    write_csv(roles, bundle["roles"])
    write_csv(staging, bundle["staging"], STAGING_FIELDS)
    provenance.write_text(json.dumps(bundle["provenance"]), encoding="utf-8")
    output = tmp_path / "experiment" / "dataset.xlsx"

    result = run(
        master_path=master,
        roles_path=roles,
        staging_path=staging,
        provenance_path=provenance,
        output_path=output,
    )

    assert result["experiment_status"] == EXPERIMENT_STATUS
    assert result["official_human_ground_truth"] is False
    assert result["split_generated"] is False
    loaded = load_dataset(output)
    assert loaded.summary["counts"] == result["validation"]["canonical_binary_validation"]["counts"]
    with pytest.raises(FileExistsError, match="OUTPUT_ALREADY_EXISTS"):
        run(
            master_path=master,
            roles_path=roles,
            staging_path=staging,
            provenance_path=provenance,
            output_path=output,
        )
