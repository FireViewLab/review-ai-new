import csv
from pathlib import Path

import openpyxl
import pytest

from app.training.p_text_v3.build_dataset import adjudicated_binary_label, build_human_rows, run
from app.training.p_text_v3.data_contract import (
    ContractError,
    PILOT_EXCLUDED_FROM,
    PILOT_MANIFEST_FIELDS,
    PILOT_MANIFEST_VERSION,
    PilotExclusions,
    load_dataset,
    load_pilot_exclusions,
    load_verified_relations,
    pilot_entity_hash,
    validate_rows,
)


def base_rows():
    return [
        {
            "platform": "platform-a",
            "product_id": f"product-{split}-{label}",
            "review_id": f"review-{split}-{label}",
            "content": f"fixture text {split} {label}",
            "label": label,
            "label_origin": "synthetic_fixture",
            "group_id": f"group-{split}-{label}",
            "split": split,
            "row_uid": f"row-{split}-{label}",
            "duplicate_cluster_id": "",
        }
        for split in ("train", "validation", "test")
        for label in ("NORMAL", "SUSPICIOUS")
    ]


def write_workbook(path: Path, rows):
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "dataset"
    headers = list(rows[0])
    sheet.append(headers)
    for row in rows:
        sheet.append([row.get(field) for field in headers])
    book.save(path)
    book.close()


def write_csv(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_row(entity_type, digest, *, annotation="annotation-1"):
    return {
        "manifest_version": PILOT_MANIFEST_VERSION,
        "pilot_version": "ptext-pilot-v1",
        "sampling_seed": "20261108",
        "policy_version": "ptext-pilot-v1",
        "annotation_id": annotation,
        "entity_type": entity_type,
        "entity_key_hash": digest,
        "exclusion_reason": "SYNTHETIC_TEST",
        "joint_leakage_group_id": "a" * 24,
        "excluded_from": PILOT_EXCLUDED_FROM,
    }


def test_fixed_split_loader_preserves_splits_and_hides_content_from_metadata(tmp_path):
    path = tmp_path / "dataset.xlsx"
    rows = base_rows()
    write_workbook(path, rows)
    original = path.read_bytes()

    data = load_dataset(path)

    assert path.read_bytes() == original
    assert {name: len(values) for name, values in data.examples.items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert data.examples["test"][0]["text"] == "fixture text test NORMAL"
    assert all("content" not in row for values in data.metadata.values() for row in values)
    assert data.summary["split_policy"].startswith("source split preserved")
    assert data.summary["leakage"] == {
        "group_id": 0,
        "product": 0,
        "normalized_content_hash": 0,
        "duplicate_cluster": 0,
        "verified_relation": 0,
    }


@pytest.mark.parametrize(
    "scope,mutate,match",
    [
        ("group", lambda rows: rows[2].update(group_id=rows[0]["group_id"]), "group_id"),
        ("product", lambda rows: rows[2].update(product_id=rows[0]["product_id"]), "product"),
        ("content", lambda rows: rows[2].update(content=rows[0]["content"]), "normalized_content_hash"),
        (
            "cluster",
            lambda rows: (rows[0].update(duplicate_cluster_id="cluster-1"), rows[2].update(duplicate_cluster_id="cluster-1")),
            "duplicate_cluster",
        ),
    ],
)
def test_cross_split_leakage_fails_closed(scope, mutate, match):
    rows = base_rows()
    mutate(rows)
    with pytest.raises(ContractError, match=match):
        validate_rows(rows)


def test_legacy_product_overlap_is_only_allowed_explicitly():
    rows = base_rows()
    rows[2]["product_id"] = rows[0]["product_id"]
    audit = validate_rows(rows, strict_product_isolation=False)
    assert audit["leakage"]["product"] == 1
    assert audit["strict_product_isolation"] is False


def test_verified_near_duplicate_relation_cannot_cross_split(tmp_path):
    relations_path = tmp_path / "relations.csv"
    write_csv(
        relations_path,
        ("row_uid", "relation_type", "group_id", "validation_status"),
        [
            {"row_uid": "row-train-NORMAL", "relation_type": "NEAR_DUPLICATE", "group_id": "near-1", "validation_status": "VERIFIED"},
            {"row_uid": "row-validation-NORMAL", "relation_type": "NEAR_DUPLICATE", "group_id": "near-1", "validation_status": "VERIFIED"},
            {"row_uid": "row-test-NORMAL", "relation_type": "NEAR_DUPLICATE", "group_id": "ignored", "validation_status": "REJECTED"},
        ],
    )
    relations = load_verified_relations(relations_path)
    assert relations.verified_row_count == 2
    with pytest.raises(ContractError, match="verified_relation"):
        validate_rows(base_rows(), relations=relations)


def test_verified_relation_can_match_pilot_exclusion(tmp_path):
    relations_path = tmp_path / "relations.csv"
    write_csv(
        relations_path,
        ("row_uid", "relation_type", "group_id", "validation_status"),
        [{"row_uid": "row-train-NORMAL", "relation_type": "NEAR_DUPLICATE", "group_id": "near-pilot", "validation_status": "VERIFIED"}],
    )
    digest = pilot_entity_hash("NEAR_DUPLICATE", "near-pilot")
    exclusions = PilotExclusions({"NEAR_DUPLICATE": frozenset({digest})}, "fixture", 1)
    with pytest.raises(ContractError, match="PILOT_RELATION_EXCLUSION_MATCH"):
        validate_rows(base_rows(), relations=load_verified_relations(relations_path), exclusions=exclusions)


def test_pilot_manifest_excludes_hashed_product_without_exposing_identifier(tmp_path):
    path = tmp_path / "pilot_exclusions.csv"
    digest = pilot_entity_hash("PLATFORM_PRODUCT", "platform-a", "product-train-NORMAL")
    write_csv(path, PILOT_MANIFEST_FIELDS, [manifest_row("PLATFORM_PRODUCT", digest)])
    exclusions = load_pilot_exclusions(path)
    assert exclusions.row_count == 1
    with pytest.raises(ContractError, match="PILOT_EXCLUSION_MATCH:1:PLATFORM_PRODUCT"):
        validate_rows(base_rows(), exclusions=exclusions)


@pytest.mark.parametrize(
    "status,decision,expected",
    [
        ("SUFFICIENT", "PRESENT", "SUSPICIOUS"),
        ("SUFFICIENT", "ABSENT", "NORMAL"),
        ("SUFFICIENT", "UNCERTAIN", None),
        ("INSUFFICIENT", "NOT_ASSESSED", None),
    ],
)
def test_adjudicated_label_mapping(status, decision, expected):
    row = {
        "adjudication_status": "resolved",
        "final_text_status": status,
        "final_signal_decision": decision,
    }
    assert adjudicated_binary_label(row) == expected


def test_human_builder_never_uses_sampling_group_as_label_or_assigns_role():
    masters = [
        {
            "annotation_id": "a1",
            "row_uid": "r1",
            "platform": "p",
            "product_id": "product",
            "review_id": "review",
            "content": "fixture review",
            "sampling_group": "CANDIDATE",
        },
        {
            "annotation_id": "a2",
            "row_uid": "r2",
            "platform": "p",
            "product_id": "product-2",
            "review_id": "review-2",
            "content": "another fixture",
            "sampling_group": "CONTROL",
        },
    ]
    finals = [
        {"annotation_id": "a1", "adjudication_status": "resolved", "final_text_status": "SUFFICIENT", "final_signal_decision": "ABSENT"},
        {"annotation_id": "a2", "adjudication_status": "resolved", "final_text_status": "SUFFICIENT", "final_signal_decision": "PRESENT"},
    ]
    rows, audit = build_human_rows(masters, finals, [{"annotation_id": "a1", "split": "train"}])
    assert len(rows) == 1
    assert rows[0]["label"] == "NORMAL"
    assert rows[0]["source_set"] == "CANDIDATE"
    assert audit["excluded_no_role"] == 1


def test_pilot_records_are_diagnostic_only_even_if_given_a_role():
    masters = [{
        "annotation_id": "pilot-a",
        "row_uid": "pilot-r",
        "platform": "p",
        "product_id": "product",
        "review_id": "review",
        "content": "pilot fixture",
        "sampling_group": "CANDIDATE",
    }]
    finals = [{"annotation_id": "pilot-a", "adjudication_status": "resolved", "final_text_status": "SUFFICIENT", "final_signal_decision": "PRESENT"}]
    digest = pilot_entity_hash("ROW_UID", "pilot-r")
    exclusions = PilotExclusions({"ROW_UID": frozenset({digest})}, "fixture", 1)
    rows, audit = build_human_rows(
        masters,
        finals,
        [{"annotation_id": "pilot-a", "split": "train"}],
        exclusions=exclusions,
    )
    assert rows == []
    assert audit["excluded_pilot"] == 1


def test_builder_end_to_end_writes_new_fixed_split_workbook_and_refuses_overwrite(tmp_path):
    masters = []
    finals = []
    roles = []
    for split in ("train", "validation", "test"):
        for decision in ("ABSENT", "PRESENT"):
            annotation = f"{split}-{decision}"
            masters.append({
                "annotation_id": annotation,
                "row_uid": f"row-{annotation}",
                "platform": "fixture",
                "product_id": f"product-{annotation}",
                "review_id": f"review-{annotation}",
                "content": f"fixture content {annotation}",
                "group_id": f"group-{annotation}",
                "duplicate_cluster_id": "",
                "sampling_group": "CANDIDATE" if decision == "PRESENT" else "CONTROL",
            })
            finals.append({
                "annotation_id": annotation,
                "adjudication_status": "resolved",
                "final_text_status": "SUFFICIENT",
                "final_signal_decision": decision,
            })
            roles.append({"annotation_id": annotation, "split": split})
    master_path, final_path, role_path = (tmp_path / name for name in ("master.csv", "final.csv", "roles.csv"))
    write_csv(master_path, tuple(masters[0]), masters)
    write_csv(final_path, tuple(finals[0]), finals)
    write_csv(role_path, ("annotation_id", "split"), roles)
    output = tmp_path / "v3.xlsx"

    result = run(master_path, final_path, role_path, output)

    assert output.is_file()
    assert result["automatic_labels_assigned"] is False
    assert result["sampling_groups_used_as_labels"] is False
    loaded = load_dataset(output)
    assert loaded.summary["counts"]["test"] == {"NORMAL": 1, "SUSPICIOUS": 1}
    with pytest.raises(FileExistsError, match="OUTPUT_ALREADY_EXISTS"):
        run(master_path, final_path, role_path, output)
