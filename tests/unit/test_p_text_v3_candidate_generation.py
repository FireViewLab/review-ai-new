from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

import pytest
from openpyxl import Workbook

from app.training.p_text_v3.candidate_generation import (
    BLIND_FIELDS,
    CandidateGenerationError,
    IdentityExclusions,
    POOL_FIELDS,
    PoolRow,
    RelationAudit,
    classify_candidate,
    load_pool,
    load_preliminary_pilot,
    load_relation_audit,
    load_v2_exclusions,
    pool_content_hash,
    run_generation,
    select_candidates,
    write_candidate_bundle,
)
from app.training.p_text_v3.data_contract import file_sha256


def _pool_row(index: int, *, group: str = "CANDIDATE", sources: str = "",
              content: str | None = None, product: str | None = None,
              cluster: str = "") -> dict[str, str]:
    text = content or f"합성 검수 문장 {index} 입니다"
    return {
        "pool_id": f"pool-{index}", "row_uid": f"row-{index}", "sampling_group": group,
        "sampling_reason": "synthetic", "label_status": "UNLABELED", "platform": "P1" if index % 2 else "P2",
        "source_row": str(index), "product_id": product or f"product-{index}", "review_id": f"review-{index}",
        "product_category_raw": "합성", "normalized_category": "cat-a" if index % 2 else "cat-b",
        "content": text, "normalized_content_hash": pool_content_hash(text), "content_length": str(len(text)),
        "duplicate_cluster_id": cluster, "candidate_sources": sources, "original_candidate_member": "0",
        "original_control_eligible": "0", "seed": "1", "policy_version": "synthetic-v1",
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _write_v2(path: Path, rows: list[dict[str, str]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "dataset"
    sheet.append(["platform", "product_id", "review_id", "content", "label", "label_origin", "group_id", "split"])
    for row in rows:
        sheet.append([row["platform"], row["product_id"], row["review_id"], row["content"],
                      "NORMAL", "HUMAN", f"g-{row['row_uid']}", "train"])
    workbook.save(path)


def _fixture_rows() -> list[dict[str, str]]:
    return [
        _pool_row(1, group="CONTROL"),
        _pool_row(2, group="CONTROL"),
        _pool_row(3, group="DIVERSITY"),
        _pool_row(4, group="DIVERSITY"),
        _pool_row(5, sources="PRODUCT_DESCRIPTION_REPLACEMENT"),
        _pool_row(6, sources="PRODUCT_DESCRIPTION_REPLACEMENT"),
        _pool_row(7, sources="PROMOTIONAL_TEMPLATE"),
        _pool_row(8, sources="PROMOTIONAL_TEMPLATE"),
        _pool_row(9, sources="PROMOTIONAL_TEMPLATE"),
        _pool_row(10, content="아직 사용하지 않았지만 효과가 확실합니다"),
        _pool_row(11, content="좋아요 정말 좋아요. 좋아요 정말 좋아요.\n끝"),
        _pool_row(12, sources="HARD_NEGATIVE_CANDIDATE"),
        _pool_row(13, sources="EXTREME_RESULT_CLAIM"),
        _pool_row(14, group="CONTROL"),
        _pool_row(15, group="DIVERSITY"),
    ]


def test_classification_is_retrieval_only_and_source_mapping() -> None:
    row = _pool_row(1, sources="PRODUCT_DESCRIPTION_REPLACEMENT|REPEATED_SHORT_TEMPLATE",
                    content="아직 사용하지 않았지만 효과가 확실합니다")
    row["model_prediction"] = "SUSPICIOUS"
    categories, evidence = classify_candidate(row)
    assert "PRODUCT_DESCRIPTION_LIKE" in categories
    assert "NO_USE_ASSERTIVE_CLAIM" in categories
    assert "WITHIN_REVIEW_REPETITION" not in categories
    assert all("SUSPICIOUS" not in item and "NORMAL" not in item for item in evidence)


def test_pool_groups_unicode_multiline_and_unlabeled_contract(tmp_path: Path) -> None:
    rows = [_pool_row(1, group="CANDIDATE", content="유니코드🙂\n둘째 줄"),
            _pool_row(2, group="CONTROL"), _pool_row(3, group="DIVERSITY")]
    path = tmp_path / "pool.csv"
    _write_csv(path, POOL_FIELDS, rows)
    before = file_sha256(path)
    loaded = load_pool(path)
    assert {row.sampling_group for row in loaded} == {"CANDIDATE", "CONTROL", "DIVERSITY"}
    assert loaded[0].content == "유니코드🙂\n둘째 줄"
    assert file_sha256(path) == before
    rows[0]["label_status"] = "SUSPICIOUS"
    _write_csv(tmp_path / "bad.csv", POOL_FIELDS, rows)
    with pytest.raises(CandidateGenerationError, match="POOL_MUST_BE_UNLABELED"):
        load_pool(tmp_path / "bad.csv")


def test_pilot_v2_exclusions_and_cross_role_verified_relation(tmp_path: Path) -> None:
    raw = _fixture_rows()
    pool_path = tmp_path / "pool.csv"
    _write_csv(pool_path, POOL_FIELDS, raw)
    rows = load_pool(pool_path)

    pilot_fields = POOL_FIELDS + ("annotation_id",)
    pilot_row = dict(raw[0], annotation_id="pilot-1")
    pilot_path = tmp_path / "pilot.csv"
    _write_csv(pilot_path, pilot_fields, [pilot_row])
    pilot = load_preliminary_pilot(pilot_path)

    v2_path = tmp_path / "v2.xlsx"
    _write_v2(v2_path, [raw[2]])
    v2 = load_v2_exclusions(v2_path)

    relation_path = tmp_path / "relations.csv"
    relation_rows = [
        {"row_uid": "row-7", "relation_type": "NEAR_DUPLICATE", "group_id": "near-a", "validation_status": "VERIFIED"},
        {"row_uid": "row-8", "relation_type": "NEAR_DUPLICATE", "group_id": "near-a", "validation_status": "VERIFIED"},
        {"row_uid": "row-11", "relation_type": "NEAR_DUPLICATE", "group_id": "near-pending", "validation_status": "PROPOSED"},
    ]
    _write_csv(relation_path, ("row_uid", "relation_type", "group_id", "validation_status"), relation_rows)
    relations = load_relation_audit(relation_path)
    targets = {
        "FROZEN_TEST": {"CONTROL_REPRESENTATIVE": 1, "PROMOTIONAL_OR_SALES_LANGUAGE": 1},
        "VALIDATION": {"DIVERSITY_REPRESENTATIVE": 1, "PROMOTIONAL_OR_SALES_LANGUAGE": 1},
        "TRAIN": {"NO_USE_ASSERTIVE_CLAIM": 1, "WITHIN_REVIEW_REPETITION": 1,
                  "HARD_NEGATIVE_CANDIDATE": 1, "HIGH_CERTAINTY_EFFICACY_CLAIM": 1},
    }
    selected, decisions, report = select_candidates(rows, pilot=pilot, v2=v2, relations=relations,
                                                     targets=targets, seed=77)
    selected_ids = {item.row.row_uid for item in selected}
    assert "row-1" not in selected_ids
    assert "row-3" not in selected_ids
    assert not {"row-7", "row-8"}.issubset(selected_ids)
    assert len({item.row.product_id for item in selected}) == len(selected)
    assert any(row["reason"].startswith("PILOT_PRELIMINARY") for row in decisions)
    assert any(row["reason"].startswith("V2_") for row in decisions)
    assert report["label_policy"].startswith("NO_AUTOMATIC_LABELS")
    assert report["final_dataset_ready"] is False  # pending relation touches row-11


def test_seed_reproducibility_and_different_seed_changes_order(tmp_path: Path) -> None:
    path = tmp_path / "pool.csv"
    raw = [_pool_row(index, group="CONTROL") for index in range(1, 15)]
    _write_csv(path, POOL_FIELDS, raw)
    rows = load_pool(path)
    targets = {"FROZEN_TEST": {"CONTROL_REPRESENTATIVE": 4}, "VALIDATION": {}, "TRAIN": {}}
    first, _, _ = select_candidates(rows, targets=targets, seed=9)
    second, _, _ = select_candidates(rows, targets=targets, seed=9)
    third, _, _ = select_candidates(rows, targets=targets, seed=10)
    assert [item.row.row_uid for item in first] == [item.row.row_uid for item in second]
    assert [item.row.row_uid for item in first] != [item.row.row_uid for item in third]


def test_blind_bundle_hides_provenance_role_scores_and_preserves_text(tmp_path: Path) -> None:
    raw = [_pool_row(1, group="CONTROL", content="첫 줄🙂\n둘째 줄")]
    path = tmp_path / "pool.csv"
    _write_csv(path, POOL_FIELDS, raw)
    selected, decisions, report = select_candidates(
        load_pool(path), targets={"FROZEN_TEST": {"CONTROL_REPRESENTATIVE": 1}, "VALIDATION": {}, "TRAIN": {}}
    )
    guideline = tmp_path / "guide.md"
    guideline.write_text("synthetic guide", encoding="utf-8")
    output = tmp_path / "bundle"
    write_candidate_bundle(output, selected, decisions, report, guideline_path=guideline)
    fields, packet = _read_csv(output / "blind_review" / "reviewer_packets" / "reviewer_a.csv")
    assert fields == list(BLIND_FIELDS)
    assert packet[0]["content"] == "첫 줄🙂\n둘째 줄"
    assert not ({"sampling_group", "candidate_sources", "split_role", "model_prediction", "heuristic_score"} & set(fields))
    master_fields, master = _read_csv(output / "master_candidate_manifest.csv")
    assert "split_role" in master_fields and "candidate_sources" in master_fields
    assert master[0]["label_status"] == "UNLABELED"
    metadata = json.loads((output / "candidate_generation_metadata.json").read_text(encoding="utf-8"))
    assert metadata["human_review"]["review_mode"] == "TWO_INDEPENDENT_BLIND_REVIEWERS"
    assert metadata["human_review"]["guideline_sha256"] == file_sha256(guideline)
    contract = json.loads((output / "blind_review" / "review_contract_manifest.json").read_text(encoding="utf-8"))
    assert contract["annotation_count"] == 1
    with pytest.raises(CandidateGenerationError, match="OUTPUT_EXISTS"):
        write_candidate_bundle(output, selected, decisions, report, guideline_path=guideline)


def test_model_prediction_column_is_rejected_in_pool_contract(tmp_path: Path) -> None:
    fields = POOL_FIELDS + ("model_prediction",)
    row = dict(_pool_row(1), model_prediction="SUSPICIOUS")
    path = tmp_path / "pool.csv"
    _write_csv(path, fields, [row])
    with pytest.raises(CandidateGenerationError, match="POOL_SCHEMA_INVALID"):
        load_pool(path)


def test_dry_run_verifies_hashes_writes_nothing_and_leaves_inputs_unchanged(tmp_path: Path) -> None:
    pool = tmp_path / "pool.csv"
    pool_rows = [_pool_row(1, group="CONTROL")]
    _write_csv(pool, POOL_FIELDS, pool_rows)
    pilot = tmp_path / "pilot.csv"
    _write_csv(pilot, POOL_FIELDS + ("annotation_id",), [dict(_pool_row(2), annotation_id="pilot-2")])
    v2 = tmp_path / "v2.xlsx"
    _write_v2(v2, [_pool_row(3)])
    guideline = tmp_path / "guide.md"
    guideline.write_text("synthetic guide", encoding="utf-8")
    inputs = (pool, pilot, v2, guideline)
    before = {path: file_sha256(path) for path in inputs}
    output = tmp_path / "must-not-exist"
    report = run_generation(
        pool_path=pool, v2_path=v2, pilot_path=pilot, guideline_path=guideline,
        output_dir=output, materialize=False,
        expected_pool_sha256=before[pool], expected_v2_sha256=before[v2],
        expected_pilot_sha256=before[pilot], expected_guideline_sha256=before[guideline],
        targets={"FROZEN_TEST": {"CONTROL_REPRESENTATIVE": 1}, "VALIDATION": {}, "TRAIN": {}},
    )
    assert report["dry_run"] is True
    assert report["final_dataset_ready"] is False
    assert set(report["publication_gates"]) == {
        "F3_4_2_RELATION_ARTIFACT_REQUIRED", "PUBLISHED_PILOT_EXCLUSION_REQUIRED"
    }
    assert not output.exists()
    assert before == {path: file_sha256(path) for path in inputs}
