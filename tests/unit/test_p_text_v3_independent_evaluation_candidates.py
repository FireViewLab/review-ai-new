from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import app.training.p_text_v3.independent_evaluation_candidates as independent
from app.training.p_text_v3.candidate_generation import (
    BLIND_FIELDS,
    CandidateGenerationError,
    IdentityExclusions,
    PoolRow,
    RelationAudit,
    classify_candidate,
    pool_content_hash,
)
from app.training.p_text_v3.data_contract import PilotExclusions, file_sha256


def _row(index: int, *, group="CONTROL", product=None, content=None, sources="", cluster="") -> PoolRow:
    text = content or f"synthetic review {index}"
    values = {
        "pool_id": f"pool-{index}", "row_uid": f"row-{index}", "sampling_group": group,
        "sampling_reason": "synthetic", "label_status": "UNLABELED", "platform": "P",
        "source_row": str(index), "product_id": product or f"product-{index}", "review_id": f"review-{index}",
        "product_category_raw": "cat", "normalized_category": f"cat-{index % 3}", "content": text,
        "normalized_content_hash": pool_content_hash(text), "content_length": str(len(text)),
        "duplicate_cluster_id": cluster, "candidate_sources": sources, "original_candidate_member": "0",
        "original_control_eligible": "1", "seed": "1", "policy_version": "synthetic",
    }
    categories, evidence = classify_candidate(values)
    return PoolRow(values, categories, evidence)


def _identity(row: PoolRow) -> IdentityExclusions:
    return IdentityExclusions(
        row_uids=frozenset({row.row_uid}), hashes=frozenset({row.normalized_content_hash}),
        reviews=frozenset({(row.platform, row.review_id)}), products=frozenset({(row.platform, row.product_id)}),
        clusters=frozenset({row.duplicate_cluster_id}) if row.duplicate_cluster_id else frozenset(),
    )


def _empty():
    return IdentityExclusions(), PilotExclusions({}, "", 0)


def test_integrated_exclusions_cover_400_extension_v2_and_verified_relation():
    old400, extension, v2 = _row(1), _row(2), _row(3)
    related = _row(4)
    control = [_row(index) for index in range(5, 10)]
    relations = RelationAudit(
        {old400.row_uid: (("NEAR_DUPLICATE", "g"),), related.row_uid: (("NEAR_DUPLICATE", "g"),)}, {}, None, {}
    )
    selected, decisions, report = independent.select_independent_candidates(
        [old400, extension, v2, related, *control],
        existing_400=_identity(old400), extension_194=_identity(extension), v2=_identity(v2),
        formal_pilot=PilotExclusions({}, "", 0), relations=relations,
        targets={"REPRESENTATIVE_GENERAL": {"CONTROL_REPRESENTATIVE": 2}},
    )
    selected_uids = {item.row.row_uid for item in selected}
    assert not {old400.row_uid, extension.row_uid, v2.row_uid, related.row_uid} & selected_uids
    reasons = {row["reason"] for row in decisions if row["decision"] == "EXCLUDED"}
    assert any(reason.startswith("EXISTING_400") for reason in reasons)
    assert any(reason.startswith("EXTENSION_194") for reason in reasons)
    assert any(reason.startswith("V2") for reason in reasons)
    assert "EXISTING_CANDIDATE_VERIFIED_RELATION" in reasons
    assert not any(report["cross_leakage_after_selection"].values())


def test_selection_blocks_same_product_content_cluster_and_relation():
    rows = [
        _row(1, product="same"), _row(2, product="same"),
        _row(3, content="duplicate text"), _row(4, content="duplicate text"),
        _row(5, cluster="c"), _row(6, cluster="c"), _row(7), _row(8),
    ]
    relations = RelationAudit(
        {"row-7": (("NEAR_DUPLICATE", "r"),), "row-8": (("NEAR_DUPLICATE", "r"),)}, {}, None, {}
    )
    empty, pilot = _empty()
    selected, _, report = independent.select_independent_candidates(
        rows, existing_400=empty, extension_194=empty, v2=empty, formal_pilot=pilot, relations=relations,
        targets={"REPRESENTATIVE_GENERAL": {"CONTROL_REPRESENTATIVE": 4}},
    )
    assert len({(item.row.platform, item.row.product_id) for item in selected}) == len(selected)
    assert len({item.row.normalized_content_hash for item in selected}) == len(selected)
    assert len({item.row.duplicate_cluster_id for item in selected if item.row.duplicate_cluster_id}) <= 1
    assert not any(report["cross_leakage_after_selection"].values())


def test_blind_packets_have_only_public_fields_and_no_fake_review_evidence(tmp_path: Path, monkeypatch):
    rows = [_row(index) for index in range(1, 8)]
    empty, pilot = _empty()
    selected, decisions, report = independent.select_independent_candidates(
        rows, existing_400=empty, extension_194=empty, v2=empty, formal_pilot=pilot,
        relations=RelationAudit({}, {}), targets={"REPRESENTATIVE_GENERAL": {"CONTROL_REPRESENTATIVE": 6}},
    )
    guide = tmp_path / "guide.md"
    guide.write_text("guide", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "status": "OFFICIAL_FOR_HUMAN_REVIEW", "guideline_version": "ptext-labeling-guide-v0.3",
        "guideline_sha256": file_sha256(guide), "packet_fields": list(BLIND_FIELDS),
    }), encoding="utf-8")
    monkeypatch.setattr(independent, "_assert_private_output", lambda path: None)
    output = tmp_path / "bundle"
    independent.write_bundle(output, selected, decisions, report, guideline=guide, guideline_contract=contract, seed=9)
    packets = []
    for slot in ("reviewer_a", "reviewer_b"):
        with (output / "blind_review" / f"{slot}_packet.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == BLIND_FIELDS
            packets.append(list(reader))
        with (output / "blind_review" / f"{slot}_annotation_template.csv").open(encoding="utf-8-sig", newline="") as handle:
            template = list(csv.DictReader(handle))
            assert all(not row["annotator_id"] and not row["annotated_at"] and not row["signal_decision"] for row in template)
    assert {row["annotation_id"] for row in packets[0]} == {row["annotation_id"] for row in packets[1]}
    assert [row["annotation_id"] for row in packets[0]] != [row["annotation_id"] for row in packets[1]]
    review_contract = json.loads((output / "blind_review" / "review_contract_manifest.json").read_text(encoding="utf-8"))
    assert review_contract["reviewer_identity_status"] == "UNASSIGNED"
    assert review_contract["labels_present"] is False
    with pytest.raises(CandidateGenerationError, match="OUTPUT_EXISTS"):
        independent.write_bundle(output, selected, decisions, report, guideline=guide, guideline_contract=contract, seed=9)


def test_merge_exclusions_preserves_overlap_without_dropping_entities():
    one, two = _row(1), _row(2)
    merged = independent.merge_identity_exclusions(_identity(one), _identity(two), _identity(one))
    assert merged.row_uids == {"row-1", "row-2"}
    assert len(merged.products) == 2


def test_guideline_contract_sha_mismatch_is_fail_closed(tmp_path: Path):
    guide = tmp_path / "guide.md"
    guide.write_text("guide", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "status": "OFFICIAL_FOR_HUMAN_REVIEW", "guideline_version": "ptext-labeling-guide-v0.3",
        "guideline_sha256": "0" * 64, "packet_fields": list(BLIND_FIELDS),
    }), encoding="utf-8")
    with pytest.raises(CandidateGenerationError, match="GUIDELINE_CONTRACT_SHA256_MISMATCH"):
        independent.validate_guideline_contract(guide, contract)


def test_same_stratum_fallback_fills_packet_and_preserves_category_shortfall():
    rows = [_row(index, sources="PRODUCT_DESCRIPTION_REPLACEMENT") for index in range(1, 8)]
    empty, pilot = _empty()
    selected, _, report = independent.select_independent_candidates(
        rows, existing_400=empty, extension_194=empty, v2=empty, formal_pilot=pilot,
        relations=RelationAudit({}, {}),
        targets={"ENRICHED_SIGNAL_RETRIEVAL": {"PRODUCT_DESCRIPTION_LIKE": 2, "PROMOTIONAL_OR_SALES_LANGUAGE": 3}},
        max_platform_share=1.0, max_category_share=1.0,
    )
    assert len(selected) == 5
    assert report["category_quota_shortfalls"]["ENRICHED_SIGNAL_RETRIEVAL"]["PROMOTIONAL_OR_SALES_LANGUAGE"] == 3
    assert report["fallback_allocations_within_stratum"]["ENRICHED_SIGNAL_RETRIEVAL:PRODUCT_DESCRIPTION_LIKE"] == 3
    assert report["shortfalls"] == {}


def test_verified_v2_lineage_is_excluded_directly():
    linked, safe = _row(1), _row(2)
    empty, pilot = _empty()
    selected, decisions, _ = independent.select_independent_candidates(
        [linked, safe], existing_400=empty, extension_194=empty, v2=empty, formal_pilot=pilot,
        relations=RelationAudit({linked.row_uid: (("V2_LINEAGE", "v2-a"),)}, {}),
        targets={"REPRESENTATIVE_GENERAL": {"CONTROL_REPRESENTATIVE": 1}},
    )
    assert [item.row.row_uid for item in selected] == [safe.row_uid]
    assert any(row["reason"] == "VERIFIED_V2_LINEAGE" for row in decisions)


def test_run_generation_rejects_input_sha_change_before_parsing(tmp_path: Path):
    paths = {}
    for name in ("pool", "existing_400", "extension_194", "v2", "published_pilot", "verified_relations", "guideline", "guideline_contract"):
        paths[name] = tmp_path / name
        paths[name].write_text("synthetic", encoding="utf-8")
    with pytest.raises(CandidateGenerationError, match="POOL_SHA256_MISMATCH"):
        independent.run_generation(
            pool_path=paths["pool"], existing_400_path=paths["existing_400"],
            extension_194_path=paths["extension_194"], v2_path=paths["v2"],
            published_pilot_path=paths["published_pilot"], verified_relations_path=paths["verified_relations"],
            guideline_path=paths["guideline"], guideline_contract_path=paths["guideline_contract"],
            expected_sha256={"pool": "0" * 64},
        )
