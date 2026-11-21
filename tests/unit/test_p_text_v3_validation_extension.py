from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.p_text_v3.candidate_generation import (
    BLIND_FIELDS,
    CandidateGenerationError,
    IdentityExclusions,
    PoolRow,
    RelationAudit,
    pool_content_hash,
)
from app.training.p_text_v3.data_contract import PilotExclusions, pilot_entity_hash
from app.training.p_text_v3.validation_extension import (
    DEFAULT_TARGETS,
    ROLE,
    scaled_targets,
    select_extension_candidates,
    write_extension_bundle,
)


def pool_row(index: int, category: str, *, product: str | None = None, cluster: str = "") -> PoolRow:
    content = f"synthetic private review {index}"
    values = {
        "pool_id": f"pool-{index}",
        "row_uid": f"row-{index}",
        "sampling_group": "CONTROL" if category == "CONTROL_REPRESENTATIVE" else "DIVERSITY" if category == "DIVERSITY_REPRESENTATIVE" else "CANDIDATE",
        "sampling_reason": "synthetic",
        "label_status": "UNLABELED",
        "platform": "P1" if index % 2 else "P2",
        "source_row": str(index),
        "product_id": product or f"product-{index}",
        "review_id": f"review-{index}",
        "product_category_raw": "fixture",
        "normalized_category": f"cat-{index % 8}",
        "content": content,
        "normalized_content_hash": pool_content_hash(content),
        "content_length": str(len(content)),
        "duplicate_cluster_id": cluster,
        "candidate_sources": "",
        "original_candidate_member": "0",
        "original_control_eligible": "0",
        "seed": "1",
        "policy_version": "fixture",
    }
    return PoolRow(values, (category,), (f"FIXTURE:{category}",))


def candidate_pool(extra: int = 4) -> list[PoolRow]:
    rows = []
    index = 0
    for category, target in DEFAULT_TARGETS.items():
        for _ in range(target + extra):
            rows.append(pool_row(index, category))
            index += 1
    return rows


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def test_selects_diverse_unlabeled_extension_without_changing_existing_roles():
    rows = candidate_pool()
    existing = IdentityExclusions(
        row_uids=frozenset({rows[0].row_uid}),
        hashes=frozenset({rows[1].normalized_content_hash}),
        products=frozenset({(rows[2].platform, rows[2].product_id)}),
        clusters=frozenset(),
    )
    before = [dict(row.values) for row in rows]
    selected, _, report = select_extension_candidates(rows, existing=existing)
    assert len(selected) == 200
    assert report["candidate_role"] == ROLE
    assert report["automatic_labels_assigned"] is False
    assert report["eligible_for_training_before_human_label"] is False
    assert report["selected_by_category"] == DEFAULT_TARGETS
    assert set(report["selected_by_band"]) == {"ENRICHED_RETRIEVAL", "CONTRAST_CONTROL"}
    assert all(item.row.label_status == "UNLABELED" for item in selected)
    assert all("label" not in item.row.values for item in selected)
    assert len({(item.row.platform, item.row.product_id) for item in selected}) == len(selected)
    assert before == [dict(row.values) for row in rows]
    assert "synthetic private review" not in json.dumps(report)


def test_pilot_and_verified_relation_to_existing_400_are_excluded():
    rows = candidate_pool()
    pilot_row = rows[10]
    pilot_hash = pilot_entity_hash("ROW_UID", pilot_row.row_uid)
    pilot = PilotExclusions({"ROW_UID": frozenset({pilot_hash})}, "fixture", 1)
    existing_uid = "existing-row"
    relation_row = rows[11]
    relations = RelationAudit(
        verified_by_row={
            existing_uid: (("NEAR_DUPLICATE", "shared"),),
            relation_row.row_uid: (("NEAR_DUPLICATE", "shared"),),
        },
        pending_by_row={},
    )
    existing = IdentityExclusions(row_uids=frozenset({existing_uid}))
    selected, decisions, report = select_extension_candidates(
        rows, existing=existing, formal_pilot=pilot, relations=relations
    )
    selected_uids = {item.row.row_uid for item in selected}
    assert pilot_row.row_uid not in selected_uids
    assert relation_row.row_uid not in selected_uids
    assert report["exclusions"]["PILOT_PUBLISHED_MANIFEST"] == 1
    assert report["exclusions"]["EXISTING_400_VERIFIED_RELATION"] == 1
    assert all("content" not in decision for decision in decisions)


def test_selected_rows_cannot_share_product_hash_cluster_or_verified_relation():
    rows = candidate_pool()
    rows.append(pool_row(1000, "PRODUCT_DESCRIPTION_LIKE", product=rows[0].product_id))
    rows.append(pool_row(1001, "PRODUCT_DESCRIPTION_LIKE", cluster="shared-cluster"))
    rows.append(pool_row(1002, "PRODUCT_DESCRIPTION_LIKE", cluster="shared-cluster"))
    relations = RelationAudit(
        verified_by_row={
            rows[-2].row_uid: (("NEAR_DUPLICATE", "near-shared"),),
            rows[-1].row_uid: (("NEAR_DUPLICATE", "near-shared"),),
        },
        pending_by_row={},
    )
    selected, _, _ = select_extension_candidates(
        rows, existing=IdentityExclusions(), relations=relations
    )
    assert len({(item.row.platform, item.row.product_id) for item in selected}) == len(selected)
    clusters = [item.row.duplicate_cluster_id for item in selected if item.row.duplicate_cluster_id]
    assert len(clusters) == len(set(clusters))
    selected_verified = [
        group
        for item in selected
        for group in relations.verified_by_row.get(item.row.row_uid, ())
    ]
    assert len(selected_verified) == len(set(selected_verified))


def test_bundle_is_blind_non_training_and_cannot_overwrite(tmp_path, monkeypatch):
    selected, decisions, report = select_extension_candidates(
        candidate_pool(), existing=IdentityExclusions()
    )
    report["pending_relation_row_uids"] = ()
    output = tmp_path / "artifacts" / "p_text_v3_validation_extensions" / "validation_extension_v1_fixture"
    import app.training.p_text_v3.validation_extension as module
    monkeypatch.setattr(module, "_assert_private_output", lambda path: None)
    write_extension_bundle(output, selected, decisions, report, seed=7)
    fields, packet = read_csv(output / "blind_review_packet.csv")
    assert fields == BLIND_FIELDS
    assert not ({"selection_category", "candidate_role", "retrieval_evidence"} & set(fields))
    _, roles = read_csv(output / "candidate_role_manifest.csv")
    assert all(row["candidate_role"] == ROLE for row in roles)
    assert all(row["eligible_for_training"] == "NO" for row in roles)
    assert all(row["eligible_for_threshold_selection"] == "NO" for row in roles)
    with pytest.raises(CandidateGenerationError, match="OUTPUT_EXISTS"):
        write_extension_bundle(output, selected, decisions, report, seed=7)


@pytest.mark.parametrize("target", [149, 251, True])
def test_target_size_is_bounded(target):
    with pytest.raises(CandidateGenerationError, match="TARGET_SIZE_OUT_OF_RANGE"):
        scaled_targets(target)


def test_scaled_targets_are_deterministic_and_sum_to_requested_size():
    assert sum(scaled_targets(150).values()) == 150
    assert sum(scaled_targets(225).values()) == 225
    assert scaled_targets(225) == scaled_targets(225)
