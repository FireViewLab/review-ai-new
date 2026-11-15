"""Build label-blind, leakage-isolated candidates for independent P_text evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_generation import (
    BLIND_FIELDS,
    POOL_FIELDS,
    CandidateGenerationError,
    IdentityExclusions,
    PoolRow,
    RelationAudit,
    _SelectionState,
    _identity_reason,
    load_pool,
    load_preliminary_pilot,
    load_relation_audit,
    load_v2_exclusions,
)
from .data_contract import PilotExclusions, exclusion_matches, file_sha256, load_pilot_exclusions


POLICY_VERSION = "ptext-v3-independent-evaluation-candidates-v1"
DEFAULT_SEED = 20261115
OUTPUT_ROOT_NAME = "p_text_v3_independent_evaluation_candidates"
EXPECTED_SHA256 = {
    "pool": "c9d9e5fd5227c5586e95324592427b19ea552c65ad06bcbb0008c798aa2af19a",
    "existing_400": "ef1c6d99808a9049eb087b804531fa84c518f6d284dd73a7b56871575a8feb99",
    "extension_194": "a802576b9d33d8ef3643e35f0368047b1e4bcb6f644d86977fde2b5c6e957c4f",
    "v2": "cd3812b26dd7c782e01ca919c0ff12e3a8bf57e81ecc218522c7e476a7490673",
    "published_pilot": "dfbed35f4ad74c1ac2bfc402402f8cadcb7be4d09020d6cd3111d14ac51548ea",
    "verified_relations": "6479658035cdd7a22b72a51b1a0ff3bb7cd3ae7a35b4eccf4f14c4f3ebf334ce",
    "guideline": "ea4d5d4566e0329bbd8129d58cfee5e82ee03b3ed5b5d7309e7d610c67e9054f",
    "guideline_contract": "2c7a1d060e8f57ca0f5e2b0a8a2168b405e845c46627eb8176c169acf5e83456",
}

STRATUM_TARGETS = {
    "REPRESENTATIVE_GENERAL": {
        "CONTROL_REPRESENTATIVE": 45,
        "DIVERSITY_REPRESENTATIVE": 15,
    },
    "ENRICHED_SIGNAL_RETRIEVAL": {
        "PRODUCT_DESCRIPTION_LIKE": 20,
        "NO_USE_ASSERTIVE_CLAIM": 15,
        "PROMOTIONAL_OR_SALES_LANGUAGE": 15,
        "HIGH_CERTAINTY_EFFICACY_CLAIM": 5,
        "WITHIN_REVIEW_REPETITION": 15,
    },
    "HARD_NEGATIVE_RETRIEVAL": {"HARD_NEGATIVE_CANDIDATE": 20},
}
MASTER_FIELDS = POOL_FIELDS + (
    "annotation_id",
    "candidate_role",
    "sampling_stratum",
    "retrieval_category",
    "retrieval_evidence",
    "candidate_status",
)
DECISION_FIELDS = ("row_uid", "decision", "reason", "annotation_id", "sampling_stratum")
ANNOTATION_TEMPLATE_FIELDS = (
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


@dataclass(frozen=True)
class IndependentCandidate:
    row: PoolRow
    stratum: str
    category: str
    annotation_id: str


def merge_identity_exclusions(*sources: IdentityExclusions) -> IdentityExclusions:
    return IdentityExclusions(
        row_uids=frozenset().union(*(source.row_uids for source in sources)),
        hashes=frozenset().union(*(source.hashes for source in sources)),
        reviews=frozenset().union(*(source.reviews for source in sources)),
        products=frozenset().union(*(source.products for source in sources)),
        clusters=frozenset().union(*(source.clusters for source in sources)),
    )


def _stable(seed: int, *parts: str) -> str:
    return sha256("\x1f".join((str(seed), POLICY_VERSION, *parts)).encode("utf-8")).hexdigest()


def _relation_groups_for_existing(existing: IdentityExclusions, relations: RelationAudit) -> frozenset[tuple[str, str]]:
    return frozenset(
        group
        for row_uid in existing.row_uids
        for group in relations.verified_by_row.get(row_uid, ())
    )


def _source_overlap(left: IdentityExclusions, right: IdentityExclusions) -> dict[str, int]:
    return {
        "row_uid": len(left.row_uids & right.row_uids),
        "normalized_content": len(left.hashes & right.hashes),
        "platform_review": len(left.reviews & right.reviews),
        "platform_product": len(left.products & right.products),
        "duplicate_cluster": len(left.clusters & right.clusters),
    }


def _audit_selected(
    selected: Sequence[IndependentCandidate],
    *,
    existing_400: IdentityExclusions,
    extension_194: IdentityExclusions,
    v2: IdentityExclusions,
    formal_pilot: PilotExclusions,
    relations: RelationAudit,
) -> dict[str, int]:
    protected_groups = _relation_groups_for_existing(
        merge_identity_exclusions(existing_400, extension_194), relations
    )
    counts = Counter()
    state = _SelectionState(relations)
    for item in selected:
        row = item.row
        counts["existing_400"] += bool(_identity_reason(row, existing_400, "AUDIT"))
        counts["extension_194"] += bool(_identity_reason(row, extension_194, "AUDIT"))
        counts["v2"] += bool(_identity_reason(row, v2, "AUDIT"))
        counts["published_pilot"] += bool(exclusion_matches(row.values, formal_pilot))
        verified = relations.verified_by_row.get(row.row_uid, ())
        counts["verified_relation"] += any(
            kind == "V2_LINEAGE" or group in protected_groups for kind, group in verified
        )
        counts["internal_identity_product_cluster_relation"] += bool(state.reason(row))
        if not state.reason(row):
            state.add(row)
    result = {
        name: counts[name]
        for name in (
            "existing_400",
            "extension_194",
            "v2",
            "published_pilot",
            "verified_relation",
            "internal_identity_product_cluster_relation",
        )
    }
    if any(result.values()):
        raise CandidateGenerationError("POST_SELECTION_LEAKAGE_DETECTED")
    return result


def select_independent_candidates(
    rows: Sequence[PoolRow],
    *,
    existing_400: IdentityExclusions,
    extension_194: IdentityExclusions,
    v2: IdentityExclusions,
    formal_pilot: PilotExclusions,
    relations: RelationAudit,
    seed: int = DEFAULT_SEED,
    targets: Mapping[str, Mapping[str, int]] = STRATUM_TARGETS,
    max_platform_share: float = 0.60,
    max_category_share: float = 0.25,
) -> tuple[list[IndependentCandidate], list[dict[str, str]], dict[str, Any]]:
    target_total = sum(sum(categories.values()) for categories in targets.values())
    if target_total <= 0 or any(value < 0 for categories in targets.values() for value in categories.values()):
        raise CandidateGenerationError("TARGETS_INVALID")
    platform_cap = max(1, math.ceil(target_total * max_platform_share))
    category_cap = max(1, math.ceil(target_total * max_category_share))
    protected_relation_groups = _relation_groups_for_existing(
        merge_identity_exclusions(existing_400, extension_194), relations
    )
    eligible: list[PoolRow] = []
    decisions: list[dict[str, str]] = []
    exclusions: Counter[str] = Counter()
    for row in rows:
        reason = _identity_reason(row, existing_400, "EXISTING_400")
        reason = reason or _identity_reason(row, extension_194, "EXTENSION_194")
        reason = reason or _identity_reason(row, v2, "V2")
        verified = relations.verified_by_row.get(row.row_uid, ())
        if not reason and exclusion_matches(row.values, formal_pilot):
            reason = "PUBLISHED_PILOT"
        if not reason and any(kind == "V2_LINEAGE" for kind, _ in verified):
            reason = "VERIFIED_V2_LINEAGE"
        if not reason and any(group in protected_relation_groups for group in verified):
            reason = "EXISTING_CANDIDATE_VERIFIED_RELATION"
        if not reason and row.row_uid in relations.pending_by_row:
            reason = "PENDING_RELATION_REVIEW_REQUIRED"
        if reason:
            exclusions[reason] += 1
            decisions.append({"row_uid": row.row_uid, "decision": "EXCLUDED", "reason": reason,
                              "annotation_id": "", "sampling_stratum": ""})
        else:
            eligible.append(row)

    state = _SelectionState(relations)
    selected: list[IndependentCandidate] = []
    platform_counts: Counter[str] = Counter()
    normalized_category_counts: Counter[str] = Counter()
    selected_by_category: Counter[str] = Counter()
    selected_by_stratum: Counter[str] = Counter()
    hard_rejections: Counter[str] = Counter()
    category_quota_shortfalls: dict[str, dict[str, int]] = {}
    stratum_shortfalls: dict[str, int] = {}
    fallback_allocations: Counter[str] = Counter()
    for stratum, categories in targets.items():
        category_shortfall = {}
        for category, target in categories.items():
            candidates = [row for row in eligible if category in row.categories and row.row_uid not in state.rows]
            candidates.sort(key=lambda row: (
                platform_counts[row.platform],
                normalized_category_counts[row.normalized_category],
                _stable(seed, stratum, category, row.row_uid),
                row.row_uid,
            ))
            chosen = 0
            for row in candidates:
                reason = state.reason(row)
                if reason:
                    hard_rejections[reason] += 1
                    continue
                if platform_counts[row.platform] >= platform_cap:
                    hard_rejections["PLATFORM_SHARE_CAP"] += 1
                    continue
                if normalized_category_counts[row.normalized_category] >= category_cap:
                    hard_rejections["NORMALIZED_CATEGORY_SHARE_CAP"] += 1
                    continue
                state.add(row)
                annotation_id = _stable(seed, "annotation", row.row_uid)[:24]
                selected.append(IndependentCandidate(row, stratum, category, annotation_id))
                platform_counts[row.platform] += 1
                normalized_category_counts[row.normalized_category] += 1
                selected_by_category[category] += 1
                selected_by_stratum[stratum] += 1
                chosen += 1
                if chosen == target:
                    break
            if chosen < target:
                category_shortfall[category] = target - chosen
        if category_shortfall:
            category_quota_shortfalls[stratum] = category_shortfall

        # A scarce retrieval proxy must not silently shrink the overall review
        # packet. Fill only from another category in the same declared stratum,
        # while retaining the original category shortfall and fallback counts.
        stratum_target = sum(categories.values())
        missing = stratum_target - selected_by_stratum[stratum]
        fallback_candidates = [
            row for row in eligible
            if row.row_uid not in state.rows and any(category in row.categories for category in categories)
        ]
        fallback_candidates.sort(key=lambda row: (
            platform_counts[row.platform],
            normalized_category_counts[row.normalized_category],
            _stable(seed, stratum, "FALLBACK", row.row_uid),
            row.row_uid,
        ))
        for row in fallback_candidates:
            if missing == 0:
                break
            reason = state.reason(row)
            if reason:
                hard_rejections[reason] += 1
                continue
            if platform_counts[row.platform] >= platform_cap:
                hard_rejections["PLATFORM_SHARE_CAP"] += 1
                continue
            if normalized_category_counts[row.normalized_category] >= category_cap:
                hard_rejections["NORMALIZED_CATEGORY_SHARE_CAP"] += 1
                continue
            available_categories = [category for category in categories if category in row.categories]
            category = min(
                available_categories,
                key=lambda name: (
                    selected_by_category[name] / max(categories[name], 1),
                    selected_by_category[name],
                    name,
                ),
            )
            state.add(row)
            annotation_id = _stable(seed, "annotation", row.row_uid)[:24]
            selected.append(IndependentCandidate(row, stratum, category, annotation_id))
            platform_counts[row.platform] += 1
            normalized_category_counts[row.normalized_category] += 1
            selected_by_category[category] += 1
            selected_by_stratum[stratum] += 1
            fallback_allocations[f"{stratum}:{category}"] += 1
            missing -= 1
        if missing:
            stratum_shortfalls[stratum] = missing

    selected_uids = {item.row.row_uid for item in selected}
    for item in selected:
        decisions.append({"row_uid": item.row.row_uid, "decision": "SELECTED", "reason": item.category,
                          "annotation_id": item.annotation_id, "sampling_stratum": item.stratum})
    already_decided = {item["row_uid"] for item in decisions}
    for row in rows:
        if row.row_uid not in already_decided:
            decisions.append({"row_uid": row.row_uid, "decision": "NOT_SELECTED",
                              "reason": "OUTSIDE_TARGET_OR_HARD_CONSTRAINT", "annotation_id": "",
                              "sampling_stratum": ""})
    if len(selected_uids) != len(selected):
        raise CandidateGenerationError("SELECTED_ROW_UID_DUPLICATE")
    report = {
        "policy_version": POLICY_VERSION,
        "seed": seed,
        "candidate_role": "INDEPENDENT_EVALUATION_CANDIDATE",
        "target_count": target_total,
        "eligible_after_exclusions": len(eligible),
        "selected_count": len(selected),
        "targets": {name: dict(values) for name, values in targets.items()},
        "selected_by_stratum": dict(sorted(selected_by_stratum.items())),
        "selected_by_category": dict(sorted(selected_by_category.items())),
        "selected_by_platform": dict(sorted(platform_counts.items())),
        "selected_by_normalized_category": dict(sorted(normalized_category_counts.items())),
        "exclusions": dict(sorted(exclusions.items())),
        "hard_constraint_rejections": dict(sorted(hard_rejections.items())),
        "category_quota_shortfalls": category_quota_shortfalls,
        "fallback_allocations_within_stratum": dict(sorted(fallback_allocations.items())),
        "shortfalls": stratum_shortfalls,
        "protected_source_overlap": {
            "existing_400_vs_extension_194": _source_overlap(existing_400, extension_194),
        },
        "cross_leakage_after_selection": _audit_selected(
            selected,
            existing_400=existing_400,
            extension_194=extension_194,
            v2=v2,
            formal_pilot=formal_pilot,
            relations=relations,
        ),
        "automatic_labels_assigned": False,
        "model_predictions_used": False,
        "threshold_selection_used": False,
        "official_human_ground_truth": False,
        "eligible_for_training": False,
        "eligible_for_threshold_selection": False,
        "dataset_publication_ready": False,
        "blind_packet_ready": not stratum_shortfalls and len(selected) == target_total,
        "sampling_interpretation": "enriched and hard-negative strata are retrieval samples, not natural prevalence",
    }
    return selected, decisions, report


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _assert_private_output(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    private_root = (repo_root / "artifacts" / OUTPUT_ROOT_NAME).resolve()
    resolved = path.resolve()
    if resolved == private_root or private_root not in resolved.parents:
        raise CandidateGenerationError("OUTPUT_NOT_IN_PRIVATE_ROOT")


def validate_guideline_contract(guideline: Path, contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "OFFICIAL_FOR_HUMAN_REVIEW":
        raise CandidateGenerationError("GUIDELINE_CONTRACT_NOT_OFFICIAL")
    if contract.get("guideline_version") != "ptext-labeling-guide-v0.3":
        raise CandidateGenerationError("GUIDELINE_VERSION_INVALID")
    if contract.get("guideline_sha256") != file_sha256(guideline):
        raise CandidateGenerationError("GUIDELINE_CONTRACT_SHA256_MISMATCH")
    if contract.get("packet_fields") != list(BLIND_FIELDS):
        raise CandidateGenerationError("GUIDELINE_BLIND_FIELDS_INVALID")
    return contract


def write_bundle(
    output_dir: Path,
    selected: Sequence[IndependentCandidate],
    decisions: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
    *,
    guideline: Path,
    guideline_contract: Path,
    seed: int,
) -> None:
    if output_dir.exists():
        raise CandidateGenerationError("OUTPUT_EXISTS")
    _assert_private_output(output_dir)
    contract = validate_guideline_contract(guideline, guideline_contract)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "master_candidate_manifest.csv", MASTER_FIELDS, (
        {field: item.row.values[field] for field in POOL_FIELDS}
        | {
            "annotation_id": item.annotation_id,
            "candidate_role": "INDEPENDENT_EVALUATION_CANDIDATE",
            "sampling_stratum": item.stratum,
            "retrieval_category": item.category,
            "retrieval_evidence": "|".join(item.row.evidence),
            "candidate_status": "UNLABELED_HUMAN_REVIEW_REQUIRED",
        }
        for item in selected
    ))
    _write_csv(output_dir / "selection_exclusion_manifest.csv", DECISION_FIELDS, decisions)
    for slot in ("reviewer_a", "reviewer_b"):
        ordered = sorted(selected, key=lambda item: _stable(seed, slot, item.annotation_id))
        _write_csv(output_dir / "blind_review" / f"{slot}_packet.csv", BLIND_FIELDS, (
            {"annotation_id": item.annotation_id, "display_order": index, "content": item.row.content}
            for index, item in enumerate(ordered, 1)
        ))
        _write_csv(output_dir / "blind_review" / f"{slot}_annotation_template.csv", ANNOTATION_TEMPLATE_FIELDS, (
            {
                "annotation_id": item.annotation_id,
                "annotator_id": "",
                "text_status": "",
                "signal_decision": "",
                "signal_types": "",
                "other_review_needed": "",
                "rationale_codes": "",
                "rationale_note": "",
                "guideline_version": "ptext-labeling-guide-v0.3",
                "annotation_round": "1",
                "annotated_at": "",
            }
            for item in ordered
        ))
    review_contract = {
        "contract_version": "ptext-independent-evaluation-review-contract-v1",
        "guideline_version": contract["guideline_version"],
        "guideline_sha256": contract["guideline_sha256"],
        "review_mode": "TWO_INDEPENDENT_BLIND_REVIEWERS_PENDING_ASSIGNMENT",
        "reviewer_slots": ["A", "B"],
        "reviewer_identity_status": "UNASSIGNED",
        "packet_fields": list(BLIND_FIELDS),
        "annotation_count": len(selected),
        "labels_present": False,
        "model_predictions_present": False,
        "timestamps_present": False,
        "dataset_publication_ready": False,
    }
    _write_json(output_dir / "blind_review" / "review_contract_manifest.json", review_contract)
    public_metadata = dict(metadata)
    public_metadata["human_review"] = review_contract
    _write_json(output_dir / "candidate_generation_metadata.json", public_metadata)


def run_generation(
    *,
    pool_path: Path,
    existing_400_path: Path,
    extension_194_path: Path,
    v2_path: Path,
    published_pilot_path: Path,
    verified_relations_path: Path,
    guideline_path: Path,
    guideline_contract_path: Path,
    output_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    materialize: bool = False,
    expected_sha256: Mapping[str, str] = EXPECTED_SHA256,
    targets: Mapping[str, Mapping[str, int]] = STRATUM_TARGETS,
) -> dict[str, Any]:
    inputs = {
        "pool": pool_path,
        "existing_400": existing_400_path,
        "extension_194": extension_194_path,
        "v2": v2_path,
        "published_pilot": published_pilot_path,
        "verified_relations": verified_relations_path,
        "guideline": guideline_path,
        "guideline_contract": guideline_contract_path,
    }
    before = {name: file_sha256(path) for name, path in inputs.items()}
    for name, expected in expected_sha256.items():
        if expected and before[name].lower() != expected.lower():
            raise CandidateGenerationError(name.upper() + "_SHA256_MISMATCH")
    validate_guideline_contract(guideline_path, guideline_contract_path)
    rows = load_pool(pool_path)
    existing_400 = load_preliminary_pilot(existing_400_path)
    extension_194 = load_preliminary_pilot(extension_194_path)
    if len(existing_400.row_uids) != 400:
        raise CandidateGenerationError("EXISTING_400_COUNT_INVALID")
    if len(extension_194.row_uids) != 194:
        raise CandidateGenerationError("EXTENSION_194_COUNT_INVALID")
    v2 = load_v2_exclusions(v2_path)
    formal_pilot = load_pilot_exclusions(published_pilot_path)
    relations = load_relation_audit(verified_relations_path)
    selected, decisions, report = select_independent_candidates(
        rows,
        existing_400=existing_400,
        extension_194=extension_194,
        v2=v2,
        formal_pilot=formal_pilot,
        relations=relations,
        seed=seed,
        targets=targets,
    )
    report.update({
        "dry_run": not materialize,
        "input_sha256": before,
        "input_rows": {"pool": len(rows), "existing_400": 400, "extension_194": 194},
        "relation_status_counts": dict(relations.status_counts or {}),
        "output_materialized": materialize,
    })
    after = {name: file_sha256(path) for name, path in inputs.items()}
    if before != after:
        raise CandidateGenerationError("INPUT_MUTATED_DURING_RUN")
    if materialize:
        if output_dir is None:
            raise CandidateGenerationError("OUTPUT_REQUIRED")
        write_bundle(output_dir, selected, decisions, report, guideline=guideline_path,
                     guideline_contract=guideline_contract_path, seed=seed)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--existing-400", type=Path, required=True)
    parser.add_argument("--extension-194", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--published-pilot-exclusion", type=Path, required=True)
    parser.add_argument("--verified-relations", type=Path, required=True)
    parser.add_argument("--guideline", type=Path, required=True)
    parser.add_argument("--guideline-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    report = run_generation(
        pool_path=args.pool,
        existing_400_path=args.existing_400,
        extension_194_path=args.extension_194,
        v2_path=args.v2,
        published_pilot_path=args.published_pilot_exclusion,
        verified_relations_path=args.verified_relations,
        guideline_path=args.guideline,
        guideline_contract_path=args.guideline_contract,
        output_dir=args.output,
        seed=args.seed,
        materialize=args.materialize,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
