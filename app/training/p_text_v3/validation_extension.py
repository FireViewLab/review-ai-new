"""Label-blind candidate generation for ``validation_extension_v1``.

The extension never modifies or resplits the original 400 candidates.  Its
rows remain UNLABELED review candidates until a separate human-review and
dataset-publication step approves them.
"""

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
    ANNOTATION_FIELDS,
    BLIND_FIELDS,
    MASTER_FIELDS,
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


POLICY_VERSION = "ptext-v3-validation-extension-v1"
DEFAULT_SEED = 20261114
EXPECTED_POOL_SHA256 = "c9d9e5fd5227c5586e95324592427b19ea552c65ad06bcbb0008c798aa2af19a"
EXPECTED_EXISTING_400_MASTER_SHA256 = "ef1c6d99808a9049eb087b804531fa84c518f6d284dd73a7b56871575a8feb99"
EXPECTED_V2_SHA256 = "cd3812b26dd7c782e01ca919c0ff12e3a8bf57e81ecc218522c7e476a7490673"
EXPECTED_PUBLISHED_PILOT_SHA256 = "dfbed35f4ad74c1ac2bfc402402f8cadcb7be4d09020d6cd3111d14ac51548ea"
EXPECTED_VERIFIED_RELATIONS_SHA256 = "6479658035cdd7a22b72a51b1a0ff3bb7cd3ae7a35b4eccf4f14c4f3ebf334ce"
ROLE = "VALIDATION_EXTENSION_CANDIDATE"
OUTPUT_ROOT_NAME = "p_text_v3_validation_extensions"
DEFAULT_TARGETS = {
    # Select the single scarce direct proxy before overlapping categories.
    "HIGH_CERTAINTY_EFFICACY_CLAIM": 1,
    "PRODUCT_DESCRIPTION_LIKE": 40,
    "NO_USE_ASSERTIVE_CLAIM": 35,
    "PROMOTIONAL_OR_SALES_LANGUAGE": 35,
    "WITHIN_REVIEW_REPETITION": 35,
    "HARD_NEGATIVE_CANDIDATE": 25,
    "CONTROL_REPRESENTATIVE": 20,
    "DIVERSITY_REPRESENTATIVE": 9,
}
ENRICHED_CATEGORIES = frozenset(
    {
        "PRODUCT_DESCRIPTION_LIKE",
        "NO_USE_ASSERTIVE_CLAIM",
        "PROMOTIONAL_OR_SALES_LANGUAGE",
        "HIGH_CERTAINTY_EFFICACY_CLAIM",
        "WITHIN_REVIEW_REPETITION",
    }
)
CONTRAST_CATEGORIES = frozenset(
    {"HARD_NEGATIVE_CANDIDATE", "CONTROL_REPRESENTATIVE", "DIVERSITY_REPRESENTATIVE"}
)
EXTENSION_MASTER_FIELDS = MASTER_FIELDS + ("candidate_band", "extension_status")
ROLE_FIELDS = ("annotation_id", "candidate_role", "eligible_for_training", "eligible_for_threshold_selection")
DECISION_FIELDS = ("row_uid", "decision", "reason", "annotation_id")


@dataclass(frozen=True)
class ExtensionCandidate:
    row: PoolRow
    category: str
    annotation_id: str

    @property
    def band(self) -> str:
        return "ENRICHED_RETRIEVAL" if self.category in ENRICHED_CATEGORIES else "CONTRAST_CONTROL"


def _stable_key(seed: int, *parts: str) -> str:
    return sha256("\x1f".join((str(seed), POLICY_VERSION, *parts)).encode("utf-8")).hexdigest()


def scaled_targets(target_size: int) -> dict[str, int]:
    if type(target_size) is not int or not 150 <= target_size <= 250:
        raise CandidateGenerationError("TARGET_SIZE_OUT_OF_RANGE")
    total = sum(DEFAULT_TARGETS.values())
    raw = {name: value * target_size / total for name, value in DEFAULT_TARGETS.items()}
    result = {name: math.floor(value) for name, value in raw.items()}
    remainder = target_size - sum(result.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - result[name]), name))
    for name in order[:remainder]:
        result[name] += 1
    return result


def _existing_verified_groups(
    existing: IdentityExclusions, relations: RelationAudit
) -> frozenset[tuple[str, str]]:
    groups: set[tuple[str, str]] = set()
    for row_uid in existing.row_uids:
        groups.update(relations.verified_by_row.get(row_uid, ()))
    return frozenset(groups)


def select_extension_candidates(
    rows: Sequence[PoolRow],
    *,
    existing: IdentityExclusions,
    v2: IdentityExclusions = IdentityExclusions(),
    formal_pilot: PilotExclusions = PilotExclusions({}, "", 0),
    relations: RelationAudit = RelationAudit({}, {}),
    seed: int = DEFAULT_SEED,
    targets: Mapping[str, int] = DEFAULT_TARGETS,
    max_platform_share: float = 0.60,
    max_category_share: float = 0.25,
) -> tuple[list[ExtensionCandidate], list[dict[str, str]], dict[str, Any]]:
    """Select review candidates without assigning labels or dataset splits."""
    if not targets or any(name not in ENRICHED_CATEGORIES | CONTRAST_CATEGORIES or value < 0 for name, value in targets.items()):
        raise CandidateGenerationError("TARGETS_INVALID")
    total_target = sum(targets.values())
    if not 150 <= total_target <= 250:
        raise CandidateGenerationError("TARGET_TOTAL_OUT_OF_RANGE")
    if not 0 < max_platform_share <= 1 or not 0 < max_category_share <= 1:
        raise CandidateGenerationError("DIVERSITY_CAP_INVALID")
    platform_cap = max(1, math.ceil(total_target * max_platform_share))
    category_cap = max(1, math.ceil(total_target * max_category_share))
    existing_verified = _existing_verified_groups(existing, relations)
    eligible: list[PoolRow] = []
    decisions: list[dict[str, str]] = []
    exclusions: Counter[str] = Counter()
    for row in rows:
        reason = _identity_reason(row, existing, "EXISTING_400")
        reason = reason or _identity_reason(row, v2, "V2")
        if not reason and exclusion_matches(row.values, formal_pilot):
            reason = "PILOT_PUBLISHED_MANIFEST"
        if not reason and any(group in existing_verified for group in relations.verified_by_row.get(row.row_uid, ())):
            reason = "EXISTING_400_VERIFIED_RELATION"
        if reason:
            exclusions[reason] += 1
            decisions.append({"row_uid": row.row_uid, "decision": "EXCLUDED", "reason": reason, "annotation_id": ""})
        else:
            eligible.append(row)

    state = _SelectionState(relations)
    selected: list[ExtensionCandidate] = []
    platform_counts: Counter[str] = Counter()
    normalized_category_counts: Counter[str] = Counter()
    selected_by_category: Counter[str] = Counter()
    hard_rejections: Counter[str] = Counter()
    shortfalls: dict[str, int] = {}
    for category, target in targets.items():
        candidates = [row for row in eligible if category in row.categories and row.row_uid not in state.rows]
        candidates.sort(
            key=lambda row: (
                platform_counts[row.platform],
                normalized_category_counts[row.normalized_category],
                _stable_key(seed, category, row.row_uid),
                row.row_uid,
            )
        )
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
            annotation_id = _stable_key(seed, "annotation", row.row_uid)[:24]
            selected.append(ExtensionCandidate(row, category, annotation_id))
            platform_counts[row.platform] += 1
            normalized_category_counts[row.normalized_category] += 1
            selected_by_category[category] += 1
            chosen += 1
            if chosen == target:
                break
        if chosen < target:
            shortfalls[category] = target - chosen

    for item in selected:
        decisions.append(
            {"row_uid": item.row.row_uid, "decision": "SELECTED", "reason": item.category, "annotation_id": item.annotation_id}
        )
    selected_uids = {item.row.row_uid for item in selected}
    for row in eligible:
        if row.row_uid not in selected_uids:
            decisions.append({"row_uid": row.row_uid, "decision": "NOT_SELECTED", "reason": "OUTSIDE_TARGET", "annotation_id": ""})
    pending_selected = sum(item.row.row_uid in relations.pending_by_row for item in selected)
    report = {
        "policy_version": POLICY_VERSION,
        "seed": seed,
        "candidate_role": ROLE,
        "existing_400_immutable": True,
        "existing_splits_modified": False,
        "automatic_labels_assigned": False,
        "model_predictions_used": False,
        "eligible_for_training_before_human_label": False,
        "eligible_for_threshold_selection_before_human_label": False,
        "input_pool_rows": len(rows),
        "eligible_after_exclusions": len(eligible),
        "selected_count": len(selected),
        "targets": dict(targets),
        "selected_by_category": dict(sorted(selected_by_category.items())),
        "selected_by_band": dict(sorted(Counter(item.band for item in selected).items())),
        "selected_by_platform": dict(sorted(platform_counts.items())),
        "selected_by_normalized_category": dict(sorted(normalized_category_counts.items())),
        "exclusions": dict(sorted(exclusions.items())),
        "hard_constraint_rejections": dict(sorted(hard_rejections.items())),
        "shortfalls": shortfalls,
        "pending_relation_selected_count": pending_selected,
        "labeling_packet_ready": not shortfalls,
        "dataset_publication_ready": False,
        "known_limitations": [
            "retrieval categories are not SUSPICIOUS labels",
            "enrichment changes prevalence and cannot estimate natural-population prevalence",
            "pending near-duplicate relations require review before dataset publication",
        ],
    }
    return selected, decisions, report


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _assert_private_output(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    private_root = (repo_root / "artifacts" / OUTPUT_ROOT_NAME).resolve()
    resolved = path.resolve()
    if resolved == private_root or private_root not in resolved.parents:
        raise CandidateGenerationError("OUTPUT_NOT_IN_PRIVATE_EXTENSION_ROOT")


def write_extension_bundle(
    output_dir: Path,
    selected: Sequence[ExtensionCandidate],
    decisions: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
    *,
    seed: int,
) -> None:
    if output_dir.exists():
        raise CandidateGenerationError("OUTPUT_EXISTS")
    _assert_private_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(
        output_dir / "master_validation_extension_v1.csv",
        EXTENSION_MASTER_FIELDS,
        (
            {field: item.row.values[field] for field in POOL_FIELDS}
            | {
                "annotation_id": item.annotation_id,
                "split_role": ROLE,
                "selection_category": item.category,
                "retrieval_evidence": "|".join(item.row.evidence),
                "near_duplicate_review_required": "YES" if item.row.row_uid in metadata.get("pending_relation_row_uids", ()) else "NO",
                "candidate_band": item.band,
                "extension_status": "UNLABELED_HUMAN_REVIEW_REQUIRED",
            }
            for item in selected
        ),
    )
    _write_csv(
        output_dir / "candidate_role_manifest.csv",
        ROLE_FIELDS,
        (
            {
                "annotation_id": item.annotation_id,
                "candidate_role": ROLE,
                "eligible_for_training": "NO",
                "eligible_for_threshold_selection": "NO",
            }
            for item in selected
        ),
    )
    _write_csv(output_dir / "selection_exclusion_manifest.csv", DECISION_FIELDS, decisions)
    ordered = sorted(selected, key=lambda item: _stable_key(seed, "blind", item.annotation_id))
    _write_csv(
        output_dir / "blind_review_packet.csv",
        BLIND_FIELDS,
        (
            {"annotation_id": item.annotation_id, "display_order": index, "content": item.row.content}
            for index, item in enumerate(ordered, 1)
        ),
    )
    _write_csv(
        output_dir / "annotation_template.csv",
        ANNOTATION_FIELDS,
        (
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
        ),
    )
    public_metadata = dict(metadata)
    public_metadata.pop("pending_relation_row_uids", None)
    _write_json(output_dir / "candidate_generation_metadata.json", public_metadata)


def run_generation(
    *,
    pool_path: Path,
    existing_master_path: Path,
    v2_path: Path,
    published_pilot_manifest: Path,
    verified_relations_path: Path,
    output_dir: Path | None = None,
    target_size: int = 200,
    seed: int = DEFAULT_SEED,
    materialize: bool = False,
    expected_pool_sha256: str = EXPECTED_POOL_SHA256,
    expected_existing_master_sha256: str = EXPECTED_EXISTING_400_MASTER_SHA256,
    expected_v2_sha256: str = EXPECTED_V2_SHA256,
    expected_published_pilot_sha256: str = EXPECTED_PUBLISHED_PILOT_SHA256,
    expected_verified_relations_sha256: str = EXPECTED_VERIFIED_RELATIONS_SHA256,
) -> dict[str, Any]:
    inputs = {
        "pool": pool_path,
        "existing_400_master": existing_master_path,
        "v2": v2_path,
        "published_pilot": published_pilot_manifest,
        "verified_relations": verified_relations_path,
    }
    before = {name: file_sha256(path) for name, path in inputs.items()}
    expected = {
        "pool": expected_pool_sha256,
        "existing_400_master": expected_existing_master_sha256,
        "v2": expected_v2_sha256,
        "published_pilot": expected_published_pilot_sha256,
        "verified_relations": expected_verified_relations_sha256,
    }
    for name, expected_sha in expected.items():
        if expected_sha and before[name].lower() != expected_sha.lower():
            raise CandidateGenerationError(name.upper() + "_SHA256_MISMATCH")
    rows = load_pool(pool_path)
    existing = load_preliminary_pilot(existing_master_path)
    v2 = load_v2_exclusions(v2_path)
    formal_pilot = load_pilot_exclusions(published_pilot_manifest)
    relations = load_relation_audit(verified_relations_path)
    selected, decisions, report = select_extension_candidates(
        rows,
        existing=existing,
        v2=v2,
        formal_pilot=formal_pilot,
        relations=relations,
        seed=seed,
        targets=scaled_targets(target_size),
    )
    report.update(
        {
            "dry_run": not materialize,
            "input_sha256": before,
            "pending_relation_row_uids": tuple(sorted(relations.pending_by_row)),
            "verified_relation_status_counts": dict(relations.status_counts or {}),
        }
    )
    after = {name: file_sha256(path) for name, path in inputs.items()}
    if before != after:
        raise CandidateGenerationError("INPUT_MUTATED_DURING_RUN")
    if materialize:
        if output_dir is None:
            raise CandidateGenerationError("OUTPUT_REQUIRED")
        write_extension_bundle(output_dir, selected, decisions, report, seed=seed)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--existing-400-master", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--published-pilot-exclusion", type=Path, required=True)
    parser.add_argument("--verified-relations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    report = run_generation(
        pool_path=args.pool,
        existing_master_path=args.existing_400_master,
        v2_path=args.v2,
        published_pilot_manifest=args.published_pilot_exclusion,
        verified_relations_path=args.verified_relations,
        output_dir=args.output,
        target_size=args.target_size,
        seed=args.seed,
        materialize=args.materialize,
    )
    safe = {key: value for key, value in report.items() if key != "pending_relation_row_uids"}
    print(json.dumps(safe, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
