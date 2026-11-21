"""Deterministic, label-blind candidate generation for the 150K review pool.

Heuristics in this module are retrieval aids only.  They never create a
SUSPICIOUS/NORMAL label, and model outputs are deliberately ignored.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from .data_contract import (
    ContractError,
    PilotExclusions,
    exclusion_matches,
    file_sha256,
    load_pilot_exclusions,
)


POLICY_VERSION = "ptext-v3-candidate-selection-v1"
DEFAULT_SEED = 20261111
GUIDELINE_VERSION = "ptext-labeling-guide-v0.2"
EXPECTED_POOL_SHA256 = "c9d9e5fd5227c5586e95324592427b19ea552c65ad06bcbb0008c798aa2af19a"
EXPECTED_V2_SHA256 = "cd3812b26dd7c782e01ca919c0ff12e3a8bf57e81ecc218522c7e476a7490673"
EXPECTED_PRELIMINARY_PILOT_SHA256 = "bb25277c6b60ecced3290b53e0f9df9f01eba6423577a08e732f94c88127730d"
EXPECTED_GUIDELINE_SHA256 = "562953720d6a6db1e531e790c13872a889f91da583fb75ce702e3d08e209102c"

POOL_FIELDS = (
    "pool_id", "row_uid", "sampling_group", "sampling_reason", "label_status",
    "platform", "source_row", "product_id", "review_id", "product_category_raw",
    "normalized_category", "content", "normalized_content_hash", "content_length",
    "duplicate_cluster_id", "candidate_sources", "original_candidate_member",
    "original_control_eligible", "seed", "policy_version",
)
PILOT_REQUIRED_FIELDS = frozenset(POOL_FIELDS + ("annotation_id",))
RELATION_FIELDS = ("row_uid", "relation_type", "group_id", "validation_status")
RELATION_TYPES = frozenset({"NEAR_DUPLICATE", "V2_LINEAGE"})
RELATION_STATUSES = frozenset({"VERIFIED", "PROPOSED", "UNREVIEWED", "REJECTED"})

SIGNAL_CATEGORIES = (
    "PRODUCT_DESCRIPTION_LIKE",
    "NO_USE_ASSERTIVE_CLAIM",
    "PROMOTIONAL_OR_SALES_LANGUAGE",
    "HIGH_CERTAINTY_EFFICACY_CLAIM",
    "WITHIN_REVIEW_REPETITION",
)
SELECTION_CATEGORIES = SIGNAL_CATEGORIES + (
    "HARD_NEGATIVE_CANDIDATE",
    "CONTROL_REPRESENTATIVE",
    "DIVERSITY_REPRESENTATIVE",
)
SOURCE_SIGNAL_MAPPING = {
    "PRODUCT_DESCRIPTION_REPLACEMENT": {
        "category": "PRODUCT_DESCRIPTION_LIKE", "mapping": "PROXY",
        "note": "legacy source name; requires human signal review",
    },
    "PROMOTIONAL_TEMPLATE": {
        "category": "PROMOTIONAL_OR_SALES_LANGUAGE", "mapping": "PROXY",
        "note": "template retrieval is not a positive label",
    },
    "EXTREME_RESULT_CLAIM": {
        "category": "HIGH_CERTAINTY_EFFICACY_CLAIM", "mapping": "PROXY",
        "note": "claim strength still requires human review",
    },
    "IMMEDIATE_EFFECT_CLAIM": {
        "category": "HIGH_CERTAINTY_EFFICACY_CLAIM", "mapping": "PROXY",
        "note": "immediacy is only a retrieval proxy",
    },
    "HARD_NEGATIVE_CANDIDATE": {
        "category": "HARD_NEGATIVE_CANDIDATE", "mapping": "PROVENANCE",
        "note": "not a NORMAL label",
    },
    "REPEATED_SHORT_TEMPLATE": {
        "category": None, "mapping": "NON_EQUIVALENT_PROXY",
        "note": "cross-review template repetition is not within-review repetition",
    },
    "EXACT_DUPLICATE_CLUSTER": {
        "category": None, "mapping": "NON_EQUIVALENT_PROXY",
        "note": "cross-review exact duplication is handled as leakage, not a signal",
    },
}

DEFAULT_TARGETS: Mapping[str, Mapping[str, int]] = {
    "FROZEN_TEST": {
        "CONTROL_REPRESENTATIVE": 45, "DIVERSITY_REPRESENTATIVE": 15,
        "PRODUCT_DESCRIPTION_LIKE": 8, "NO_USE_ASSERTIVE_CLAIM": 6,
        "PROMOTIONAL_OR_SALES_LANGUAGE": 7, "HIGH_CERTAINTY_EFFICACY_CLAIM": 7,
        "WITHIN_REVIEW_REPETITION": 7, "HARD_NEGATIVE_CANDIDATE": 5,
    },
    "VALIDATION": {
        "CONTROL_REPRESENTATIVE": 35, "DIVERSITY_REPRESENTATIVE": 13,
        "PRODUCT_DESCRIPTION_LIKE": 6, "NO_USE_ASSERTIVE_CLAIM": 5,
        "PROMOTIONAL_OR_SALES_LANGUAGE": 6, "HIGH_CERTAINTY_EFFICACY_CLAIM": 5,
        "WITHIN_REVIEW_REPETITION": 5, "HARD_NEGATIVE_CANDIDATE": 5,
    },
    "TRAIN": {
        "CONTROL_REPRESENTATIVE": 25, "DIVERSITY_REPRESENTATIVE": 10,
        "PRODUCT_DESCRIPTION_LIKE": 35, "NO_USE_ASSERTIVE_CLAIM": 30,
        "PROMOTIONAL_OR_SALES_LANGUAGE": 30, "HIGH_CERTAINTY_EFFICACY_CLAIM": 25,
        "WITHIN_REVIEW_REPETITION": 25, "HARD_NEGATIVE_CANDIDATE": 40,
    },
}

MASTER_FIELDS = POOL_FIELDS + (
    "annotation_id", "split_role", "selection_category", "retrieval_evidence",
    "near_duplicate_review_required",
)
BLIND_FIELDS = ("annotation_id", "display_order", "content")
ANNOTATION_FIELDS = (
    "annotation_id", "annotator_id", "text_status", "signal_decision", "signal_types",
    "other_review_needed", "rationale_codes", "rationale_note", "guideline_version",
    "annotation_round", "annotated_at",
)
ADJUDICATION_FIELDS = (
    "annotation_id", "adjudication_status", "adjudicator_id", "final_text_status",
    "final_signal_decision", "final_signal_types", "final_other_review_needed",
    "adjudication_codes", "adjudication_note", "adjudicated_at",
)
ROLE_FIELDS = ("annotation_id", "split", "role_name")
DECISION_FIELDS = ("row_uid", "decision", "reason", "annotation_id", "split_role")


class CandidateGenerationError(ContractError):
    """Stable, content-free candidate generation failure."""


@dataclass(frozen=True)
class PoolRow:
    values: Mapping[str, str]
    categories: tuple[str, ...]
    evidence: tuple[str, ...]

    def __getattr__(self, name: str) -> str:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class IdentityExclusions:
    row_uids: frozenset[str] = frozenset()
    hashes: frozenset[str] = frozenset()
    reviews: frozenset[tuple[str, str]] = frozenset()
    products: frozenset[tuple[str, str]] = frozenset()
    clusters: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RelationAudit:
    verified_by_row: Mapping[str, tuple[tuple[str, str], ...]]
    pending_by_row: Mapping[str, tuple[tuple[str, str], ...]]
    source_sha256: str | None = None
    status_counts: Mapping[str, int] | None = None


@dataclass(frozen=True)
class SelectedRow:
    row: PoolRow
    role: str
    category: str
    annotation_id: str


_NO_USE = re.compile(r"(사용(?:해|한|하지|전|없이)|써\s*보지|먹어\s*보지|구매만|아직\s*(?:안|못))")
_ASSERTIVE = re.compile(r"(확실|무조건|분명|효과|개선|완치|최고|필수|보장|탁월|강력)")
_SENTENCE_SPLIT = re.compile(r"[.!?。！？\n]+")


def _required(value: object, code: str) -> str:
    text = str(value) if value is not None else ""
    if not text or text != text.strip():
        raise CandidateGenerationError(code)
    return text


def _stable_key(seed: int, *parts: str) -> str:
    return sha256("\x1f".join((str(seed), POLICY_VERSION, *parts)).encode("utf-8")).hexdigest()


def pool_content_hash(value: str) -> str:
    """Fingerprint used by the materialized 150K pool (no case folding)."""
    return sha256(" ".join(value.split()).encode("utf-8")).hexdigest()


def _within_review_repetition(content: str) -> bool:
    units = [re.sub(r"\s+", " ", value).strip().casefold() for value in _SENTENCE_SPLIT.split(content)]
    units = [value for value in units if len(value) >= 5]
    if len(units) != len(set(units)):
        return True
    tokens = re.findall(r"[0-9A-Za-z가-힣]+", content.casefold())
    if len(tokens) < 8:
        return False
    grams = [tuple(tokens[index:index + 4]) for index in range(len(tokens) - 3)]
    return len(grams) != len(set(grams))


def classify_candidate(row: Mapping[str, str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return retrieval categories/evidence without deriving a truth label."""
    categories: set[str] = set()
    evidence: set[str] = set()
    sources = {value for value in row.get("candidate_sources", "").split("|") if value}
    for source in sources:
        mapping = SOURCE_SIGNAL_MAPPING.get(source)
        if mapping and mapping["category"]:
            categories.add(str(mapping["category"]))
            evidence.add(f"SOURCE_PROXY:{source}")
    content = row.get("content", "")
    if _NO_USE.search(content) and _ASSERTIVE.search(content):
        categories.add("NO_USE_ASSERTIVE_CLAIM")
        evidence.add("HEURISTIC:NO_USE_AND_ASSERTIVE_TERM")
    if _within_review_repetition(content):
        categories.add("WITHIN_REVIEW_REPETITION")
        evidence.add("HEURISTIC:REPEATED_SENTENCE_OR_4GRAM")
    if row.get("sampling_group") == "CONTROL":
        categories.add("CONTROL_REPRESENTATIVE")
        evidence.add("PROVENANCE:CONTROL")
    if row.get("sampling_group") == "DIVERSITY":
        categories.add("DIVERSITY_REPRESENTATIVE")
        evidence.add("PROVENANCE:DIVERSITY")
    return tuple(sorted(categories)), tuple(sorted(evidence))


def _read_csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        return fields, [dict(row) for row in reader]


def load_pool(path: str | Path) -> list[PoolRow]:
    fields, raw_rows = _read_csv_rows(Path(path))
    if fields != POOL_FIELDS:
        raise CandidateGenerationError("POOL_SCHEMA_INVALID")
    rows: list[PoolRow] = []
    seen: set[str] = set()
    for raw in raw_rows:
        uid = _required(raw["row_uid"], "POOL_ROW_UID_INVALID")
        if uid in seen:
            raise CandidateGenerationError("POOL_ROW_UID_DUPLICATE")
        seen.add(uid)
        if raw["label_status"] != "UNLABELED":
            raise CandidateGenerationError("POOL_MUST_BE_UNLABELED")
        if raw["sampling_group"] not in {"CANDIDATE", "CONTROL", "DIVERSITY"}:
            raise CandidateGenerationError("POOL_SAMPLING_GROUP_INVALID")
        if pool_content_hash(raw["content"]) != raw["normalized_content_hash"].lower():
            raise CandidateGenerationError("POOL_CONTENT_HASH_MISMATCH")
        categories, evidence = classify_candidate(raw)
        rows.append(PoolRow(raw, categories, evidence))
    return rows


def load_preliminary_pilot(path: str | Path) -> IdentityExclusions:
    fields, rows = _read_csv_rows(Path(path))
    if not PILOT_REQUIRED_FIELDS.issubset(fields):
        raise CandidateGenerationError("PRELIMINARY_PILOT_SCHEMA_INVALID")
    return IdentityExclusions(
        frozenset(row["row_uid"] for row in rows),
        frozenset(row["normalized_content_hash"].lower() for row in rows),
        frozenset((row["platform"], row["review_id"]) for row in rows),
        frozenset((row["platform"], row["product_id"]) for row in rows),
        frozenset(row["duplicate_cluster_id"] for row in rows if row["duplicate_cluster_id"]),
    )


def load_v2_exclusions(path: str | Path) -> IdentityExclusions:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required") from exc
    workbook = load_workbook(Path(path), read_only=True, data_only=True)
    try:
        if "dataset" not in workbook.sheetnames:
            raise CandidateGenerationError("V2_DATASET_SHEET_MISSING")
        iterator = workbook["dataset"].iter_rows(values_only=True)
        headers = tuple(str(value).strip() if value is not None else "" for value in next(iterator, ()))
        required = {"platform", "product_id", "review_id", "content"}
        if not required.issubset(headers):
            raise CandidateGenerationError("V2_SCHEMA_INVALID")
        indices = {name: headers.index(name) for name in headers}
        hashes, reviews, products, clusters = set(), set(), set(), set()
        for values in iterator:
            if not values or all(value is None for value in values):
                continue
            platform = str(values[indices["platform"]])
            product = str(values[indices["product_id"]])
            review = str(values[indices["review_id"]])
            content = str(values[indices["content"]])
            hashes.add(pool_content_hash(content))
            reviews.add((platform, review))
            products.add((platform, product))
            if "duplicate_cluster_id" in indices and values[indices["duplicate_cluster_id"]]:
                clusters.add(str(values[indices["duplicate_cluster_id"]]))
        return IdentityExclusions(hashes=frozenset(hashes), reviews=frozenset(reviews),
                                  products=frozenset(products), clusters=frozenset(clusters))
    finally:
        workbook.close()


def load_relation_audit(path: str | Path | None) -> RelationAudit:
    if path is None:
        return RelationAudit({}, {}, None, {})
    source = Path(path)
    fields, rows = _read_csv_rows(source)
    if fields != RELATION_FIELDS:
        raise CandidateGenerationError("RELATION_SCHEMA_INVALID")
    verified: dict[str, set[tuple[str, str]]] = defaultdict(set)
    pending: dict[str, set[tuple[str, str]]] = defaultdict(set)
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        relation_type, status = row["relation_type"], row["validation_status"]
        if relation_type not in RELATION_TYPES or status not in RELATION_STATUSES:
            raise CandidateGenerationError("RELATION_VALUE_INVALID")
        key = (_required(row["row_uid"], "RELATION_ROW_INVALID"), relation_type,
               _required(row["group_id"], "RELATION_GROUP_INVALID"))
        if key in seen:
            raise CandidateGenerationError("RELATION_DUPLICATE")
        seen.add(key)
        counts[status] += 1
        if status == "VERIFIED":
            verified[key[0]].add((key[1], key[2]))
        elif status in {"PROPOSED", "UNREVIEWED"}:
            pending[key[0]].add((key[1], key[2]))
    return RelationAudit(
        {key: tuple(sorted(value)) for key, value in verified.items()},
        {key: tuple(sorted(value)) for key, value in pending.items()},
        file_sha256(source), dict(sorted(counts.items())),
    )


def _identity_reason(row: PoolRow, exclusions: IdentityExclusions, prefix: str) -> str | None:
    if row.row_uid in exclusions.row_uids:
        return f"{prefix}_ROW_UID"
    if row.normalized_content_hash.lower() in exclusions.hashes:
        return f"{prefix}_EXACT_CONTENT"
    if (row.platform, row.review_id) in exclusions.reviews:
        return f"{prefix}_REVIEW"
    if (row.platform, row.product_id) in exclusions.products:
        return f"{prefix}_PRODUCT"
    if row.duplicate_cluster_id and row.duplicate_cluster_id in exclusions.clusters:
        return f"{prefix}_DUPLICATE_CLUSTER"
    return None


class _SelectionState:
    def __init__(self, relations: RelationAudit):
        self.relations = relations
        self.rows: set[str] = set()
        self.hashes: set[str] = set()
        self.reviews: set[tuple[str, str]] = set()
        self.products: set[tuple[str, str]] = set()
        self.clusters: set[str] = set()
        self.verified: set[tuple[str, str]] = set()

    def reason(self, row: PoolRow) -> str | None:
        checks = (
            (row.row_uid in self.rows, "CROSS_ROLE_ROW_UID"),
            (row.normalized_content_hash in self.hashes, "CROSS_ROLE_EXACT_CONTENT"),
            ((row.platform, row.review_id) in self.reviews, "CROSS_ROLE_REVIEW"),
            ((row.platform, row.product_id) in self.products, "CROSS_ROLE_PRODUCT"),
            (bool(row.duplicate_cluster_id and row.duplicate_cluster_id in self.clusters),
             "CROSS_ROLE_DUPLICATE_CLUSTER"),
            (any(value in self.verified for value in self.relations.verified_by_row.get(row.row_uid, ())),
             "CROSS_ROLE_VERIFIED_RELATION"),
        )
        return next((reason for matched, reason in checks if matched), None)

    def add(self, row: PoolRow) -> None:
        if self.reason(row):
            raise CandidateGenerationError("SELECTION_STATE_CONFLICT")
        self.rows.add(row.row_uid)
        self.hashes.add(row.normalized_content_hash)
        self.reviews.add((row.platform, row.review_id))
        self.products.add((row.platform, row.product_id))
        if row.duplicate_cluster_id:
            self.clusters.add(row.duplicate_cluster_id)
        self.verified.update(self.relations.verified_by_row.get(row.row_uid, ()))


def select_candidates(
    rows: Sequence[PoolRow], *, pilot: IdentityExclusions = IdentityExclusions(),
    v2: IdentityExclusions = IdentityExclusions(), formal_pilot: PilotExclusions = PilotExclusions({}, "", 0),
    relations: RelationAudit = RelationAudit({}, {}), seed: int = DEFAULT_SEED,
    targets: Mapping[str, Mapping[str, int]] = DEFAULT_TARGETS,
) -> tuple[list[SelectedRow], list[dict[str, str]], dict[str, object]]:
    """Assign roles before labels, enforcing identity/relation isolation globally."""
    eligible: list[PoolRow] = []
    decisions: list[dict[str, str]] = []
    exclusion_counts: Counter[str] = Counter()
    for row in rows:
        reason = _identity_reason(row, pilot, "PILOT_PRELIMINARY")
        reason = reason or _identity_reason(row, v2, "V2")
        if not reason and exclusion_matches(row.values, formal_pilot):
            reason = "PILOT_PUBLISHED_MANIFEST"
        if reason:
            exclusion_counts[reason] += 1
            decisions.append({"row_uid": row.row_uid, "decision": "EXCLUDED", "reason": reason,
                              "annotation_id": "", "split_role": ""})
        else:
            eligible.append(row)

    state = _SelectionState(relations)
    selected: list[SelectedRow] = []
    shortfalls: dict[str, dict[str, int]] = {}
    hard_rejections: Counter[str] = Counter()
    for role in ("FROZEN_TEST", "VALIDATION", "TRAIN"):
        role_shortfalls: dict[str, int] = {}
        feature_counts: Counter[tuple[str, str]] = Counter()
        for category, target in targets.get(role, {}).items():
            if category not in SELECTION_CATEGORIES or target < 0:
                raise CandidateGenerationError("TARGETS_INVALID")
            candidates = [row for row in eligible if category in row.categories and row.row_uid not in state.rows]
            candidates.sort(key=lambda row: (
                feature_counts[("platform", row.platform)],
                feature_counts[("category", row.normalized_category)],
                _stable_key(seed, role, category, row.row_uid), row.row_uid,
            ))
            chosen = 0
            for row in candidates:
                reason = state.reason(row)
                if reason:
                    hard_rejections[reason] += 1
                    continue
                state.add(row)
                annotation_id = _stable_key(seed, "annotation", row.row_uid)[:24]
                selected.append(SelectedRow(row, role, category, annotation_id))
                feature_counts[("platform", row.platform)] += 1
                feature_counts[("category", row.normalized_category)] += 1
                chosen += 1
                if chosen == target:
                    break
            if chosen < target:
                role_shortfalls[category] = target - chosen
        if role_shortfalls:
            shortfalls[role] = role_shortfalls

    for item in selected:
        decisions.append({"row_uid": item.row.row_uid, "decision": "SELECTED",
                          "reason": item.category, "annotation_id": item.annotation_id,
                          "split_role": item.role})
    decided = {row["row_uid"] for row in decisions}
    for row in eligible:
        if row.row_uid not in decided:
            decisions.append({"row_uid": row.row_uid, "decision": "NOT_SELECTED",
                              "reason": "QUOTA_OR_HARD_CONSTRAINT", "annotation_id": "",
                              "split_role": ""})

    pending_selected = {
        item.row.row_uid: relations.pending_by_row[item.row.row_uid]
        for item in selected if item.row.row_uid in relations.pending_by_row
    }
    pending_roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in selected:
        for relation in relations.pending_by_row.get(item.row.row_uid, ()):
            pending_roles[relation].add(item.role)
    cross_role_pending = sorted(f"{kind}:{group}" for (kind, group), roles in pending_roles.items() if len(roles) > 1)
    report: dict[str, object] = {
        "policy_version": POLICY_VERSION,
        "seed": seed,
        "targets": {role: dict(values) for role, values in targets.items()},
        "selected_by_role": dict(sorted(Counter(item.role for item in selected).items())),
        "selected_by_category": dict(sorted(Counter(item.category for item in selected).items())),
        "shortfalls": shortfalls,
        "exclusions": dict(sorted(exclusion_counts.items())),
        "hard_constraint_rejections": dict(sorted(hard_rejections.items())),
        "pending_relation_selected_rows": len(pending_selected),
        "unresolved_cross_role_relation_risks": cross_role_pending,
        "final_dataset_ready": not shortfalls and not pending_selected and not cross_role_pending,
        "label_policy": "NO_AUTOMATIC_LABELS; human adjudication required",
        "model_output_policy": "model predictions are ignored and never treated as truth",
        "evaluation_policy": "roles assigned before labels; frozen test is never rebuilt for class balance",
        "known_omission_risks": [
            "lexical retrieval can miss implicit or novel suspicious signals",
            "source proxies are not equivalent to formal human-review signals",
            "pending near-duplicate relations require review before final dataset publication",
        ],
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


def write_candidate_bundle(
    output_dir: str | Path, selected: Sequence[SelectedRow], decisions: Sequence[Mapping[str, str]],
    metadata: Mapping[str, object], *, guideline_path: str | Path, reviewer_ids: Sequence[str] = ("reviewer_a", "reviewer_b"),
    seed: int = DEFAULT_SEED,
) -> None:
    root = Path(output_dir)
    if root.exists():
        raise CandidateGenerationError("OUTPUT_EXISTS")
    if len(reviewer_ids) != 2 or len(set(reviewer_ids)) != 2:
        raise CandidateGenerationError("TWO_DISTINCT_REVIEWERS_REQUIRED")
    guideline = Path(guideline_path)
    root.mkdir(parents=True, exist_ok=False)
    master = []
    for item in selected:
        values = {field: item.row.values[field] for field in POOL_FIELDS}
        values.update({
            "annotation_id": item.annotation_id, "split_role": item.role,
            "selection_category": item.category, "retrieval_evidence": "|".join(item.row.evidence),
            "near_duplicate_review_required": "YES" if item.row.row_uid in metadata.get("pending_relation_row_uids", ()) else "NO",
        })
        master.append(values)
    _write_csv(root / "master_candidate_manifest.csv", MASTER_FIELDS, master)
    _write_csv(root / "split_role_manifest.csv", ROLE_FIELDS, (
        {"annotation_id": item.annotation_id,
         "split": {"TRAIN": "train", "VALIDATION": "validation", "FROZEN_TEST": "test"}[item.role],
         "role_name": item.role} for item in selected
    ))
    _write_csv(root / "selection_exclusion_manifest.csv", DECISION_FIELDS, decisions)
    for reviewer in reviewer_ids:
        ordered = sorted(selected, key=lambda item: _stable_key(seed, "reviewer", reviewer, item.annotation_id))
        _write_csv(root / "blind_review" / "reviewer_packets" / f"{reviewer}.csv", BLIND_FIELDS, (
            {"annotation_id": item.annotation_id, "display_order": index, "content": item.row.content}
            for index, item in enumerate(ordered, 1)
        ))
        _write_csv(root / "blind_review" / "annotations" / f"{reviewer}_round1.csv", ANNOTATION_FIELDS, (
            {
                "annotation_id": item.annotation_id, "annotator_id": reviewer,
                "text_status": "", "signal_decision": "", "signal_types": "",
                "other_review_needed": "", "rationale_codes": "", "rationale_note": "",
                "guideline_version": GUIDELINE_VERSION, "annotation_round": "1", "annotated_at": "",
            }
            for item in ordered
        ))
    _write_csv(root / "blind_review" / "adjudication" / "final_labels.csv", ADJUDICATION_FIELDS, (
        {
            "annotation_id": item.annotation_id, "adjudication_status": "", "adjudicator_id": "",
            "final_text_status": "", "final_signal_decision": "", "final_signal_types": "",
            "final_other_review_needed": "", "adjudication_codes": "", "adjudication_note": "",
            "adjudicated_at": "",
        }
        for item in sorted(selected, key=lambda value: value.annotation_id)
    ))
    review_contract = {
        "contract_version": "ptext-human-review-contract-v0.2",
        "guideline_version": GUIDELINE_VERSION,
        "guideline_sha256": file_sha256(guideline),
        "reviewer_ids": list(reviewer_ids),
        "review_mode": "TWO_INDEPENDENT_BLIND_REVIEWERS",
        "packet_fields": list(BLIND_FIELDS),
        "annotation_fields": list(ANNOTATION_FIELDS),
        "adjudication_fields": list(ADJUDICATION_FIELDS),
        "annotation_count": len(selected),
        "sampling_seed": seed,
        "guideline_file": guideline.name,
    }
    _write_json(root / "blind_review" / "review_contract_manifest.json", review_contract)
    generation = dict(metadata)
    generation.pop("pending_relation_row_uids", None)
    generation.update({"source_signal_mapping": SOURCE_SIGNAL_MAPPING, "human_review": review_contract})
    _write_json(root / "candidate_generation_metadata.json", generation)


def verify_sha(path: str | Path, expected: str, code: str) -> str:
    actual = file_sha256(path)
    if expected and actual.lower() != expected.lower():
        raise CandidateGenerationError(code)
    return actual


def _assert_private_output_path(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    private_root = (repo_root / "artifacts" / "p_text_v3_candidates").resolve()
    resolved = path.resolve()
    if resolved == private_root or private_root not in resolved.parents:
        raise CandidateGenerationError("OUTPUT_NOT_IN_GITIGNORED_PRIVATE_ROOT")


def run_generation(
    *, pool_path: Path, v2_path: Path, pilot_path: Path, guideline_path: Path,
    output_dir: Path | None = None, relation_path: Path | None = None,
    published_pilot_manifest: Path | None = None, seed: int = DEFAULT_SEED,
    materialize: bool = False, expected_pool_sha256: str = EXPECTED_POOL_SHA256,
    expected_v2_sha256: str = EXPECTED_V2_SHA256,
    expected_pilot_sha256: str = EXPECTED_PRELIMINARY_PILOT_SHA256,
    expected_guideline_sha256: str = EXPECTED_GUIDELINE_SHA256,
    targets: Mapping[str, Mapping[str, int]] = DEFAULT_TARGETS,
) -> dict[str, object]:
    inputs = {"pool": pool_path, "v2": v2_path, "pilot_preliminary": pilot_path, "guideline": guideline_path}
    expected = {"pool": expected_pool_sha256, "v2": expected_v2_sha256,
                "pilot_preliminary": expected_pilot_sha256, "guideline": expected_guideline_sha256}
    before = {name: verify_sha(path, expected[name], f"{name.upper()}_SHA256_MISMATCH") for name, path in inputs.items()}
    rows = load_pool(pool_path)
    pilot = load_preliminary_pilot(pilot_path)
    v2 = load_v2_exclusions(v2_path)
    relations = load_relation_audit(relation_path)
    formal = load_pilot_exclusions(published_pilot_manifest)
    selected, decisions, report = select_candidates(rows, pilot=pilot, v2=v2, formal_pilot=formal,
                                                     relations=relations, seed=seed, targets=targets)
    report.update({
        "input_sha256": before,
        "input_rows": len(rows),
        "dry_run": not materialize,
        "published_pilot_exclusion_sha256": formal.manifest_sha256 or None,
        "relation_sha256": relations.source_sha256,
        "relation_status_counts": dict(relations.status_counts or {}),
        "pending_relation_row_uids": tuple(sorted(relations.pending_by_row)),
    })
    publication_gates = []
    if relation_path is None:
        publication_gates.append("F3_4_2_RELATION_ARTIFACT_REQUIRED")
    if published_pilot_manifest is None:
        publication_gates.append("PUBLISHED_PILOT_EXCLUSION_REQUIRED")
    report["publication_gates"] = publication_gates
    report["final_dataset_ready"] = bool(report["final_dataset_ready"] and not publication_gates)
    after = {name: file_sha256(path) for name, path in inputs.items()}
    if before != after:
        raise CandidateGenerationError("INPUT_MUTATED_DURING_RUN")
    if materialize:
        if output_dir is None:
            raise CandidateGenerationError("OUTPUT_REQUIRED")
        _assert_private_output_path(output_dir)
        write_candidate_bundle(output_dir, selected, decisions, report, guideline_path=guideline_path, seed=seed)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label-blind P_text v3 review candidates")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--pilot-preliminary", type=Path, required=True)
    parser.add_argument("--guideline", type=Path, required=True)
    parser.add_argument("--verified-relations", type=Path)
    parser.add_argument("--published-pilot-exclusion", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--materialize", action="store_true", help="write a new private output directory")
    args = parser.parse_args()
    report = run_generation(pool_path=args.pool, v2_path=args.v2, pilot_path=args.pilot_preliminary,
                            guideline_path=args.guideline, output_dir=args.output,
                            relation_path=args.verified_relations,
                            published_pilot_manifest=args.published_pilot_exclusion,
                            seed=args.seed, materialize=args.materialize)
    # Counts and hashes only; review content and identifiers are never logged.
    print(json.dumps({key: value for key, value in report.items() if key not in {"pending_relation_row_uids"}},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
