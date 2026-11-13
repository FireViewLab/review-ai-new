"""Fail-closed contracts for fixed-split P_text v3 datasets.

The loader never samples, deduplicates, labels, or reassigns a split.  It can
read the canonical v2 ``dataset`` sheet and v3 workbooks produced by the
builder.  Pilot exclusions are hashed manifests, so private review text and
identifiers are never copied into an audit report.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


LABEL_MAPPING = {"NORMAL": 0, "SUSPICIOUS": 1}
SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = frozenset(
    {"platform", "product_id", "review_id", "content", "label", "label_origin", "group_id", "split"}
)

PILOT_MANIFEST_VERSION = "ptext-pilot-exclusion-v1"
PILOT_EXCLUDED_FROM = "TRAIN|VALIDATION|EVALUATION"
PILOT_ENTITY_TYPES = frozenset(
    {
        "ROW_UID",
        "NORMALIZED_CONTENT_HASH",
        "PLATFORM_REVIEW",
        "PLATFORM_PRODUCT",
        "DUPLICATE_CLUSTER",
        "NEAR_DUPLICATE",
        "V2_LINEAGE",
    }
)
PILOT_MANIFEST_FIELDS = (
    "manifest_version",
    "pilot_version",
    "sampling_seed",
    "policy_version",
    "annotation_id",
    "entity_type",
    "entity_key_hash",
    "exclusion_reason",
    "joint_leakage_group_id",
    "excluded_from",
)
RELATION_FIELDS = ("row_uid", "relation_type", "group_id", "validation_status")
VERIFIED_RELATION_TYPES = frozenset({"NEAR_DUPLICATE", "V2_LINEAGE"})
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
HEX_24 = re.compile(r"^[0-9a-f]{24}$")


class ContractError(ValueError):
    """Stable, non-sensitive data-contract failure."""


@dataclass(frozen=True)
class PilotExclusions:
    hashes_by_type: Mapping[str, frozenset[str]]
    manifest_sha256: str
    row_count: int


@dataclass(frozen=True)
class VerifiedRelations:
    groups_by_row: Mapping[str, tuple[tuple[str, str], ...]]
    source_sha256: str | None
    verified_row_count: int


@dataclass(frozen=True)
class FixedDataset:
    examples: Mapping[str, list[dict[str, Any]]]
    metadata: Mapping[str, list[dict[str, Any]]]
    summary: Mapping[str, Any]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalized_content_hash(value: str) -> str:
    return sha256(normalize_text(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(code)
    if any(ord(character) < 32 for character in value):
        raise ContractError(code)
    return value


def _canonical_entity_value(entity_type: str, *parts: object) -> str:
    if entity_type not in PILOT_ENTITY_TYPES:
        raise ContractError("PILOT_ENTITY_TYPE_INVALID")
    expected = 2 if entity_type in {"PLATFORM_REVIEW", "PLATFORM_PRODUCT"} else 1
    if len(parts) != expected:
        raise ContractError("PILOT_ENTITY_PART_COUNT_INVALID")
    values = [_required_text(part, "PILOT_ENTITY_VALUE_INVALID") for part in parts]
    if entity_type == "NORMALIZED_CONTENT_HASH":
        digest = values[0].lower()
        if not HEX_64.fullmatch(digest):
            raise ContractError("PILOT_CONTENT_HASH_INVALID")
        return digest
    if expected == 2:
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return values[0]


def pilot_entity_hash(entity_type: str, *parts: object) -> str:
    canonical = _canonical_entity_value(entity_type, *parts)
    payload = f"{PILOT_MANIFEST_VERSION}\x1f{entity_type}\x1f{canonical}"
    return sha256(payload.encode("utf-8")).hexdigest()


def load_pilot_exclusions(path: str | Path | None) -> PilotExclusions:
    if path is None:
        return PilotExclusions({}, "", 0)
    source = Path(path)
    by_type: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str, str]] = set()
    annotation_groups: dict[str, str] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PILOT_MANIFEST_FIELDS:
            raise ContractError("PILOT_MANIFEST_SCHEMA_INVALID")
        rows = list(reader)
    if not rows:
        raise ContractError("PILOT_MANIFEST_EMPTY")
    for row in rows:
        if row["manifest_version"] != PILOT_MANIFEST_VERSION:
            raise ContractError("PILOT_MANIFEST_VERSION_INVALID")
        if row["excluded_from"] != PILOT_EXCLUDED_FROM:
            raise ContractError("PILOT_EXCLUSION_SCOPE_INVALID")
        entity_type = row["entity_type"]
        if entity_type not in PILOT_ENTITY_TYPES:
            raise ContractError("PILOT_ENTITY_TYPE_INVALID")
        if not HEX_64.fullmatch(row["entity_key_hash"]):
            raise ContractError("PILOT_ENTITY_HASH_INVALID")
        if not HEX_24.fullmatch(row["joint_leakage_group_id"]):
            raise ContractError("PILOT_GROUP_ID_INVALID")
        if row["pilot_version"] != row["policy_version"]:
            raise ContractError("PILOT_POLICY_VERSION_MISMATCH")
        try:
            int(row["sampling_seed"])
        except (TypeError, ValueError) as exc:
            raise ContractError("PILOT_SAMPLING_SEED_INVALID") from exc
        annotation_id = _required_text(row["annotation_id"], "PILOT_ANNOTATION_ID_INVALID")
        group_id = row["joint_leakage_group_id"]
        if annotation_groups.setdefault(annotation_id, group_id) != group_id:
            raise ContractError("PILOT_ANNOTATION_GROUP_MISMATCH")
        key = (annotation_id, entity_type, row["entity_key_hash"])
        if key in seen:
            raise ContractError("PILOT_MANIFEST_DUPLICATE")
        seen.add(key)
        by_type[entity_type].add(row["entity_key_hash"])
    return PilotExclusions(
        {name: frozenset(values) for name, values in by_type.items()},
        file_sha256(source),
        len(rows),
    )


def load_verified_relations(path: str | Path | None) -> VerifiedRelations:
    if path is None:
        return VerifiedRelations({}, None, 0)
    source = Path(path)
    groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    seen: set[tuple[str, str, str]] = set()
    verified = 0
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RELATION_FIELDS:
            raise ContractError("RELATION_SCHEMA_INVALID")
        for row in reader:
            relation_type = row["relation_type"]
            status = row["validation_status"]
            if relation_type not in VERIFIED_RELATION_TYPES:
                raise ContractError("RELATION_TYPE_INVALID")
            if status not in {"VERIFIED", "PROPOSED", "UNREVIEWED", "REJECTED"}:
                raise ContractError("RELATION_STATUS_INVALID")
            row_uid = _required_text(row["row_uid"], "RELATION_ROW_UID_INVALID")
            group_id = _required_text(row["group_id"], "RELATION_GROUP_ID_INVALID")
            key = (row_uid, relation_type, group_id)
            if key in seen:
                raise ContractError("RELATION_DUPLICATE")
            seen.add(key)
            if status == "VERIFIED":
                groups[row_uid].add((relation_type, group_id))
                verified += 1
    return VerifiedRelations(
        {uid: tuple(sorted(values)) for uid, values in groups.items()},
        file_sha256(source),
        verified,
    )


def _row_entities(row: Mapping[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    entities: list[tuple[str, tuple[str, ...]]] = [
        ("NORMALIZED_CONTENT_HASH", (normalized_content_hash(str(row["content"])),)),
        ("PLATFORM_REVIEW", (str(row["platform"]), str(row["review_id"]))),
        ("PLATFORM_PRODUCT", (str(row["platform"]), str(row["product_id"]))),
    ]
    for field, entity_type in (("row_uid", "ROW_UID"), ("duplicate_cluster_id", "DUPLICATE_CLUSTER")):
        value = row.get(field)
        if value is not None and str(value).strip():
            entities.append((entity_type, (str(value).strip(),)))
    return entities


def exclusion_matches(row: Mapping[str, Any], exclusions: PilotExclusions) -> tuple[str, ...]:
    matches = []
    for entity_type, parts in _row_entities(row):
        digest = pilot_entity_hash(entity_type, *parts)
        if digest in exclusions.hashes_by_type.get(entity_type, frozenset()):
            matches.append(entity_type)
    return tuple(matches)


def _load_workbook_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required for P_text v3 datasets") from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if "dataset" not in workbook.sheetnames:
            raise ContractError("DATASET_SHEET_MISSING")
        iterator = workbook["dataset"].iter_rows(values_only=True)
        headers = [str(value).strip() if value is not None else "" for value in next(iterator, ())]
        if not headers or len(headers) != len(set(headers)) or any(not name for name in headers):
            raise ContractError("DATASET_HEADERS_INVALID")
        missing = REQUIRED_COLUMNS.difference(headers)
        if missing:
            raise ContractError("DATASET_COLUMNS_MISSING:" + ",".join(sorted(missing)))
        rows = []
        for excel_row, values in enumerate(iterator, 2):
            if all(value is None for value in values):
                continue
            rows.append(dict(zip(headers, values, strict=True)) | {"_excel_row": excel_row})
        return headers, rows
    finally:
        workbook.close()


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclusions: PilotExclusions | None = None,
    relations: VerifiedRelations | None = None,
    strict_product_isolation: bool = True,
    require_both_labels_per_split: bool = True,
) -> dict[str, Any]:
    if not rows:
        raise ContractError("DATASET_EMPTY")
    exclusions = exclusions or PilotExclusions({}, "", 0)
    relations = relations or VerifiedRelations({}, None, 0)
    scopes: dict[str, dict[Any, set[str]]] = {
        name: defaultdict(set)
        for name in ("group_id", "product", "normalized_content_hash", "duplicate_cluster", "verified_relation")
    }
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    row_uids: set[str] = set()
    split_labels: dict[str, set[str]] = {split: set() for split in SPLITS}
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    origins: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}

    for position, row in enumerate(rows, 1):
        split = row.get("split")
        label = row.get("label")
        if split not in SPLITS:
            raise ContractError(f"SPLIT_INVALID:{position}")
        if label not in LABEL_MAPPING:
            raise ContractError(f"LABEL_INVALID:{position}")
        for field in ("platform", "product_id", "review_id", "group_id", "label_origin"):
            value = row.get(field)
            if value is None or not str(value).strip():
                raise ContractError(f"FIELD_INVALID:{field}:{position}")
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ContractError(f"CONTENT_INVALID:{position}")
        matches = exclusion_matches(row, exclusions)
        if matches:
            raise ContractError(f"PILOT_EXCLUSION_MATCH:{position}:{','.join(matches)}")
        content_hash = normalized_content_hash(content)
        scopes["group_id"][str(row["group_id"])].add(str(split))
        scopes["product"][(str(row["platform"]), str(row["product_id"]))].add(str(split))
        scopes["normalized_content_hash"][content_hash].add(str(split))
        labels_by_hash[content_hash].add(str(label))
        cluster = row.get("duplicate_cluster_id")
        if cluster is not None and str(cluster).strip():
            scopes["duplicate_cluster"][str(cluster).strip()].add(str(split))
        uid = row.get("row_uid")
        if uid is not None and str(uid).strip():
            uid = str(uid).strip()
            if uid in row_uids:
                raise ContractError("ROW_UID_DUPLICATE")
            row_uids.add(uid)
            for relation_type, group_id in relations.groups_by_row.get(uid, ()):
                relation_digest = pilot_entity_hash(relation_type, group_id)
                if relation_digest in exclusions.hashes_by_type.get(relation_type, frozenset()):
                    raise ContractError(f"PILOT_RELATION_EXCLUSION_MATCH:{position}:{relation_type}")
                scopes["verified_relation"][(relation_type, group_id)].add(str(split))
        split_labels[str(split)].add(str(label))
        counts[str(split)][str(label)] += 1
        origins[str(split)][str(row["label_origin"])] += 1

    if any(len(labels) > 1 for labels in labels_by_hash.values()):
        raise ContractError("NORMALIZED_CONTENT_LABEL_CONFLICT")
    leak_counts = {
        scope: sum(len(split_set) > 1 for split_set in values.values())
        for scope, values in scopes.items()
    }
    mandatory = ("group_id", "normalized_content_hash", "duplicate_cluster", "verified_relation")
    leaking = [scope for scope in mandatory if leak_counts[scope]]
    if strict_product_isolation and leak_counts["product"]:
        leaking.append("product")
    if leaking:
        raise ContractError("CROSS_SPLIT_LEAKAGE:" + ",".join(sorted(leaking)))
    if require_both_labels_per_split:
        for split in SPLITS:
            if split_labels[split] != set(LABEL_MAPPING):
                raise ContractError(f"SPLIT_LABEL_SUPPORT_INVALID:{split}")
    return {
        "counts": {split: dict(counts[split]) for split in SPLITS},
        "origins": {split: dict(origins[split]) for split in SPLITS},
        "leakage": leak_counts,
        "strict_product_isolation": strict_product_isolation,
        "pilot_exclusion_manifest_sha256": exclusions.manifest_sha256 or None,
        "pilot_exclusion_manifest_rows": exclusions.row_count,
        "verified_relations_sha256": relations.source_sha256,
        "verified_relation_rows": relations.verified_row_count,
    }


def load_dataset(
    path: str | Path,
    *,
    exclusion_manifest: str | Path | None = None,
    verified_relations: str | Path | None = None,
    strict_product_isolation: bool = True,
    require_both_labels_per_split: bool = True,
) -> FixedDataset:
    source = Path(path)
    _, rows = _load_workbook_rows(source)
    exclusions = load_pilot_exclusions(exclusion_manifest)
    relations = load_verified_relations(verified_relations)
    audit = validate_rows(
        rows,
        exclusions=exclusions,
        relations=relations,
        strict_product_isolation=strict_product_isolation,
        require_both_labels_per_split=require_both_labels_per_split,
    )
    examples = {split: [] for split in SPLITS}
    metadata = {split: [] for split in SPLITS}
    for row in rows:
        split = str(row["split"])
        examples[split].append({"text": row["content"], "label": LABEL_MAPPING[str(row["label"])]})
        metadata[split].append({key: value for key, value in row.items() if key not in {"content", "_excel_row"}})
    summary = {
        "contract_version": "ptext-v3-presplit-v1",
        "path": str(source.resolve()),
        "sha256": file_sha256(source),
        "split_policy": "source split preserved; no resplit, sampling, or automatic labels",
        **audit,
    }
    return FixedDataset(examples, metadata, summary)


class TokenizedTextDataset:
    """Tokenizer receives review text only; all metadata stays outside the model."""

    def __init__(self, examples: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int = 256):
        self.encodings = tokenizer(
            [str(row["text"]) for row in examples], truncation=True, padding=False, max_length=max_length
        )
        self.labels = [int(row["label"]) for row in examples]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            **{key: value[index] for key, value in self.encodings.items()},
            "labels": self.labels[index],
        }
