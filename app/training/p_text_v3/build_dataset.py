"""Build a fixed-split v3 workbook from adjudicated human-review records.

Sampling groups are deliberately ignored as labels.  Only adjudicated
``SUFFICIENT/PRESENT`` and ``SUFFICIENT/ABSENT`` records are mapped to the
binary task.  Split membership must come from a separately approved role
manifest; this command never creates a split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from .data_contract import (
    ContractError,
    SPLITS,
    exclusion_matches,
    file_sha256,
    load_pilot_exclusions,
    load_dataset,
    load_verified_relations,
    validate_rows,
)
from .human_review_validator import validate_human_review


MASTER_REQUIRED = frozenset(
    {"annotation_id", "row_uid", "platform", "product_id", "review_id", "content"}
)
FINAL_REQUIRED = frozenset(
    {"annotation_id", "adjudication_status", "final_text_status", "final_signal_decision"}
)
ROLE_FIELDS = ("annotation_id", "split")


def _read_csv(path: str | Path, required: frozenset[str]) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = required.difference(fields)
        if missing:
            raise ContractError("CSV_COLUMNS_MISSING:" + ",".join(sorted(missing)))
        rows = list(reader)
    return rows


def _unique_by(rows: Iterable[Mapping[str, str]], field: str, code: str) -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        key = str(row.get(field, "")).strip()
        if not key or key in result:
            raise ContractError(code)
        result[key] = row
    return result


def adjudicated_binary_label(row: Mapping[str, str]) -> str | None:
    if row.get("adjudication_status") not in {"not_required", "resolved"}:
        return None
    if row.get("final_text_status") != "SUFFICIENT":
        return None
    if row.get("final_signal_decision") == "PRESENT":
        return "SUSPICIOUS"
    if row.get("final_signal_decision") == "ABSENT":
        return "NORMAL"
    return None


def build_human_rows(
    master_rows: Iterable[Mapping[str, str]],
    final_rows: Iterable[Mapping[str, str]],
    role_rows: Iterable[Mapping[str, str]],
    *,
    exclusions=None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    masters = _unique_by(master_rows, "annotation_id", "MASTER_ANNOTATION_ID_INVALID")
    finals = _unique_by(final_rows, "annotation_id", "FINAL_ANNOTATION_ID_INVALID")
    roles = _unique_by(role_rows, "annotation_id", "ROLE_ANNOTATION_ID_INVALID")
    exclusions = exclusions or load_pilot_exclusions(None)
    output: list[dict[str, Any]] = []
    audit = {"master": len(masters), "eligible_label": 0, "excluded_non_binary": 0, "excluded_no_role": 0, "excluded_pilot": 0}
    for annotation_id, master in masters.items():
        final = finals.get(annotation_id)
        label = adjudicated_binary_label(final or {})
        if label is None:
            audit["excluded_non_binary"] += 1
            continue
        audit["eligible_label"] += 1
        role = roles.get(annotation_id)
        if role is None:
            audit["excluded_no_role"] += 1
            continue
        split = role.get("split")
        if split not in SPLITS:
            raise ContractError("ROLE_SPLIT_INVALID")
        row = {
            "platform": str(master["platform"]).strip(),
            "product_id": str(master["product_id"]).strip(),
            "review_id": str(master["review_id"]).strip(),
            "content": master["content"],
            "label": label,
            "label_origin": "human_review_v3",
            "group_id": str(master.get("group_id") or f"review:{master['row_uid']}").strip(),
            "split": split,
            "row_uid": str(master["row_uid"]).strip(),
            "duplicate_cluster_id": str(master.get("duplicate_cluster_id", "")).strip(),
            "annotation_id": annotation_id,
            "source_set": str(master.get("sampling_group", "")).strip(),
        }
        if exclusion_matches(row, exclusions):
            audit["excluded_pilot"] += 1
            continue
        output.append(row)
    return output, audit


def write_dataset_workbook(path: str | Path, rows: list[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    from openpyxl import Workbook

    target = Path(path)
    if target.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "platform", "product_id", "review_id", "content", "label", "label_origin",
        "group_id", "split", "row_uid", "duplicate_cluster_id", "annotation_id", "source_set",
    )
    workbook = Workbook(write_only=True)
    dataset = workbook.create_sheet("dataset")
    dataset.append(fields)
    for row in rows:
        dataset.append([row.get(field) for field in fields])
    page = workbook.create_sheet("summary")
    page.append(("key", "value"))
    for key, value in summary.items():
        page.append((key, json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value))
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.stem + ".", suffix=".tmp", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        workbook.save(temp_path)
        temp_path.replace(target)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def run(
    master_path: str | Path,
    final_labels_path: str | Path,
    roles_path: str | Path,
    output_path: str | Path,
    *,
    guideline_path: str | Path,
    review_contract_path: str | Path,
    source_packet_contract_path: str | Path,
    reviewer_a_path: str | Path,
    reviewer_b_path: str | Path,
    base_v2_path: str | Path | None = None,
    exclusion_manifest: str | Path | None = None,
    verified_relations: str | Path | None = None,
) -> dict[str, Any]:
    masters = _read_csv(master_path, MASTER_REQUIRED)
    roles = _read_csv(roles_path, frozenset(ROLE_FIELDS))
    review_validation = validate_human_review(
        master_rows=masters,
        role_rows=roles,
        guideline_path=guideline_path,
        review_contract_path=review_contract_path,
        source_packet_contract_path=source_packet_contract_path,
        reviewer_a_path=reviewer_a_path,
        reviewer_b_path=reviewer_b_path,
        final_labels_path=final_labels_path,
    )
    finals = _read_csv(final_labels_path, FINAL_REQUIRED)
    exclusions = load_pilot_exclusions(exclusion_manifest)
    relations = load_verified_relations(verified_relations)
    human_rows, build_audit = build_human_rows(masters, finals, roles, exclusions=exclusions)
    base_rows: list[dict[str, Any]] = []
    if base_v2_path is not None:
        # Canonical v2 is compatible in explicit legacy-audit mode.  Only its
        # fixed train split is carried into v3; v2 validation/test never become
        # the new validation/frozen test and their known product overlap is not
        # propagated.
        base = load_dataset(base_v2_path, strict_product_isolation=False)
        for example, metadata in zip(base.examples["train"], base.metadata["train"], strict=True):
            base_rows.append(
                dict(metadata)
                | {
                    "content": example["text"],
                    "label": "SUSPICIOUS" if example["label"] else "NORMAL",
                    "split": "train",
                }
            )
    rows = base_rows + human_rows
    validation = validate_rows(
        rows,
        exclusions=exclusions,
        relations=relations,
        strict_product_isolation=True,
        require_both_labels_per_split=True,
    )
    summary = {
        "contract_version": "ptext-v3-presplit-v1",
        "automatic_labels_assigned": False,
        "sampling_groups_used_as_labels": False,
        "split_generated": False,
        "official_human_review_validated": True,
        "human_review_validation": review_validation,
        "build": build_audit,
        "base_v2_train_rows": len(base_rows),
        "human_review_rows": len(human_rows),
        "validation": validation,
        "inputs": {
            "master_sha256": file_sha256(master_path),
            "final_labels_sha256": file_sha256(final_labels_path),
            "roles_sha256": file_sha256(roles_path),
            "guideline_sha256": file_sha256(guideline_path),
            "review_contract_sha256": file_sha256(review_contract_path),
            "source_packet_contract_sha256": file_sha256(source_packet_contract_path),
            "reviewer_a_sha256": file_sha256(reviewer_a_path),
            "reviewer_b_sha256": file_sha256(reviewer_b_path),
            "base_v2_sha256": file_sha256(base_v2_path) if base_v2_path is not None else None,
        },
    }
    write_dataset_workbook(output_path, rows, summary)
    return summary | {"output_sha256": file_sha256(output_path)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True)
    parser.add_argument("--final-labels", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--guideline", required=True)
    parser.add_argument("--review-contract", required=True)
    parser.add_argument("--source-packet-contract", required=True)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    parser.add_argument("--base-v2")
    parser.add_argument("--pilot-exclusions")
    parser.add_argument("--verified-relations")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run(
        args.master,
        args.final_labels,
        args.roles,
        args.output,
        guideline_path=args.guideline,
        review_contract_path=args.review_contract,
        source_packet_contract_path=args.source_packet_contract,
        reviewer_a_path=args.reviewer_a,
        reviewer_b_path=args.reviewer_b,
        base_v2_path=args.base_v2,
        exclusion_manifest=args.pilot_exclusions,
        verified_relations=args.verified_relations,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
