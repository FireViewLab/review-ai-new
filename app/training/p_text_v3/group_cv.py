"""Read-only group-stratified cross-validation planning for P_text v3 Train."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data_contract import VerifiedRelations, load_dataset, load_verified_relations, normalized_content_hash


class GroupCVError(ValueError):
    """Content-free planning error."""


class _UnionFind:
    def __init__(self, values: Sequence[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _stable(seed: int, value: str) -> str:
    return sha256(f"{seed}\x1f{value}".encode()).hexdigest()


def _connected_groups(
    rows: Sequence[Mapping[str, Any]], relations: VerifiedRelations
) -> list[list[Mapping[str, Any]]]:
    uids = [str(row.get("row_uid", "")).strip() for row in rows]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise GroupCVError("TRAIN_ROW_UID_INVALID")
    union = _UnionFind(uids)
    owners: dict[tuple[Any, ...], str] = {}
    for row, uid in zip(rows, uids, strict=True):
        keys: list[tuple[Any, ...]] = [
            ("product", str(row["platform"]), str(row["product_id"])),
            ("content", normalized_content_hash(str(row["content"]))),
        ]
        cluster_value = row.get("duplicate_cluster_id")
        cluster = str(cluster_value).strip() if cluster_value is not None else ""
        if cluster:
            keys.append(("cluster", cluster))
        keys.extend(
            ("verified", relation_type, group_id)
            for relation_type, group_id in relations.groups_by_row.get(uid, ())
        )
        for key in keys:
            if key in owners:
                union.union(uid, owners[key])
            else:
                owners[key] = uid
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row, uid in zip(rows, uids, strict=True):
        grouped[union.find(uid)].append(row)
    return list(grouped.values())


def plan_group_kfold(
    rows: Sequence[Mapping[str, Any]],
    *,
    relations: VerifiedRelations = VerifiedRelations({}, None, 0),
    n_splits: int = 5,
    seed: int = 20261114,
) -> dict[str, Any]:
    """Create aggregate-only fold diagnostics; never returns row identifiers."""
    if type(n_splits) is not int or n_splits < 2:
        raise GroupCVError("N_SPLITS_INVALID")
    train_rows = [row for row in rows if row.get("split") == "train"]
    held_out = Counter(str(row.get("split")) for row in rows if row.get("split") != "train")
    if not train_rows:
        raise GroupCVError("TRAIN_ROWS_MISSING")
    if any(row.get("label") not in {"NORMAL", "SUSPICIOUS"} for row in train_rows):
        raise GroupCVError("TRAIN_LABEL_INVALID")
    groups = _connected_groups(train_rows, relations)
    group_records = []
    label_group_support = Counter()
    total_labels = Counter(str(row["label"]) for row in train_rows)
    for group in groups:
        labels = Counter(str(row["label"]) for row in group)
        for label in labels:
            label_group_support[label] += 1
        fingerprint = sha256(
            "\x1f".join(sorted(str(row["row_uid"]) for row in group)).encode()
        ).hexdigest()
        group_records.append((fingerprint, group, labels))
    reasons = []
    for label in ("NORMAL", "SUSPICIOUS"):
        if total_labels[label] < n_splits:
            reasons.append(f"{label}_ROW_SUPPORT_LT_N_SPLITS")
        if label_group_support[label] < n_splits:
            reasons.append(f"{label}_GROUP_SUPPORT_LT_N_SPLITS")
    if len(groups) < n_splits:
        reasons.append("GROUP_COUNT_LT_N_SPLITS")

    target = {label: total_labels[label] / n_splits for label in ("NORMAL", "SUSPICIOUS")}
    target_total = len(train_rows) / n_splits
    fold_groups: list[list[Mapping[str, Any]]] = [[] for _ in range(n_splits)]
    fold_counts = [Counter() for _ in range(n_splits)]
    ordered = sorted(
        group_records,
        key=lambda item: (
            -max(item[2].values()),
            -len(item[1]),
            _stable(seed, item[0]),
        ),
    )
    for fingerprint, group, labels in ordered:
        def score(fold: int) -> tuple[float, int, str]:
            error = 0.0
            for label in ("NORMAL", "SUSPICIOUS"):
                values = [fold_counts[index][label] for index in range(n_splits)]
                values[fold] += labels[label]
                error += sum(
                    ((value - target[label]) / max(target[label], 1)) ** 2 for value in values
                )
            sizes = [sum(fold_counts[index].values()) for index in range(n_splits)]
            sizes[fold] += len(group)
            error += 0.05 * sum(
                ((value - target_total) / max(target_total, 1)) ** 2 for value in sizes
            )
            return error, sum(fold_counts[fold].values()), _stable(seed, f"{fingerprint}:{fold}")

        destination = min(range(n_splits), key=score)
        fold_groups[destination].extend(group)
        fold_counts[destination].update(labels)

    fold_reports = []
    for fold, validation_counts in enumerate(fold_counts):
        train_counts = total_labels - validation_counts
        fold_reports.append(
            {
                "fold": fold,
                "validation": {label: validation_counts[label] for label in ("NORMAL", "SUSPICIOUS")},
                "train": {label: train_counts[label] for label in ("NORMAL", "SUSPICIOUS")},
                "validation_groups": 0,
            }
        )
    # Replace the row-derived expression above with the actual number of
    # connected components assigned to each fold, without exposing IDs.
    component_counts = [0] * n_splits
    for _, group, _ in ordered:
        for fold, values in enumerate(fold_groups):
            if group and group[0] in values:
                component_counts[fold] += 1
                break
    for fold, count in enumerate(component_counts):
        fold_reports[fold]["validation_groups"] = count

    fold_class_support = all(
        report["validation"]["NORMAL"] > 0 and report["validation"]["SUSPICIOUS"] > 0
        for report in fold_reports
    )
    if not fold_class_support:
        reasons.append("AT_LEAST_ONE_FOLD_MISSING_CLASS")
    return {
        "status": "FEASIBLE" if not reasons else "NOT_FEASIBLE",
        "requested_splits": n_splits,
        "train_only": True,
        "excluded_non_train_rows": dict(sorted(held_out.items())),
        "train_rows": len(train_rows),
        "train_label_support": dict(sorted(total_labels.items())),
        "connected_group_count": len(groups),
        "label_group_support": dict(sorted(label_group_support.items())),
        "largest_group_size": max(map(len, groups)),
        "folds": fold_reports,
        "blocking_reasons": sorted(set(reasons)),
        "sklearn_available": importlib.util.find_spec("sklearn") is not None,
        "recommended_splitter": "sklearn.model_selection.StratifiedGroupKFold" if not reasons else None,
        "fallback_planner": "deterministic connected-group stratification",
        "threshold_stability_plan": {
            "fit_on_each_fold_training_partition": True,
            "select_threshold_on_each_fold_validation_partition": True,
            "report": ["median", "IQR", "min", "max", "per-fold class support"],
            "external_validation_or_frozen_test_used": False,
        },
    }


def run(
    dataset_path: str | Path,
    *,
    verified_relations_path: str | Path | None = None,
    pilot_exclusions_path: str | Path | None = None,
    n_splits: int = 5,
    seed: int = 20261114,
) -> dict[str, Any]:
    dataset = load_dataset(
        dataset_path,
        exclusion_manifest=pilot_exclusions_path,
        verified_relations=verified_relations_path,
        strict_product_isolation=True,
        require_both_labels_per_split=True,
    )
    rows = []
    for split in ("train", "validation", "test"):
        for example, metadata in zip(dataset.examples[split], dataset.metadata[split], strict=True):
            rows.append(
                dict(metadata)
                | {
                    "content": example["text"],
                    "label": "SUSPICIOUS" if example["label"] else "NORMAL",
                    "split": split,
                }
            )
    relations = load_verified_relations(verified_relations_path)
    report = plan_group_kfold(rows, relations=relations, n_splits=n_splits, seed=seed)
    report["dataset_sha256"] = dataset.summary["sha256"]
    report["verified_relations_sha256"] = relations.source_sha256
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--verified-relations")
    parser.add_argument("--pilot-exclusions")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.dataset,
                verified_relations_path=args.verified_relations,
                pilot_exclusions_path=args.pilot_exclusions,
                n_splits=args.folds,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
