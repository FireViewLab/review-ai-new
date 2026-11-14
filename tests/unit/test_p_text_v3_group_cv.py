from __future__ import annotations

from app.training.p_text_v3.data_contract import VerifiedRelations
from app.training.p_text_v3.group_cv import plan_group_kfold


def row(index: int, label: str, *, split: str = "train", product: str | None = None):
    return {
        "row_uid": f"row-{index}",
        "platform": "fixture",
        "product_id": product or f"product-{index}",
        "content": f"synthetic cv content {index}",
        "duplicate_cluster_id": "",
        "label": label,
        "split": split,
    }


def test_five_fold_plan_uses_train_only_and_reports_each_fold_support():
    rows = [row(index, "SUSPICIOUS") for index in range(35)]
    rows += [row(100 + index, "NORMAL") for index in range(109)]
    rows += [row(300 + index, "SUSPICIOUS", split="validation") for index in range(5)]
    rows += [row(400 + index, "NORMAL", split="test") for index in range(60)]
    report = plan_group_kfold(rows, n_splits=5)
    assert report["status"] == "FEASIBLE"
    assert report["train_rows"] == 144
    assert report["train_label_support"] == {"NORMAL": 109, "SUSPICIOUS": 35}
    assert report["excluded_non_train_rows"] == {"test": 60, "validation": 5}
    assert all(fold["validation"]["SUSPICIOUS"] > 0 for fold in report["folds"])
    assert all(fold["validation"]["NORMAL"] > 0 for fold in report["folds"])
    assert report["threshold_stability_plan"]["external_validation_or_frozen_test_used"] is False


def test_product_cluster_and_verified_relation_form_connected_groups():
    rows = [row(index, "SUSPICIOUS" if index < 10 else "NORMAL") for index in range(30)]
    rows[1]["product_id"] = rows[0]["product_id"]
    rows[2]["duplicate_cluster_id"] = "cluster"
    rows[3]["duplicate_cluster_id"] = "cluster"
    relations = VerifiedRelations(
        {"row-4": (("NEAR_DUPLICATE", "near"),), "row-5": (("NEAR_DUPLICATE", "near"),)},
        "fixture",
        2,
    )
    report = plan_group_kfold(rows, relations=relations, n_splits=5)
    assert report["connected_group_count"] == 27
    assert report["largest_group_size"] == 2


def test_blank_excel_cluster_values_do_not_form_a_false_group():
    rows = [row(index, "SUSPICIOUS" if index < 5 else "NORMAL") for index in range(15)]
    for value in rows:
        value["duplicate_cluster_id"] = None
    report = plan_group_kfold(rows, n_splits=5)
    assert report["connected_group_count"] == 15
    assert report["largest_group_size"] == 1


def test_insufficient_positive_groups_blocks_requested_fold_count():
    rows = [row(index, "SUSPICIOUS", product="positive-shared") for index in range(5)]
    rows += [row(100 + index, "NORMAL") for index in range(20)]
    report = plan_group_kfold(rows, n_splits=5)
    assert report["status"] == "NOT_FEASIBLE"
    assert "SUSPICIOUS_GROUP_SUPPORT_LT_N_SPLITS" in report["blocking_reasons"]
    assert report["recommended_splitter"] is None
