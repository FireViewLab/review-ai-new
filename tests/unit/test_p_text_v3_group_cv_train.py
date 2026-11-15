from app.training.p_text_v3.data_contract import VerifiedRelations
from app.training.p_text_v3.group_cv_train import aggregate, assign_folds


def _row(uid, product, label):
    return {"row_uid": uid, "platform": "P", "product_id": product, "content": uid, "label": label, "split": "train"}


def test_assign_folds_keeps_product_group_together_and_is_deterministic():
    rows = [_row("a", "same", "NORMAL"), _row("b", "same", "SUSPICIOUS"), _row("c", "c", "NORMAL"), _row("d", "d", "SUSPICIOUS")]
    one = assign_folds(rows, VerifiedRelations({}, None, 0), n_splits=2, seed=7)
    two = assign_folds(rows, VerifiedRelations({}, None, 0), n_splits=2, seed=7)
    assert [[row["row_uid"] for row in fold] for fold in one] == [[row["row_uid"] for row in fold] for fold in two]
    assert any({row["row_uid"] for row in fold} >= {"a", "b"} for fold in one)


def test_aggregate_reports_population_variation():
    rows = [
        {"precision": 1, "recall": 0.5, "f1": 2 / 3, "pr_auc": 0.8, "fpr": 0},
        {"precision": 0.5, "recall": 1, "f1": 2 / 3, "pr_auc": 0.6, "fpr": 0.1},
    ]
    result = aggregate(rows)
    assert result["recall"]["mean"] == 0.75
    assert result["fpr"]["std"] == 0.05
