from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.training.p_text_v3.train import (
    Config,
    balanced_class_weights,
    make_weighted_trainer,
    parse_args,
    trainer_selection_policy,
    weighted_cross_entropy,
)


def test_balanced_class_weights_use_train_distribution_formula():
    labels = [0] * 109 + [1] * 35
    weights = balanced_class_weights(labels)
    assert weights == pytest.approx([144 / (2 * 109), 144 / (2 * 35)])
    assert weights[1] > weights[0]


@pytest.mark.parametrize("labels", [[], [0, 0], [1, 1], [0, 2]])
def test_balanced_class_weights_fail_closed(labels):
    with pytest.raises(ValueError, match="CLASS_WEIGHT"):
        balanced_class_weights(labels)


def test_weighted_cross_entropy_increases_minority_error_cost():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, -2.0], [2.0, -2.0], [2.0, -2.0], [2.0, -2.0]])
    labels = torch.tensor([0, 0, 0, 1])
    unweighted = torch.nn.functional.cross_entropy(logits, labels)
    weighted = weighted_cross_entropy(logits, labels, [0.5, 2.0])
    assert weighted > unweighted


def test_weighted_trainer_uses_custom_loss_and_returns_outputs():
    torch = pytest.importorskip("torch")

    class BaseTrainer:
        pass

    class Model:
        def __call__(self, **inputs):
            return SimpleNamespace(logits=torch.tensor([[1.0, -1.0], [-1.0, 1.0]], requires_grad=True))

    trainer = make_weighted_trainer(BaseTrainer, [0.5, 2.0])()
    labels = torch.tensor([0, 1])
    loss, outputs = trainer.compute_loss(Model(), {"labels": labels}, return_outputs=True)
    assert loss.item() == pytest.approx(weighted_cross_entropy(outputs.logits, labels, [0.5, 2.0]).item())
    assert loss.requires_grad


def test_pr_auc_checkpoint_policy_and_cli_flags_are_explicit():
    config = parse_args(["--best-model-metric", "pr_auc", "--class-weighting"])
    assert config.best_model_metric == "pr_auc"
    assert config.class_weighting is True
    assert trainer_selection_policy(config) == {
        "metric_for_best_model": "pr_auc",
        "greater_is_better": True,
        "load_best_model_at_end": True,
    }


def test_default_training_behavior_remains_unweighted_f1():
    config = Config()
    assert config.best_model_metric == "f1"
    assert config.class_weighting is False
    assert trainer_selection_policy(config)["metric_for_best_model"] == "f1"
