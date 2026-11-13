import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.training.p_text_v3.evaluation import (
    average_precision,
    binary_metrics,
    evaluate_probabilities,
    score_distribution,
    select_validation_policy,
    temperature_scale_probability,
)
from app.training.p_text_v3.frozen_test import evaluate_pair, load_selection
from app.training.p_text_v3.train import parse_args
from app.training.p_text_v3.validation_policy import run as freeze_validation_policy


def test_precision_recall_f1_pr_auc_confusion_and_brier():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.8, 0.7, 0.9]
    metrics = binary_metrics(labels, probabilities, 0.5)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["brier_score"] == pytest.approx((0.01 + 0.64 + 0.09 + 0.01) / 4)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_validation_policy_records_validation_only_and_honors_normal_fpr():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.35, 0.9]
    policy = select_validation_policy(labels, probabilities, max_normal_fpr=0, calibrate=False)
    assert policy["fitted_on"] == "validation_only"
    assert policy["threshold_selection"]["selected_on"] == "validation_only"
    assert policy["threshold_selection"]["validation_normal_fpr"] == 0
    assert policy["temperature"] == 1


def test_raw_calibrated_and_runtime_temperature_2_scores_are_separate():
    report = evaluate_probabilities(
        [0, 1], [0.002, 0.8], threshold=0.5, calibration_temperature=1.5
    )
    assert report["raw_ptext"]["distribution"] != report["calibrated_ptext"]["distribution"]
    assert "display-only" in report["runtime_temperature_2_ptext"]["interpretation"]
    assert temperature_scale_probability(0.002, 2) > 0.002


def test_score_distribution_has_histogram_and_saturation_ratios():
    result = score_distribution([0, 1, 50, 99, 100])
    assert result["histogram"]["000-009"] == 2
    assert result["histogram"]["050-059"] == 1
    assert result["histogram"]["100"] == 1
    assert result["le_1_ratio"] == 0.4
    assert result["ge_99_ratio"] == 0.4


def test_frozen_pair_uses_supplied_policies_without_selection():
    report = evaluate_pair(
        [0, 0, 1, 1],
        [0.1, 0.6, 0.7, 0.8],
        [0.1, 0.2, 0.8, 0.9],
        {"threshold": 0.5, "temperature": 1.0},
        {"threshold": 0.7, "temperature": 1.0},
    )
    assert report["comparison_policy"].startswith("same frozen test")
    assert report["candidate"]["classification"]["f1"] == 1
    assert report["delta"]["f1"] > 0


def test_frozen_selection_rejects_test_fitted_policy(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"fitted_on": "test", "threshold": 0.5, "temperature": 1.0}))
    with pytest.raises(ValueError, match="VALIDATION_ONLY"):
        load_selection(path)


def test_low_memory_defaults_and_no_implicit_download():
    config = parse_args([])
    assert config.batch_size == 2
    assert config.gradient_accumulation_steps == 4
    assert config.gradient_checkpointing
    assert config.cpu_threads == 1
    assert config.dataloader_workers == 0
    assert config.local_files_only
    assert config.batch_size * config.gradient_accumulation_steps == 8


def test_existing_model_policy_uses_validation_split_only(tmp_path):
    data = SimpleNamespace(
        examples={
            "train": [{"text": "train fixture", "label": 0}],
            "validation": [
                {"text": "validation normal", "label": 0},
                {"text": "validation suspicious", "label": 1},
            ],
            "test": [{"text": "must not be read", "label": 1}],
        },
        summary={"sha256": "dataset-hash"},
    )
    output = tmp_path / "baseline-selection.json"
    with patch("app.training.p_text_v3.validation_policy.load_dataset", return_value=data), patch(
        "app.training.p_text_v3.validation_policy.predict_probabilities", return_value=[0.1, 0.9]
    ) as predict:
        policy = freeze_validation_policy("fixture.xlsx", tmp_path / "model", output, calibrate=False)
    predict.assert_called_once_with(tmp_path / "model", ["validation normal", "validation suspicious"])
    assert policy["fitted_on"] == "validation_only"
    assert output.is_file()
