import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.training.p_text_v3.evaluation import (
    average_precision,
    binary_metrics,
    calibrated_threshold_to_raw,
    evaluate_probabilities,
    score_distribution,
    select_validation_policy,
    temperature_scale_probability,
)
from app.training.p_text_v3.frozen_test import evaluate_pair, load_selection
from app.training.p_text_v3.train import build_runtime_inference_config, parse_args
from app.training.p_text_v3.validation_policy import run as freeze_validation_policy
from app.analyzers import p_text


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
    assert policy["threshold_space"] == "calibrated_probability"
    assert policy["runtime_threshold_space"] == "raw_probability"
    assert policy["runtime_raw_probability_threshold"] == policy["threshold"]


@pytest.mark.parametrize("temperature", [0.25, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("threshold", [1e-9, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-9])
def test_calibrated_threshold_to_raw_preserves_decisions(temperature, threshold):
    raw_threshold = calibrated_threshold_to_raw(threshold, temperature)
    lower = raw_threshold * 0.5
    upper = raw_threshold + (1 - raw_threshold) * 0.5
    assert temperature_scale_probability(lower, temperature) < threshold
    assert lower < raw_threshold
    assert temperature_scale_probability(upper, temperature) >= threshold
    assert upper >= raw_threshold


@pytest.mark.parametrize("threshold", [0.0, 1e-12, 0.1, 0.5, 0.9, 1 - 1e-12, 1.0])
def test_temperature_one_keeps_threshold_unchanged(threshold):
    assert calibrated_threshold_to_raw(threshold, 1.0) == pytest.approx(threshold, abs=1e-15)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
@pytest.mark.parametrize("temperature", [0.1, 1.0, 10.0])
def test_threshold_boundary_values_are_preserved(threshold, temperature):
    assert calibrated_threshold_to_raw(threshold, temperature) == threshold


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_calibrated_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="THRESHOLD_INVALID"):
        calibrated_threshold_to_raw(threshold, 1.0)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True, "2", None])
def test_invalid_calibration_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="TEMPERATURE_INVALID"):
        calibrated_threshold_to_raw(0.5, temperature)


def test_runtime_inference_config_uses_equivalent_raw_threshold_without_runtime_change():
    selection = {
        "threshold": 0.8,
        "temperature": 2.0,
        "threshold_space": "calibrated_probability",
        "runtime_raw_probability_threshold": calibrated_threshold_to_raw(0.8, 2.0),
    }
    config = build_runtime_inference_config(selection, max_length=256, dataset_sha256="fixture-sha")
    assert config["threshold_space"] == "raw_probability"
    assert config["threshold"] == config["raw_probability_threshold"]
    assert config["validation_calibrated_probability_threshold"] == 0.8
    assert config["calibration_temperature"] == 2.0
    for probability in (0.1, config["threshold"] * 0.99, min(1.0, config["threshold"] * 1.01), 1.0):
        expected = "SUSPICIOUS" if temperature_scale_probability(probability, 2.0) >= 0.8 else "NORMAL"
        assert p_text.probability_result(probability, config["threshold"])["predicted_label"] == expected


def test_runtime_inference_config_rejects_mismatched_recorded_raw_threshold():
    with pytest.raises(ValueError, match="RUNTIME_RAW_THRESHOLD_MISMATCH"):
        build_runtime_inference_config(
            {
                "threshold": 0.8,
                "temperature": 2.0,
                "threshold_space": "calibrated_probability",
                "runtime_raw_probability_threshold": 0.8,
            },
            max_length=256,
            dataset_sha256="fixture-sha",
        )


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


def test_frozen_selection_rejects_threshold_space_or_conversion_mismatch(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({
        "fitted_on": "validation_only", "threshold": 0.8, "temperature": 2.0,
        "threshold_space": "raw_probability", "runtime_raw_probability_threshold": 0.8,
    }))
    with pytest.raises(ValueError, match="THRESHOLD_SPACE_INVALID"):
        load_selection(path)
    path.write_text(json.dumps({
        "fitted_on": "validation_only", "threshold": 0.8, "temperature": 2.0,
        "threshold_space": "calibrated_probability", "runtime_raw_probability_threshold": 0.8,
    }))
    with pytest.raises(ValueError, match="RAW_THRESHOLD_MISMATCH"):
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
