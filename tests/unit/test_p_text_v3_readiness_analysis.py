import pytest

from app.training.p_text_v3.readiness_analysis import metric_intervals, threshold_sensitivity, wilson_interval


def test_wilson_small_support_is_wide():
    interval = wilson_interval(3, 5)
    assert interval[0] < 0.25
    assert interval[1] > 0.88


def test_metric_intervals_use_correct_denominators():
    report = metric_intervals([[62, 0], [2, 3]])
    assert report["recall"] == wilson_interval(3, 5)
    assert report["precision"] == wilson_interval(3, 3)
    assert report["fpr"] == wilson_interval(0, 62)


def test_threshold_sensitivity_keeps_calibrated_and_raw_spaces_explicit():
    report = threshold_sensitivity([0, 0, 1, 1], [0.1, 0.4, 0.45, 0.8], 2.0)
    assert report["space"].startswith("calibrated_probability")
    assert report["operating_point"]["calibrated_threshold"] == 0.5
    assert report["minimum_fpr_to_reach_recall"]["1.0"] is not None


def test_wilson_rejects_invalid_support():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(2, 1) is None
