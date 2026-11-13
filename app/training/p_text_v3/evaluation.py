"""Binary metrics, validation-only selection, and P_text distribution audits."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _validate(labels: Sequence[int], probabilities: Sequence[float]) -> None:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("EVALUATION_INPUT_LENGTH_INVALID")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("EVALUATION_LABEL_INVALID")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("EVALUATION_PROBABILITY_INVALID")


def temperature_scale_probability(probability: float, temperature: float) -> float:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("TEMPERATURE_INVALID")
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 1.0
    logit = math.log(probability / (1 - probability))
    return 1 / (1 + math.exp(-logit / temperature))


def temperature_scale(probabilities: Sequence[float], temperature: float) -> list[float]:
    return [temperature_scale_probability(value, temperature) for value in probabilities]


def confusion(labels: Sequence[int], probabilities: Sequence[float], threshold: float) -> list[list[int]]:
    _validate(labels, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("THRESHOLD_INVALID")
    tn = fp = fn = tp = 0
    for label, probability in zip(labels, probabilities, strict=True):
        predicted = int(probability >= threshold)
        if label == 0 and predicted == 0:
            tn += 1
        elif label == 0:
            fp += 1
        elif predicted == 0:
            fn += 1
        else:
            tp += 1
    return [[tn, fp], [fn, tp]]


def average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    """Average precision, i.e. step-wise area under the precision-recall curve."""
    _validate(labels, probabilities)
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(probabilities, labels, strict=True), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    _validate(labels, probabilities)
    return sum((probability - label) ** 2 for label, probability in zip(labels, probabilities, strict=True)) / len(labels)


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> dict:
    matrix = confusion(labels, probabilities, threshold)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support": len(labels),
        "positive_support": sum(labels),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": average_precision(labels, probabilities),
        "brier_score": brier_score(labels, probabilities),
        "confusion_matrix": matrix,
    }


def ptext_scores(probabilities: Sequence[float], temperature: float = 1.0) -> list[int]:
    scaled = temperature_scale(probabilities, temperature)
    return [round((1 - value) * 100) for value in scaled]


def score_distribution(scores: Sequence[int]) -> dict:
    if not scores or any(type(value) is not int or not 0 <= value <= 100 for value in scores):
        raise ValueError("PTEXT_SCORE_INVALID")
    ordered = sorted(scores)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    histogram = {f"{lower:03d}-{lower + 9:03d}": 0 for lower in range(0, 100, 10)}
    histogram["100"] = 0
    for score in scores:
        if score == 100:
            histogram["100"] += 1
        else:
            lower = (score // 10) * 10
            histogram[f"{lower:03d}-{lower + 9:03d}"] += 1
    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / len(scores)
    return {
        "count": len(scores),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(scores),
        "max": max(scores),
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "unique_count": len(set(scores)),
        "le_1_ratio": sum(value <= 1 for value in scores) / len(scores),
        "ge_99_ratio": sum(value >= 99 for value in scores) / len(scores),
        "histogram": histogram,
    }


def distribution_by_label(labels: Sequence[int], scores: Sequence[int]) -> dict[str, dict]:
    if len(labels) != len(scores):
        raise ValueError("LABEL_SCORE_LENGTH_INVALID")
    return {
        name: score_distribution([score for label, score in zip(labels, scores, strict=True) if label == value])
        for name, value in (("NORMAL", 0), ("SUSPICIOUS", 1))
        if any(label == value for label in labels)
    }


def _negative_log_likelihood(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    epsilon = 1e-12
    return -sum(
        label * math.log(min(max(probability, epsilon), 1 - epsilon))
        + (1 - label) * math.log(min(max(1 - probability, epsilon), 1 - epsilon))
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)


def fit_temperature(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    """Fit one temperature on validation only using a deterministic log-grid search."""
    _validate(labels, probabilities)
    candidates = [10 ** (-1 + index * 2 / 200) for index in range(201)]
    best_temperature = min(
        candidates,
        key=lambda value: (_negative_log_likelihood(labels, temperature_scale(probabilities, value)), abs(value - 1)),
    )
    # Refine around the best coarse point without scipy.
    lower, upper = best_temperature / 1.08, best_temperature * 1.08
    refined = [lower + (upper - lower) * index / 200 for index in range(201)]
    best_temperature = min(
        refined,
        key=lambda value: (_negative_log_likelihood(labels, temperature_scale(probabilities, value)), abs(value - 1)),
    )
    calibrated = temperature_scale(probabilities, best_temperature)
    return {
        "temperature": best_temperature,
        "validation_nll": _negative_log_likelihood(labels, calibrated),
        "validation_brier": brier_score(labels, calibrated),
    }


def select_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    normal_mask: Sequence[bool] | None = None,
    max_normal_fpr: float = 0.05,
) -> dict:
    """Select a threshold using validation labels only.

    Feasible thresholds satisfy the declared NORMAL false-positive-rate cap.
    Within that set the positive F1, recall, lower FPR, and proximity to 0.5
    determine the stable ordering.
    """
    _validate(labels, probabilities)
    if not 0 <= max_normal_fpr <= 1:
        raise ValueError("MAX_NORMAL_FPR_INVALID")
    if normal_mask is None:
        normal_mask = [label == 0 for label in labels]
    if len(normal_mask) != len(labels) or not any(normal_mask):
        raise ValueError("NORMAL_MASK_INVALID")
    thresholds = sorted({0.0, 0.5, 1.0, *probabilities})
    candidates = []
    for threshold in thresholds:
        metrics = binary_metrics(labels, probabilities, threshold)
        normal_indices = [index for index, selected in enumerate(normal_mask) if selected]
        false_positives = sum(probabilities[index] >= threshold for index in normal_indices)
        normal_fpr = false_positives / len(normal_indices)
        candidates.append({"threshold": threshold, "normal_fpr": normal_fpr, "metrics": metrics})
    feasible = [item for item in candidates if item["normal_fpr"] <= max_normal_fpr]
    pool = feasible or candidates
    selected = max(
        pool,
        key=lambda item: (
            item["metrics"]["f1"],
            item["metrics"]["recall"],
            -item["normal_fpr"],
            -abs(item["threshold"] - 0.5),
        ),
    )
    return {
        "threshold": selected["threshold"],
        "constraint_met": bool(feasible),
        "max_normal_fpr": max_normal_fpr,
        "validation_normal_fpr": selected["normal_fpr"],
        "validation_metrics": selected["metrics"],
        "selected_on": "validation_only",
    }


def select_validation_policy(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    normal_mask: Sequence[bool] | None = None,
    max_normal_fpr: float = 0.05,
    calibrate: bool = True,
) -> dict:
    calibration = fit_temperature(labels, probabilities) if calibrate else {
        "temperature": 1.0,
        "validation_nll": _negative_log_likelihood(labels, probabilities),
        "validation_brier": brier_score(labels, probabilities),
    }
    calibrated = temperature_scale(probabilities, calibration["temperature"])
    threshold = select_threshold(
        labels, calibrated, normal_mask=normal_mask, max_normal_fpr=max_normal_fpr
    )
    return {
        "policy_version": "ptext-v3-validation-selection-v1",
        "fitted_on": "validation_only",
        "temperature": calibration["temperature"],
        "threshold": threshold["threshold"],
        "calibration": calibration,
        "threshold_selection": threshold,
        "runtime_temperature_2_is_display_only": True,
    }


def evaluate_probabilities(
    labels: Sequence[int],
    raw_probabilities: Sequence[float],
    *,
    threshold: float,
    calibration_temperature: float,
) -> dict:
    calibrated = temperature_scale(raw_probabilities, calibration_temperature)
    raw_scores = ptext_scores(raw_probabilities, 1.0)
    calibrated_scores = ptext_scores(raw_probabilities, calibration_temperature)
    runtime_temperature_2_scores = ptext_scores(raw_probabilities, 2.0)
    return {
        "classification": binary_metrics(labels, calibrated, threshold),
        "raw_probability_brier": brier_score(labels, raw_probabilities),
        "calibrated_probability_brier": brier_score(labels, calibrated),
        "raw_ptext": {
            "distribution": score_distribution(raw_scores),
            "by_label": distribution_by_label(labels, raw_scores),
        },
        "calibrated_ptext": {
            "distribution": score_distribution(calibrated_scores),
            "by_label": distribution_by_label(labels, calibrated_scores),
        },
        "runtime_temperature_2_ptext": {
            "interpretation": "display-only parity with current Runtime; not the raw model probability",
            "distribution": score_distribution(runtime_temperature_2_scores),
            "by_label": distribution_by_label(labels, runtime_temperature_2_scores),
        },
    }
