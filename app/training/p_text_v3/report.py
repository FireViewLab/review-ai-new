"""JSON-safe comparison report helpers for a shared frozen test."""

from __future__ import annotations

from typing import Mapping


def compare_frozen_test(baseline: Mapping, candidate: Mapping) -> dict:
    left = baseline["classification"]
    right = candidate["classification"]
    return {
        "comparison_policy": "same frozen test; no threshold, calibration, or model selection on test",
        "support_equal": left["support"] == right["support"],
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            metric: right[metric] - left[metric]
            for metric in ("precision", "recall", "f1", "pr_auc", "brier_score")
            if left.get(metric) is not None and right.get(metric) is not None
        },
    }


def render_markdown(report: Mapping) -> str:
    left = report["baseline"]["classification"]
    right = report["candidate"]["classification"]
    lines = [
        "# P_text v2/v3 frozen-test comparison",
        "",
        report["comparison_policy"],
        "",
        "| Model | Precision | Recall | F1 | PR-AUC | Brier | Confusion matrix |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, values in (("v2 baseline", left), ("v3 candidate", right)):
        lines.append(
            f"| {name} | {values['precision']:.6f} | {values['recall']:.6f} | "
            f"{values['f1']:.6f} | {values['pr_auc']:.6f} | {values['brier_score']:.6f} | "
            f"`{values['confusion_matrix']}` |"
        )
    lines += [
        "",
        "Raw P_text uses uncalibrated model probability. The separate runtime-temperature-2 "
        "distribution is display parity only and is not used to select either model.",
        "",
    ]
    return "\n".join(lines)
