"""Validation-only diagnostics for a P_text candidate; never evaluates Frozen Test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

from .data_contract import load_dataset
from .evaluation import binary_metrics, calibrated_threshold_to_raw, temperature_scale
from .frozen_test import predict_probabilities


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float] | None:
    if trials <= 0 or not 0 <= successes <= trials:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def metric_intervals(matrix: Sequence[Sequence[int]]) -> dict:
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {
        "method": "Wilson score interval, 95%",
        "recall": wilson_interval(tp, tp + fn),
        "precision": wilson_interval(tp, tp + fp),
        "fpr": wilson_interval(fp, fp + tn),
    }


def stratified_bootstrap(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float, *, samples: int = 10000, seed: int = 20261115
) -> dict:
    positive = [index for index, label in enumerate(labels) if label == 1]
    negative = [index for index, label in enumerate(labels) if label == 0]
    if not positive or not negative:
        raise ValueError("BOOTSTRAP_CLASS_MISSING")
    rng = random.Random(seed)
    values = {name: [] for name in ("precision", "recall", "f1", "fpr")}
    for _ in range(samples):
        indices = [rng.choice(positive) for _ in positive] + [rng.choice(negative) for _ in negative]
        sampled_labels = [labels[index] for index in indices]
        sampled_probabilities = [probabilities[index] for index in indices]
        result = binary_metrics(sampled_labels, sampled_probabilities, threshold)
        tn, fp = result["confusion_matrix"][0]
        for name in ("precision", "recall", "f1"):
            values[name].append(float(result[name]))
        values["fpr"].append(fp / (tn + fp))

    def summary(items: list[float]) -> dict:
        ordered = sorted(items)
        return {
            "mean": mean(items),
            "std": pstdev(items),
            "percentile_95": [ordered[int(0.025 * samples)], ordered[min(samples - 1, int(0.975 * samples))]],
            "unique_values": len(set(items)),
        }

    return {
        "method": "stratified nonparametric bootstrap; class counts fixed",
        "samples": samples,
        "seed": seed,
        "warning": "Only five positive rows; bootstrap is discrete and does not establish generalization.",
        "metrics": {name: summary(items) for name, items in values.items()},
    }


def threshold_sensitivity(labels: list[int], raw_probabilities: list[float], temperature: float) -> dict:
    calibrated = temperature_scale(raw_probabilities, temperature)
    thresholds = sorted({0.0, 0.5, 1.0, *calibrated}, reverse=True)
    rows = []
    for threshold in thresholds:
        result = binary_metrics(labels, calibrated, threshold)
        tn, fp = result["confusion_matrix"][0]
        rows.append(
            {
                "calibrated_threshold": threshold,
                "equivalent_raw_threshold": calibrated_threshold_to_raw(threshold, temperature),
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "fpr": fp / (tn + fp),
                "confusion_matrix": result["confusion_matrix"],
            }
        )
    feasible = [row for row in rows if row["fpr"] <= 0.05]
    by_recall = {}
    for target in (0.8, 1.0):
        candidates = [row for row in rows if row["recall"] >= target]
        by_recall[str(target)] = min(candidates, key=lambda row: (row["fpr"], -row["calibrated_threshold"])) if candidates else None
    return {
        "space": "calibrated_probability with explicitly equivalent raw threshold",
        "temperature": temperature,
        "max_normal_fpr": 0.05,
        "operating_point": next(row for row in rows if row["calibrated_threshold"] == 0.5),
        "best_f1_under_fpr_cap": max(feasible, key=lambda row: (row["f1"], row["recall"], row["calibrated_threshold"])),
        "minimum_fpr_to_reach_recall": by_recall,
        "curve": rows,
        "warning": "Sensitivity analysis on reused Validation only; no Runtime setting was changed.",
    }


def _load_staging(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["annotation_id"]: row for row in csv.DictReader(handle)}


def run(args) -> dict:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")
    data = load_dataset(
        args.dataset,
        exclusion_manifest=args.pilot_exclusions,
        verified_relations=args.verified_relations,
        strict_product_isolation=True,
        require_both_labels_per_split=True,
    )
    texts = [row["text"] for row in data.examples["validation"]]
    labels = [int(row["label"]) for row in data.examples["validation"]]
    metadata = data.metadata["validation"]
    models = {
        "baseline": args.baseline_model,
        "v3_epoch1": args.v3_epoch1_model,
        "v3_1_b": args.candidate_model,
    }
    probabilities = {name: predict_probabilities(path, texts, batch_size=8) for name, path in models.items()}
    temperature = float(args.temperature)
    calibrated = temperature_scale(probabilities["v3_1_b"], temperature)
    staging = _load_staging(Path(args.staging))
    false_negatives = []
    for index, (label, probability) in enumerate(zip(labels, calibrated, strict=True)):
        if label != 1 or probability >= args.calibrated_threshold:
            continue
        annotation_id = str(metadata[index]["annotation_id"])
        evidence = staging.get(annotation_id, {})
        rationale_codes = evidence.get("rationale_codes", "")
        false_negatives.append(
            {
                "annotation_id": annotation_id,
                "row_uid": metadata[index].get("row_uid"),
                "content": texts[index],
                "actual_label": "SUSPICIOUS",
                "prediction": "NORMAL",
                "probabilities": {
                    name: {
                        "raw": values[index],
                        "candidate_temperature_calibrated": temperature_scale([values[index]], temperature)[0],
                    }
                    for name, values in probabilities.items()
                },
                "threshold": {
                    "calibrated": args.calibrated_threshold,
                    "raw": calibrated_threshold_to_raw(args.calibrated_threshold, temperature),
                    "calibrated_margin": probability - args.calibrated_threshold,
                    "raw_margin": probabilities["v3_1_b"][index]
                    - calibrated_threshold_to_raw(args.calibrated_threshold, temperature),
                },
                "guide_evidence": {
                    "source_status": "AI_ASSISTED_CONFIRMED_DRAFT_NOT_INDEPENDENT_GROUND_TRUTH",
                    "text_status": evidence.get("text_status", ""),
                    "signal_decision": evidence.get("signal_decision", ""),
                    "signal_types": evidence.get("signal_types", ""),
                    "rationale_codes": rationale_codes,
                    "rationale_note": evidence.get("rationale_note", ""),
                    "ambiguity_indicator": bool("BOUNDARY" in rationale_codes or evidence.get("other_review_needed") == "YES"),
                },
            }
        )
    if len(false_negatives) != 2:
        raise ValueError(f"UNEXPECTED_FALSE_NEGATIVE_COUNT:{len(false_negatives)}")

    sensitivity = threshold_sensitivity(labels, probabilities["v3_1_b"], temperature)
    operating = sensitivity["operating_point"]
    uncertainty = {
        "support": {"NORMAL": labels.count(0), "SUSPICIOUS": labels.count(1)},
        "operating_point": operating,
        "wilson_95": metric_intervals(operating["confusion_matrix"]),
        "bootstrap": stratified_bootstrap(labels, calibrated, args.calibrated_threshold),
        "paired_prediction_differences": {
            name: {
                "mean_raw_probability_delta_vs_v3_1_b": mean(
                    left - right for left, right in zip(probabilities[name], probabilities["v3_1_b"], strict=True)
                ),
                "changed_raw_0_5_decisions": sum(
                    (left >= 0.5) != (right >= 0.5)
                    for left, right in zip(probabilities[name], probabilities["v3_1_b"], strict=True)
                ),
            }
            for name in ("baseline", "v3_epoch1")
        },
    }
    output.mkdir(parents=True)
    (output / "false_negative_audit.json").write_text(
        json.dumps(
            {
                "privacy": "PRIVATE_LOCAL_CONTAINS_REVIEW_TEXT_AND_OPAQUE_IDENTIFIERS",
                "dataset_sha256": data.summary["sha256"],
                "frozen_test_evaluated": False,
                "validation_extension_194_used": False,
                "false_negatives": false_negatives,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "threshold_sensitivity.json").write_text(json.dumps(sensitivity, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "uncertainty_metrics.json").write_text(json.dumps(uncertainty, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"false_negative_count": 2, "sensitivity": sensitivity, "uncertainty": uncertainty}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--v3-epoch1-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--pilot-exclusions", required=True)
    parser.add_argument("--verified-relations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--calibrated-threshold", type=float, required=True)
    return parser.parse_args(argv)


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"false_negative_count": result["false_negative_count"], "frozen_test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
