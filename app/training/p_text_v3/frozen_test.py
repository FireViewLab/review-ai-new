"""Evaluate frozen v2/v3 models once on the same pre-approved test split.

This module consumes already frozen validation policies.  It contains no
threshold fitting, calibration fitting, checkpoint selection, or training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .data_contract import load_dataset
from .evaluation import evaluate_probabilities
from .report import compare_frozen_test, render_markdown


def load_selection(path: str | Path) -> dict[str, float]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("fitted_on") != "validation_only":
        raise ValueError("SELECTION_NOT_VALIDATION_ONLY")
    threshold = value.get("threshold")
    temperature = value.get("temperature")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("SELECTION_THRESHOLD_INVALID")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("SELECTION_TEMPERATURE_INVALID")
    return {"threshold": float(threshold), "temperature": float(temperature)}


def predict_probabilities(model_path: str | Path, texts: list[str], *, max_length: int = 256, batch_size: int = 8) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    if model.config.id2label != {0: "NORMAL", 1: "SUSPICIOUS"}:
        raise ValueError("MODEL_LABEL_MAPPING_INVALID")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    probabilities: list[float] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            batch = torch.softmax(model(**encoded).logits, dim=-1)[:, 1].cpu().tolist()
        probabilities.extend(float(value) for value in batch)
    return probabilities


def evaluate_pair(
    labels: list[int],
    baseline_probabilities: list[float],
    candidate_probabilities: list[float],
    baseline_selection: Mapping[str, float],
    candidate_selection: Mapping[str, float],
) -> dict:
    baseline = evaluate_probabilities(
        labels,
        baseline_probabilities,
        threshold=float(baseline_selection["threshold"]),
        calibration_temperature=float(baseline_selection["temperature"]),
    )
    candidate = evaluate_probabilities(
        labels,
        candidate_probabilities,
        threshold=float(candidate_selection["threshold"]),
        calibration_temperature=float(candidate_selection["temperature"]),
    )
    return compare_frozen_test(baseline, candidate)


def run(
    dataset_path: str | Path,
    baseline_model: str | Path,
    candidate_model: str | Path,
    baseline_selection_path: str | Path,
    candidate_selection_path: str | Path,
    output_dir: str | Path,
    *,
    exclusion_manifest: str | Path | None = None,
    verified_relations: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")
    data = load_dataset(
        dataset_path,
        exclusion_manifest=exclusion_manifest,
        verified_relations=verified_relations,
        strict_product_isolation=True,
        require_both_labels_per_split=True,
    )
    baseline_selection = load_selection(baseline_selection_path)
    candidate_selection = load_selection(candidate_selection_path)
    texts = [str(row["text"]) for row in data.examples["test"]]
    labels = [int(row["label"]) for row in data.examples["test"]]
    baseline_probabilities = predict_probabilities(baseline_model, texts)
    candidate_probabilities = predict_probabilities(candidate_model, texts)
    report = evaluate_pair(
        labels,
        baseline_probabilities,
        candidate_probabilities,
        baseline_selection,
        candidate_selection,
    )
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    # Opaque metadata only: no review content is written to predictions.
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for metadata, label, baseline_probability, candidate_probability in zip(
            data.metadata["test"], labels, baseline_probabilities, candidate_probabilities, strict=True
        ):
            record = {
                "row_uid": metadata.get("row_uid"),
                "annotation_id": metadata.get("annotation_id"),
                "actual_label": "SUSPICIOUS" if label else "NORMAL",
                "baseline_raw_probability": baseline_probability,
                "candidate_raw_probability": candidate_probability,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-selection", required=True)
    parser.add_argument("--candidate-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot-exclusions")
    parser.add_argument("--verified-relations")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = run(
        args.dataset,
        args.baseline_model,
        args.candidate_model,
        args.baseline_selection,
        args.candidate_selection,
        args.output_dir,
        exclusion_manifest=args.pilot_exclusions,
        verified_relations=args.verified_relations,
    )
    print(json.dumps({"support": report["candidate"]["classification"]["support"]}, indent=2))


if __name__ == "__main__":
    main()
