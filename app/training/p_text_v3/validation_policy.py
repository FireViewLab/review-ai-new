"""Freeze threshold and calibration for an existing model using validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_contract import load_dataset
from .evaluation import select_validation_policy
from .frozen_test import predict_probabilities


def run(
    dataset_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    exclusion_manifest: str | Path | None = None,
    verified_relations: str | Path | None = None,
    max_normal_fpr: float = 0.05,
    calibrate: bool = True,
) -> dict:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")
    data = load_dataset(
        dataset_path,
        exclusion_manifest=exclusion_manifest,
        verified_relations=verified_relations,
        strict_product_isolation=True,
        require_both_labels_per_split=True,
    )
    labels = [int(row["label"]) for row in data.examples["validation"]]
    texts = [str(row["text"]) for row in data.examples["validation"]]
    probabilities = predict_probabilities(model_path, texts)
    policy = select_validation_policy(
        labels,
        probabilities,
        normal_mask=[label == 0 for label in labels],
        max_normal_fpr=max_normal_fpr,
        calibrate=calibrate,
    )
    policy["dataset_sha256"] = data.summary["sha256"]
    policy["model_path"] = str(Path(model_path).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return policy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-exclusions")
    parser.add_argument("--verified-relations")
    parser.add_argument("--max-normal-fpr", type=float, default=0.05)
    parser.add_argument("--no-calibration", action="store_false", dest="calibrate")
    parser.set_defaults(calibrate=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    policy = run(
        args.dataset,
        args.model,
        args.output,
        exclusion_manifest=args.pilot_exclusions,
        verified_relations=args.verified_relations,
        max_normal_fpr=args.max_normal_fpr,
        calibrate=args.calibrate,
    )
    print(json.dumps(policy, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
