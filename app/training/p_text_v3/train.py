"""Low-memory KoELECTRA trainer for an approved fixed-split v3 workbook.

This entry point trains and selects calibration/threshold on validation only.
It never predicts the frozen test set.  Use ``frozen_test.py`` after the model
and validation policy have been frozen and separately approved.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

from .data_contract import TokenizedTextDataset, load_dataset
from .evaluation import (
    binary_metrics,
    calibrated_threshold_to_raw,
    select_validation_policy,
)


@dataclass(frozen=True)
class Config:
    data_path: str = "data/p_text_v3/dataset.xlsx"
    output_dir: str = "artifacts/p_text_v3/run"
    model_name: str = "monologg/koelectra-base-v3-discriminator"
    exclusion_manifest: str | None = None
    verified_relations: str | None = None
    seed: int = 42
    max_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 2
    cpu_threads: int = 1
    dataloader_workers: int = 0
    max_normal_fpr: float = 0.05
    gradient_checkpointing: bool = True
    local_files_only: bool = True
    calibrate: bool = True


def parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=Config.data_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--model-name", default=Config.model_name)
    parser.add_argument("--pilot-exclusions", dest="exclusion_manifest")
    parser.add_argument("--verified-relations")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--max-length", type=int, default=Config.max_length)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=Config.gradient_accumulation_steps)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--cpu-threads", type=int, default=Config.cpu_threads)
    parser.add_argument("--dataloader-workers", type=int, default=Config.dataloader_workers)
    parser.add_argument("--max-normal-fpr", type=float, default=Config.max_normal_fpr)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--allow-model-download", action="store_false", dest="local_files_only")
    parser.add_argument("--no-calibration", action="store_false", dest="calibrate")
    parser.set_defaults(
        gradient_checkpointing=Config.gradient_checkpointing,
        local_files_only=Config.local_files_only,
        calibrate=Config.calibrate,
    )
    config = Config(**vars(parser.parse_args(argv)))
    positive = (
        config.max_length,
        config.batch_size,
        config.gradient_accumulation_steps,
        config.epochs,
        config.cpu_threads,
    )
    if any(value < 1 for value in positive) or config.dataloader_workers < 0:
        parser.error("length, batch, accumulation, epochs, and threads must be positive; workers must be non-negative")
    if config.max_length > 512 or config.learning_rate <= 0 or config.weight_decay < 0:
        parser.error("invalid max length, learning rate, or weight decay")
    if not 0 <= config.max_normal_fpr <= 1:
        parser.error("max-normal-fpr must be in [0, 1]")
    return config


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _softmax_positive(logits) -> list[float]:
    import numpy as np

    values = np.asarray(logits, dtype=float)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials[:, 1] / exponentials.sum(axis=1)).tolist()


def build_runtime_inference_config(
    selection: dict[str, Any], *, max_length: int, dataset_sha256: str
) -> dict[str, Any]:
    """Package a validation policy for the unchanged raw-threshold Runtime."""
    if type(max_length) is not int or not 1 <= max_length <= 512:
        raise ValueError("MAX_LENGTH_INVALID")
    if not isinstance(dataset_sha256, str) or not dataset_sha256:
        raise ValueError("DATASET_SHA256_INVALID")
    calibrated_threshold = selection.get("threshold")
    temperature = selection.get("temperature")
    raw_threshold = calibrated_threshold_to_raw(calibrated_threshold, temperature)
    recorded_raw = selection.get("runtime_raw_probability_threshold")
    if recorded_raw is not None and (
        isinstance(recorded_raw, bool)
        or not isinstance(recorded_raw, (int, float))
        or not math.isclose(float(recorded_raw), raw_threshold, rel_tol=0, abs_tol=1e-15)
    ):
        raise ValueError("RUNTIME_RAW_THRESHOLD_MISMATCH")
    if selection.get("threshold_space", "calibrated_probability") != "calibrated_probability":
        raise ValueError("VALIDATION_THRESHOLD_SPACE_INVALID")
    return {
        # Existing Runtime reads this key and compares it with raw P(SUSPICIOUS).
        "threshold": raw_threshold,
        "threshold_space": "raw_probability",
        "raw_probability_threshold": raw_threshold,
        "validation_calibrated_probability_threshold": float(calibrated_threshold),
        "validation_threshold_space": "calibrated_probability",
        "calibration_temperature": float(temperature),
        "calibration_method": "sigmoid(logit(raw_probability)/temperature)",
        "classification_equivalence": (
            "raw_probability >= raw_probability_threshold iff calibrated_probability "
            ">= validation_calibrated_probability_threshold"
        ),
        "max_length": max_length,
        "score_formula": "round((1-temperature_scale(raw_probability,2.0))*100)",
        "score_temperature": 2.0,
        "runtime_temperature_2_is_display_only": True,
        "dataset_sha256": dataset_sha256,
    }


def run(config: Config) -> dict[str, Any]:
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")
    data = load_dataset(
        config.data_path,
        exclusion_manifest=config.exclusion_manifest,
        verified_relations=config.verified_relations,
        strict_product_isolation=True,
        require_both_labels_per_split=True,
    )
    try:
        import torch
        import transformers
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("Install the project ml dependencies before training") from exc

    torch.set_num_threads(config.cpu_threads)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=config.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "NORMAL", 1: "SUSPICIOUS"},
        label2id={"NORMAL": 0, "SUSPICIOUS": 1},
        local_files_only=config.local_files_only,
    )
    datasets = {
        split: TokenizedTextDataset(rows, tokenizer, config.max_length)
        for split, rows in data.examples.items()
        if split in {"train", "validation"}
    }

    def compute_metrics(prediction) -> dict[str, float]:
        probabilities = _softmax_positive(prediction.predictions)
        values = binary_metrics(prediction.label_ids.tolist(), probabilities, 0.5)
        return {key: float(values[key]) for key in ("precision", "recall", "f1", "pr_auc", "brier_score")}

    class RecoveryCheckpointCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % 20 == 0:
                control.should_save = True
            return control

    arguments = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=config.dataloader_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        logging_steps=20,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[RecoveryCheckpointCallback()],
    )
    output.mkdir(parents=True)
    started = time.monotonic()
    training_result = trainer.train()
    # Validation is the only source for calibration and threshold selection.
    trainer.compute_metrics = None
    validation_prediction = trainer.predict(datasets["validation"])
    validation_probabilities = _softmax_positive(validation_prediction.predictions)
    validation_labels = validation_prediction.label_ids.tolist()
    normal_mask = [label == 0 for label in validation_labels]
    selection = select_validation_policy(
        validation_labels,
        validation_probabilities,
        normal_mask=normal_mask,
        max_normal_fpr=config.max_normal_fpr,
        calibrate=config.calibrate,
    )
    model_dir = output / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    record = {
        "config": asdict(config),
        "data": data.summary,
        "environment": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cpu_threads": torch.get_num_threads(),
            "dataloader_workers": config.dataloader_workers,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        },
        "training_seconds": time.monotonic() - started,
        "training_metrics": training_result.metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "selection": selection,
        "frozen_test_evaluated": False,
    }
    _write_json(output / "training_record.json", record)
    _write_json(output / "validation_selection.json", selection)
    _write_json(
        model_dir / "inference_config.json",
        build_runtime_inference_config(
            selection, max_length=config.max_length, dataset_sha256=data.summary["sha256"]
        ),
    )
    return record


def main() -> None:
    config = parse_args()
    result = run(config)
    print(json.dumps({"output_dir": config.output_dir, "selection": result["selection"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
