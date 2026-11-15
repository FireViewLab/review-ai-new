"""CPU Train-only group cross-validation for P_text procedure stability."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from .data_contract import TokenizedTextDataset, load_dataset, load_verified_relations
from .evaluation import binary_metrics
from .group_cv import _connected_groups, _stable, plan_group_kfold
from .train import _softmax_positive, balanced_class_weights, make_weighted_trainer


def assign_folds(rows, relations, *, n_splits: int, seed: int):
    groups = _connected_groups(rows, relations)
    total = Counter(str(row["label"]) for row in rows)
    targets = {label: total[label] / n_splits for label in ("NORMAL", "SUSPICIOUS")}
    target_size = len(rows) / n_splits
    records = []
    for group in groups:
        labels = Counter(str(row["label"]) for row in group)
        fingerprint = _stable(seed, "\x1f".join(sorted(str(row["row_uid"]) for row in group)))
        records.append((fingerprint, group, labels))
    records.sort(key=lambda item: (-max(item[2].values()), -len(item[1]), _stable(seed, item[0])))
    folds = [[] for _ in range(n_splits)]
    counts = [Counter() for _ in range(n_splits)]
    for fingerprint, group, labels in records:
        def score(index):
            error = 0.0
            for label in ("NORMAL", "SUSPICIOUS"):
                values = [counts[i][label] for i in range(n_splits)]
                values[index] += labels[label]
                error += sum(((value - targets[label]) / max(targets[label], 1)) ** 2 for value in values)
            sizes = [sum(counts[i].values()) for i in range(n_splits)]
            sizes[index] += len(group)
            error += 0.05 * sum(((value - target_size) / max(target_size, 1)) ** 2 for value in sizes)
            return error, sum(counts[index].values()), _stable(seed, f"{fingerprint}:{index}")
        selected = min(range(n_splits), key=score)
        folds[selected].extend(group)
        counts[selected].update(labels)
    return folds


def aggregate(folds):
    names = ("precision", "recall", "f1", "pr_auc", "fpr")
    return {
        name: {"mean": mean(float(row[name]) for row in folds), "std": pstdev(float(row[name]) for row in folds)}
        for name in names
    }


def run(args):
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
    relations = load_verified_relations(args.verified_relations)
    rows = []
    for example, metadata in zip(data.examples["train"], data.metadata["train"], strict=True):
        rows.append(dict(metadata) | {"content": example["text"], "label": "SUSPICIOUS" if example["label"] else "NORMAL", "split": "train"})
    plan = plan_group_kfold(rows, relations=relations, n_splits=args.folds, seed=args.seed)
    if plan["status"] != "FEASIBLE":
        raise ValueError("GROUP_CV_NOT_FEASIBLE:" + ",".join(plan["blocking_reasons"]))
    fold_rows = assign_folds(rows, relations, n_splits=args.folds, seed=args.seed)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments, set_seed

    torch.set_num_threads(args.cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    output.mkdir(parents=True)
    reports = []
    for fold_index, evaluation_rows in enumerate(fold_rows):
        started = time.monotonic()
        evaluation_uids = {str(row["row_uid"]) for row in evaluation_rows}
        training_rows = [row for row in rows if str(row["row_uid"]) not in evaluation_uids]
        if len(training_rows) + len(evaluation_rows) != len(rows):
            raise ValueError("FOLD_PARTITION_INVALID")
        set_seed(args.seed + fold_index)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            id2label={0: "NORMAL", 1: "SUSPICIOUS"},
            label2id={"NORMAL": 0, "SUSPICIOUS": 1},
            local_files_only=True,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        train_examples = [{"text": row["content"], "label": int(row["label"] == "SUSPICIOUS")} for row in training_rows]
        eval_examples = [{"text": row["content"], "label": int(row["label"] == "SUSPICIOUS")} for row in evaluation_rows]
        train_labels = [row["label"] for row in train_examples]
        weights = balanced_class_weights(train_labels)

        def compute_metrics(prediction):
            values = binary_metrics(prediction.label_ids.tolist(), _softmax_positive(prediction.predictions), 0.5)
            return {key: float(values[key]) for key in ("precision", "recall", "f1", "pr_auc", "brier_score")}

        class BestState(TrainerCallback):
            def __init__(self):
                self.metric = float("-inf")
                self.state = None
                self.epoch = None

            def on_evaluate(self, args_, state, control, metrics=None, model=None, **kwargs):
                metric = float((metrics or {})["eval_pr_auc"])
                if metric > self.metric:
                    self.metric = metric
                    self.epoch = float(state.epoch)
                    self.state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        best = BestState()
        training_args = TrainingArguments(
            output_dir=str(output / f"fold_{fold_index}_trainer"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="no",
            seed=args.seed + fold_index,
            data_seed=args.seed + fold_index,
            use_cpu=True,
            dataloader_num_workers=0,
            report_to="none",
            disable_tqdm=True,
        )
        trainer_class = make_weighted_trainer(Trainer, weights)
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=TokenizedTextDataset(train_examples, tokenizer, args.max_length),
            eval_dataset=TokenizedTextDataset(eval_examples, tokenizer, args.max_length),
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[best],
        )
        trainer.train()
        if best.state is None:
            raise ValueError("BEST_FOLD_STATE_MISSING")
        model.load_state_dict(best.state)
        prediction = trainer.predict(TokenizedTextDataset(eval_examples, tokenizer, args.max_length))
        probabilities = _softmax_positive(prediction.predictions)
        labels = prediction.label_ids.tolist()
        metrics = binary_metrics(labels, probabilities, 0.5)
        tn, fp = metrics["confusion_matrix"][0]
        epoch_history = [
            {key.removeprefix("eval_"): value for key, value in entry.items() if key == "epoch" or key.startswith("eval_")}
            for entry in trainer.state.log_history if "eval_pr_auc" in entry
        ]
        report = {
            "fold": fold_index,
            "seed": args.seed + fold_index,
            "train_support": {"NORMAL": train_labels.count(0), "SUSPICIOUS": train_labels.count(1)},
            "evaluation_support": {"NORMAL": labels.count(0), "SUSPICIOUS": labels.count(1)},
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "pr_auc": metrics["pr_auc"],
            "fpr": fp / (tn + fp),
            "confusion_matrix": metrics["confusion_matrix"],
            "fixed_raw_threshold": 0.5,
            "best_epoch_by_fold_pr_auc": best.epoch,
            "epoch_history": epoch_history,
            "class_weights_train_only": weights,
            "training_seconds": time.monotonic() - started,
        }
        reports.append(report)
        (output / f"fold_{fold_index}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        del trainer, model, best

    result = {
        "status": "COMPLETED_TRAIN_ONLY_PROCEDURE_STABILITY_DIAGNOSTIC",
        "dataset_sha256": data.summary["sha256"],
        "frozen_test_evaluated": False,
        "original_validation_used": False,
        "validation_extension_194_used": False,
        "threshold_policy": "pre-fixed raw probability threshold 0.5; no fold threshold tuning",
        "selection_policy": "best epoch within each fold by that fold PR-AUC; optimistic for checkpoint selection and not independent performance",
        "plan": plan,
        "folds": reports,
        "aggregate": aggregate(reports),
    }
    (output / "group_cv_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pilot-exclusions", required=True)
    parser.add_argument("--verified-relations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="monologg/koelectra-base-v3-discriminator")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.set_defaults(gradient_checkpointing=True)
    return parser.parse_args(argv)


def main():
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
