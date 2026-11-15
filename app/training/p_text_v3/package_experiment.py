"""Create an immutable, self-describing P_text experimental model package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REQUIRED_MODEL_FILES = {
    "config.json",
    "inference_config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_package(
    experiment_dir: str | Path,
    output_dir: str | Path,
    comparison_markdown: str | Path,
) -> dict:
    source = Path(experiment_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")
    model_source = source / "model"
    missing = sorted(name for name in REQUIRED_MODEL_FILES if not (model_source / name).is_file())
    if missing:
        raise ValueError(f"MODEL_PACKAGE_FILES_MISSING:{','.join(missing)}")

    inference = json.loads((model_source / "inference_config.json").read_text(encoding="utf-8"))
    if inference.get("threshold_space") != "raw_probability":
        raise ValueError("RUNTIME_THRESHOLD_SPACE_INVALID")
    if inference.get("validation_threshold_space") != "calibrated_probability":
        raise ValueError("VALIDATION_THRESHOLD_SPACE_INVALID")

    output.mkdir(parents=True)
    shutil.copytree(model_source, output / "model")
    for name in ("training_record.json", "validation_results.json", "validation_selection.json"):
        shutil.copy2(source / name, output / name)
    shutil.copy2(comparison_markdown, output / "model_comparison.md")
    (output / "README.md").write_text(
        "# P_text v3.1 experimental\n\n"
        "이 패키지는 단일 검수자+AI 보조 라벨로 학습한 **실험 모델**이며 운영 Runtime에 활성화되지 않았습니다. "
        "공식 2인 Human Ground Truth나 일반화 성능 확정 결과로 해석하면 안 됩니다.\n\n"
        "## CPU 로딩\n\n"
        "```python\n"
        "from transformers import AutoModelForSequenceClassification, AutoTokenizer\n"
        "tokenizer = AutoTokenizer.from_pretrained('model', local_files_only=True)\n"
        "model = AutoModelForSequenceClassification.from_pretrained('model', local_files_only=True)\n"
        "```\n\n"
        "판정에는 `model/inference_config.json`의 `raw_probability_threshold`를 사용합니다. "
        "Validation calibration threshold와 혼동하지 마십시오. P_text 표시 점수의 temperature=2.0은 별도입니다.\n\n"
        "Frozen Test는 평가하지 않았고 Validation Extension 194건은 학습·선택·보정에 사용하지 않았습니다.\n",
        encoding="utf-8",
    )

    manifest = {
        "package_status": "PTEXT_V3_1_EXPERIMENTAL_NOT_RUNTIME_ACTIVATED",
        "selected_experiment": source.name,
        "official_human_ground_truth": False,
        "frozen_test_evaluated": False,
        "validation_extension_194_used": False,
        "files": {},
    }
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        manifest["files"][relative] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comparison-markdown", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = create_package(args.experiment_dir, args.output_dir, args.comparison_markdown)
    print(json.dumps({"status": manifest["package_status"], "file_count": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
