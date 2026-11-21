"""Build a code-only, private team-share ZIP for the integrated AI candidate."""

from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile


MODEL_WEIGHTS_SHA256 = "012b65174cdbc2e1fb36d75519438043bebd4317a69ea9997325b66bb436474b"
MODEL_ZIP_SHA256 = "f1fc7abe2248c32a9d310834ca38687e17d72d4eb24a528045be345836e368b9"
PACKAGE_NAME = "review-integrated-ai-v3-candidate-final-20261116"


SOURCE_FILES = (
    "app/__init__.py",
    "app/analyzers/__init__.py",
    "app/analyzers/p_text.py",
    "app/analyzers/behavior.py",
    "app/analyzers/network.py",
    "app/integrations/__init__.py",
    "app/integrations/similarity.py",
    "app/scoring/__init__.py",
    "app/scoring/meta_scorer.py",
    "app/training/__init__.py",
    "app/training/p_text_v3/integrated_ai_candidate.py",
    "app/training/p_text_v3/run_integrated_ai_candidate.py",
    "app/training/p_text_v3/review_evidence_tier_v1.py",
    "app/training/p_text_v3/review_evidence_tier_v2.py",
    "app/training/p_text_v3/review_evidence_tier_v3.py",
    "app/training/p_text_v3/review_evidence_policy_index_v3.py",
    "tests/unit/test_integrated_ai_candidate.py",
    "tests/unit/test_review_evidence_tier_v3.py",
    "tests/unit/test_review_evidence_policy_index_v3.py",
)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_sources(repo: Path, package: Path) -> None:
    for relative in SOURCE_FILES:
        source = repo / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (package / "app/training/p_text_v3/__init__.py").write_text(
        '"""Standalone offline Re:view integrated AI candidate."""\n',
        encoding="utf-8",
    )


def _write_examples(package: Path) -> None:
    examples = package / "examples"
    examples.mkdir()
    single = {
        "platform": "synthetic",
        "product_id": "product-example",
        "review_id": "review-example-001",
        "content": "2주 동안 직접 사용했고 오후에는 조금 건조했지만 발림성과 지속력은 좋았습니다.",
        "rating": 4,
        "rating_scale_max": 5,
        "rating_source": "synthetic_explicit",
        "rating_source_verified": True,
        "image_count": 1,
        "image_count_source": "synthetic_explicit",
        "image_count_source_verified": True,
        "behavior": None,
    }
    batch = {"reviews": [single, {
        **single,
        "review_id": "review-example-002",
        "content": "아주 좋아요",
        "rating": 5,
        "image_count": 0,
    }]}
    (examples / "single_input.json").write_text(
        json.dumps(single, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (examples / "product_batch_input.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build(
    *, repo: Path, output_root: Path, readme: Path,
    regression_summary: Path, synthetic_smoke: Path,
) -> tuple[Path, Path, dict]:
    package = output_root / PACKAGE_NAME
    archive = output_root / f"{PACKAGE_NAME}.zip"
    if package.exists() or archive.exists():
        raise FileExistsError("package directory or ZIP already exists")
    package.mkdir(parents=True)
    _copy_sources(repo, package)
    shutil.copy2(readme, package / "README.md")
    requirements = (
        "torch>=2.2,<3.0\ntransformers>=4.46,<5.0\nnumpy>=1.26,<3.0\n"
        "accelerate>=1.2,<2.0\npytest>=8,<9\n"
    )
    (package / "requirements-ai.txt").write_text(requirements, encoding="utf-8")
    _write_examples(package)
    verification = package / "verification"
    verification.mkdir()
    shutil.copy2(regression_summary, verification / "policy_index_603_summary.json")
    shutil.copy2(synthetic_smoke, verification / "synthetic_smoke_result.json")

    files = {}
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        relative = path.relative_to(package).as_posix()
        files[relative] = {"sha256": _sha(path), "size_bytes": path.stat().st_size}
    manifest = {
        "package_status": "OFFLINE_TEAM_REVIEW_CANDIDATE_NOT_RUNTIME_DEFAULT",
        "created_date": str(date.today()),
        "components": {
            "p_text": "v3.1 B",
            "p_network": "baseline",
            "p_behavior": "baseline",
            "meta_scorer": "existing",
            "evidence_tier": "review_evidence_tier_v3",
            "policy_index": "BALANCED_INTERIOR_V1",
        },
        "external_model_package": {
            "zip_name": "ptext-v3.1-b-experimental-seed20261115-share-20261116.zip",
            "zip_sha256": MODEL_ZIP_SHA256,
            "model_weights_sha256": MODEL_WEIGHTS_SHA256,
            "included_in_this_zip": False,
        },
        "excluded": [
            "model_weights", "private_reviews", "identifiers", "training_data",
            "frozen_test", "independent_evaluation_150", "review_labels",
        ],
        "runtime_default_changed": False,
        "files": files,
    }
    manifest_path = package / "package_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sum_paths = sorted(item for item in package.rglob("*") if item.is_file())
    (package / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.relative_to(package).as_posix()}\n" for path in sum_paths),
        encoding="utf-8",
    )

    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(item for item in package.rglob("*") if item.is_file()):
            handle.write(path, (Path(PACKAGE_NAME) / path.relative_to(package)).as_posix())
    return package, archive, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--regression-summary", type=Path, required=True)
    parser.add_argument("--synthetic-smoke", type=Path, required=True)
    args = parser.parse_args()
    package, archive, manifest = build(
        repo=args.repo.resolve(), output_root=args.output_root,
        readme=args.readme, regression_summary=args.regression_summary,
        synthetic_smoke=args.synthetic_smoke,
    )
    print(json.dumps({
        "status": manifest["package_status"],
        "package": str(package.resolve()),
        "zip": str(archive.resolve()),
        "zip_sha256": _sha(archive),
        "zip_size_bytes": archive.stat().st_size,
        "file_count": len(list(item for item in package.rglob("*") if item.is_file())),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
