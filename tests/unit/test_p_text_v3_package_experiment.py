import json

import pytest

from app.training.p_text_v3.package_experiment import REQUIRED_MODEL_FILES, create_package


def _experiment(tmp_path):
    experiment = tmp_path / "experiment"
    model = experiment / "model"
    model.mkdir(parents=True)
    for name in REQUIRED_MODEL_FILES:
        content = "weights" if name == "model.safetensors" else "{}"
        (model / name).write_text(content, encoding="utf-8")
    (model / "inference_config.json").write_text(
        json.dumps(
            {
                "threshold_space": "raw_probability",
                "validation_threshold_space": "calibrated_probability",
            }
        ),
        encoding="utf-8",
    )
    for name in ("training_record.json", "validation_results.json", "validation_selection.json"):
        (experiment / name).write_text("{}", encoding="utf-8")
    comparison = tmp_path / "comparison.md"
    comparison.write_text("# comparison", encoding="utf-8")
    return experiment, comparison


def test_create_package_is_explicitly_experimental_and_hashed(tmp_path):
    experiment, comparison = _experiment(tmp_path)
    output = tmp_path / "package"
    manifest = create_package(experiment, output, comparison)
    assert manifest["frozen_test_evaluated"] is False
    assert manifest["validation_extension_194_used"] is False
    assert "model/model.safetensors" in manifest["files"]
    assert (output / "package_manifest.json").is_file()


def test_create_package_refuses_overwrite(tmp_path):
    experiment, comparison = _experiment(tmp_path)
    output = tmp_path / "package"
    output.mkdir()
    with pytest.raises(FileExistsError, match="OUTPUT_ALREADY_EXISTS"):
        create_package(experiment, output, comparison)


def test_create_package_rejects_threshold_space_confusion(tmp_path):
    experiment, comparison = _experiment(tmp_path)
    config = experiment / "model" / "inference_config.json"
    config.write_text(
        json.dumps({"threshold_space": "calibrated_probability", "validation_threshold_space": "calibrated_probability"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="RUNTIME_THRESHOLD_SPACE_INVALID"):
        create_package(experiment, tmp_path / "package", comparison)
