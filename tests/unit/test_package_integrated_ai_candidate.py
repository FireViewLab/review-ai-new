from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from app.training.p_text_v3.package_integrated_ai_candidate import (
    PACKAGE_NAME,
    build,
)


def test_code_only_package_excludes_weights_and_private_data(tmp_path) -> None:
    readme = tmp_path / "README-source.md"
    regression = tmp_path / "regression.json"
    smoke = tmp_path / "smoke.json"
    readme.write_text("handoff", encoding="utf-8")
    regression.write_text('{"rows": 603}', encoding="utf-8")
    smoke.write_text('{"synthetic": true}', encoding="utf-8")
    package, archive, manifest = build(
        repo=Path.cwd(), output_root=tmp_path / "output", readme=readme,
        regression_summary=regression, synthetic_smoke=smoke,
    )
    assert archive.is_file()
    assert manifest["external_model_package"]["included_in_this_zip"] is False
    assert not list(package.rglob("*.safetensors"))
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    assert all(name.startswith(PACKAGE_NAME + "/") for name in names)
    assert not any(name.endswith((".safetensors", ".xlsx", ".csv")) for name in names)
    assert any(name.endswith("README.md") for name in names)
    assert any(name.endswith("integrated_ai_candidate.py") for name in names)


def test_package_refuses_overwrite(tmp_path) -> None:
    output = tmp_path / "output"
    (output / PACKAGE_NAME).mkdir(parents=True)
    file = tmp_path / "input"
    file.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build(
            repo=Path.cwd(), output_root=output, readme=file,
            regression_summary=file, synthetic_smoke=file,
        )
