import json
import subprocess
import sys
from pathlib import Path

import pytest

from predictive_maintenance.deploy.stage import stage_site, stamp_notebook
from predictive_maintenance.training.unit import unit_paths

WHEEL = "predictive_maintenance-0.1.0-py3-none-any.whl"
NOTEBOOK = 'import marimo\n\nBUILD_RELEASE = "local run"\nMODEL = "toy"\n'
REPO_NOTEBOOK = Path(__file__).resolve().parents[3] / "marimo_app.py"


def header_block(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[: lines.index("# ///", 1) + 1])


def test_the_stamped_notebook_installs_the_local_wheel_from_a_pep_723_header() -> None:
    stamped = stamp_notebook(NOTEBOOK, wheel_name=WHEEL, release="v0.3")
    assert stamped.startswith("# /// script\n")
    assert f'"predictive-maintenance @ ./dist/{WHEEL}"' in header_block(stamped)
    assert stamped.endswith(NOTEBOOK.replace('"local run"', '"v0.3"'))


def test_the_stamped_notebook_shows_the_given_release() -> None:
    stamped = stamp_notebook(NOTEBOOK, wheel_name=WHEEL, release="v0.3")
    assert 'BUILD_RELEASE = "v0.3"' in stamped
    assert "local run" not in stamped


def test_stamping_refuses_a_notebook_without_the_release_placeholder() -> None:
    with pytest.raises(ValueError, match="BUILD_RELEASE"):
        stamp_notebook("import marimo\n", wheel_name=WHEEL, release="v0.3")


@pytest.mark.parametrize("release", ['a"b', "a\nb", "", "a b"])
def test_stamping_refuses_a_release_that_is_not_a_plain_identifier(release: str) -> None:
    with pytest.raises(ValueError, match="release"):
        stamp_notebook(NOTEBOOK, wheel_name=WHEEL, release=release)


def test_the_repository_notebook_can_be_stamped() -> None:
    stamped = stamp_notebook(REPO_NOTEBOOK.read_text(), wheel_name=WHEEL, release="v0.3")
    assert 'BUILD_RELEASE = "v0.3"' in stamped


def make_inputs(root: Path) -> dict[str, Path]:
    models = root / "models"
    models.mkdir()
    unit = unit_paths(models, "toy")
    for path in (*unit.sidecars, unit.manifest):
        path.write_text(json.dumps({"file": path.name}))
    (models / "other-model.pipeline.joblib").write_text("not part of the unit")
    config = root / "default.toml"
    config.write_text("seed = 1\n")
    wheel = root / WHEEL
    wheel.write_bytes(b"wheel")
    notebook = root / "marimo_app.py"
    notebook.write_text(NOTEBOOK)
    return {"models": models, "config": config, "wheel": wheel, "notebook": notebook}


def test_staging_lays_out_the_notebook_wheel_unit_and_config_for_the_export(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    build = tmp_path / "build"
    notebook = stage_site(
        notebook=inputs["notebook"],
        models_dir=inputs["models"],
        config_path=inputs["config"],
        wheel=inputs["wheel"],
        release="v0.3",
        model="toy",
        build_dir=build,
    )
    assert notebook == build / "marimo_app.py"
    assert 'BUILD_RELEASE = "v0.3"' in notebook.read_text()
    assert (build / "dist" / WHEEL).read_bytes() == b"wheel"
    unit = unit_paths(build / "public" / "models", "toy")
    assert all(path.is_file() for path in (*unit.sidecars, unit.manifest))
    assert (build / "public" / "configs" / "default.toml").read_text() == "seed = 1\n"


def test_staging_ships_only_the_five_files_of_the_unit(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    stage_site(
        notebook=inputs["notebook"],
        models_dir=inputs["models"],
        config_path=inputs["config"],
        wheel=inputs["wheel"],
        release="v0.3",
        model="toy",
        build_dir=tmp_path / "build",
    )
    shipped = sorted(path.name for path in (tmp_path / "build" / "public" / "models").iterdir())
    assert len(shipped) == 5
    assert not any(name.startswith("other") for name in shipped)


def test_staging_fails_when_a_unit_file_is_missing(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    unit_paths(inputs["models"], "toy").pipeline.unlink()
    with pytest.raises(FileNotFoundError):
        stage_site(
            notebook=inputs["notebook"],
            models_dir=inputs["models"],
            config_path=inputs["config"],
            wheel=inputs["wheel"],
            release="v0.3",
            model="toy",
            build_dir=tmp_path / "build",
        )


def test_staging_starts_from_an_empty_build_directory(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    build = tmp_path / "build"
    (build / "public").mkdir(parents=True)
    (build / "public" / "stale.txt").write_text("old")
    stage_site(
        notebook=inputs["notebook"],
        models_dir=inputs["models"],
        config_path=inputs["config"],
        wheel=inputs["wheel"],
        release="v0.3",
        model="toy",
        build_dir=build,
    )
    assert not (build / "public" / "stale.txt").exists()


def test_the_command_stages_from_the_arguments(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    build = tmp_path / "build"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "predictive_maintenance.deploy.stage",
            "--notebook",
            str(inputs["notebook"]),
            "--models",
            str(inputs["models"]),
            "--config",
            str(inputs["config"]),
            "--wheel",
            str(inputs["wheel"]),
            "--release",
            "v0.3",
            "--model",
            "toy",
            "--build-dir",
            str(build),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (build / "marimo_app.py").is_file()
