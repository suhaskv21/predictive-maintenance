import dataclasses
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictive_maintenance.config import Config, DeployConfig, load_config
from predictive_maintenance.deploy.contract import (
    PYODIDE_BUNDLED,
    check_site_size,
    check_unit,
    check_wheel_dependencies,
    main,
    run_contract,
)
from predictive_maintenance.evaluation.threshold import OperatingPoint
from predictive_maintenance.training.unit import (
    build_operating_metrics,
    reject_policy,
    unit_paths,
    write_unit,
)

MODEL = "toy"
THRESHOLD = 0.218


def readings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [298.0, 300.0, 302.0, 299.0],
            "Process temperature [K]": [308.0, 310.0, 312.0, 309.0],
            "Rotational speed [rpm]": [1400, 1500, 1600, 1450],
            "Torque [Nm]": [30.0, 40.0, 50.0, 35.0],
            "Tool wear [min]": [0, 100, 200, 50],
        }
    )


def write_toy_unit(models_dir: Path, *, threshold: float = THRESHOLD) -> None:
    frame = readings()
    pipeline = Pipeline([("model", DummyClassifier(strategy="prior"))]).fit(frame, [0, 1, 0, 1])
    thresholds = {5.0: 0.3, 10.0: threshold}
    write_unit(
        models_dir,
        pipeline=pipeline,
        point=OperatingPoint(
            model=MODEL,
            default_cost_ratio=10.0,
            thresholds=thresholds,
            bands=None,
            risk_floor=0.01,
        ),
        policy=reject_policy(MODEL, 10.0),
        metrics=build_operating_metrics(
            MODEL, [0, 0, 1, 1], np.array([[0.1, 0.2, 0.6, 0.9]]), thresholds, frame
        ),
        golden=frame.head(2),
        golden_scores=[0.5, 0.5],
    )


def config(*, budget_mb: float = 100.0, tolerance: float = 0.0005) -> Config:
    return dataclasses.replace(
        load_config(),
        deploy=DeployConfig(
            accepted_thresholds={MODEL: 0.218},
            threshold_tolerance=tolerance,
            site_budget_mb=budget_mb,
        ),
    )


def write_wheel(directory: Path, requirements: list[str]) -> Path:
    wheel = directory / "toy-0.1.0-py3-none-any.whl"
    lines = ["Metadata-Version: 2.3", "Name: toy", "Version: 0.1.0", *requirements]
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("toy-0.1.0.dist-info/METADATA", "\n".join(lines) + "\n")
    return wheel


def write_site(directory: Path, megabytes: float) -> Path:
    directory.mkdir()
    (directory / "index.html").write_bytes(b"x" * int(megabytes * 1024 * 1024))
    return directory


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    path = tmp_path / "models"
    write_toy_unit(path)
    return path


def test_a_complete_unit_passes(models_dir: Path) -> None:
    assert check_unit(models_dir, MODEL, config()) == []


@pytest.mark.parametrize("name", ["pipeline", "operating_point", "missing", "operating_metrics"])
def test_a_missing_unit_file_fails_the_build(models_dir: Path, name: str) -> None:
    getattr(unit_paths(models_dir, MODEL), name).unlink()
    failures = check_unit(models_dir, MODEL, config())
    assert len(failures) == 1
    assert "missing file" in failures[0]


def test_a_missing_manifest_fails_the_build(models_dir: Path) -> None:
    unit_paths(models_dir, MODEL).manifest.unlink()
    assert any("missing file" in failure for failure in check_unit(models_dir, MODEL, config()))


def test_a_file_that_differs_from_its_manifest_hash_fails_the_build(models_dir: Path) -> None:
    unit_paths(models_dir, MODEL).missing.write_text("{}")
    failures = check_unit(models_dir, MODEL, config())
    assert any("does not match the manifest" in failure for failure in failures)


def test_a_threshold_outside_tolerance_of_the_accepted_figure_fails_the_build(
    tmp_path: Path,
) -> None:
    write_toy_unit(tmp_path, threshold=0.25)
    failures = check_unit(tmp_path, MODEL, config())
    assert any("accepted" in failure for failure in failures)


def test_a_threshold_within_tolerance_passes(tmp_path: Path) -> None:
    write_toy_unit(tmp_path, threshold=0.21837)
    assert check_unit(tmp_path, MODEL, config()) == []


def test_a_model_with_no_accepted_threshold_fails_the_build(models_dir: Path) -> None:
    base = config()
    unaccepted = dataclasses.replace(
        base, deploy=dataclasses.replace(base.deploy, accepted_thresholds={})
    )
    assert any("accepted" in failure for failure in check_unit(models_dir, MODEL, unaccepted))


def test_a_site_within_the_budget_passes(tmp_path: Path) -> None:
    assert check_site_size(write_site(tmp_path / "site", 1), 2) == []


def test_a_site_over_the_budget_fails_the_build(tmp_path: Path) -> None:
    failures = check_site_size(write_site(tmp_path / "site", 3), 2)
    assert len(failures) == 1
    assert "budget" in failures[0]


def test_a_missing_site_fails_the_build(tmp_path: Path) -> None:
    assert check_site_size(tmp_path / "absent", 100) != []


def test_a_wheel_that_depends_only_on_bundled_packages_passes(tmp_path: Path) -> None:
    wheel = write_wheel(
        tmp_path, ["Requires-Dist: scikit-learn==1.8.0", "Requires-Dist: polars==1.33.1"]
    )
    assert check_wheel_dependencies(wheel) == []


def test_a_wheel_that_depends_on_a_package_pyodide_does_not_bundle_fails_the_build(
    tmp_path: Path,
) -> None:
    wheel = write_wheel(
        tmp_path, ["Requires-Dist: joblib==1.5.3", "Requires-Dist: streamlit>=1.65"]
    )
    failures = check_wheel_dependencies(wheel)
    assert len(failures) == 1
    assert "streamlit" in failures[0]


def test_an_extra_only_dependency_is_not_installed_in_the_browser(tmp_path: Path) -> None:
    wheel = write_wheel(tmp_path, ['Requires-Dist: streamlit>=1.65; extra == "streamlit"'])
    assert check_wheel_dependencies(wheel) == []


def test_the_bundled_set_names_the_packages_marimo_ships() -> None:
    assert {"scikit-learn", "polars", "joblib", "numpy", "scipy"} <= PYODIDE_BUNDLED


def test_the_contract_reports_every_failure_at_once(tmp_path: Path) -> None:
    write_toy_unit(tmp_path / "models", threshold=0.25)
    wheel = write_wheel(tmp_path, ["Requires-Dist: streamlit>=1.65"])
    site = write_site(tmp_path / "site", 3)
    failures = run_contract(
        models_dir=tmp_path / "models",
        model=MODEL,
        wheel=wheel,
        site=site,
        config=config(budget_mb=2),
    )
    assert len(failures) == 3


def test_the_command_exits_nonzero_on_a_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_toy_unit(tmp_path / "models")
    wheel = write_wheel(tmp_path, ["Requires-Dist: polars==1.33.1"])
    site = write_site(tmp_path / "site", 1)
    args = ["--models", str(tmp_path / "models"), "--model", MODEL, "--wheel", str(wheel)]
    args += ["--site", str(site)]
    # The repository config does not accept the toy model.
    assert main(args) == 1
    assert "accepted" in capsys.readouterr().out
