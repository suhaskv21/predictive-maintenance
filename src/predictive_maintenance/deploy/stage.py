"""Stage the entry notebook for the WebAssembly export (ADR 0010, ADR 0012).

``python -m predictive_maintenance.deploy.stage`` writes a build directory that
``marimo export html-wasm`` turns into the site: the notebook stamped with its build release and a
PEP 723 header naming the local wheel, the wheel itself, and a ``public/`` folder holding the
unit's five files and the config. marimo copies ``public/`` into the site and installs the wheel
in the browser; the notebook fetches the files from there (``inference.site``).
"""

import argparse
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

from predictive_maintenance.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from predictive_maintenance.training.train import DEFAULT_MODEL
from predictive_maintenance.training.unit import unit_paths

RELEASE_PLACEHOLDER = 'BUILD_RELEASE = "local run"'
_RELEASE = re.compile(r"[0-9A-Za-z._-]+")


def stamp_notebook(source: str, *, wheel_name: str, release: str) -> str:
    """The notebook text with the wheel header prepended and ``release`` as its build release.

    Raises:
        ValueError: if ``release`` is not a plain identifier, or the notebook does not contain the
            ``BUILD_RELEASE`` placeholder exactly once.
    """
    if _RELEASE.fullmatch(release) is None:
        raise ValueError(f"release {release!r} is not a plain identifier")
    if source.count(RELEASE_PLACEHOLDER) != 1:
        raise ValueError(f"the notebook must contain exactly one {RELEASE_PLACEHOLDER!r}")
    header = (
        "# /// script\n"
        '# requires-python = ">=3.12"\n'
        "# dependencies = [\n"
        f'#     "predictive-maintenance @ ./dist/{wheel_name}",\n'
        "# ]\n"
        "# ///\n\n"
    )
    return header + source.replace(RELEASE_PLACEHOLDER, f'BUILD_RELEASE = "{release}"')


def stage_site(
    *,
    notebook: Path,
    models_dir: Path,
    config_path: Path,
    wheel: Path,
    release: str,
    model: str,
    build_dir: Path,
) -> Path:
    """Write the build directory for the export and return the stamped notebook's path.

    Raises:
        FileNotFoundError: if a file of the unit is missing.
    """
    stamped = stamp_notebook(notebook.read_text(), wheel_name=wheel.name, release=release)
    unit = unit_paths(models_dir, model)
    files = (*unit.sidecars, unit.manifest)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    shutil.rmtree(build_dir, ignore_errors=True)
    (build_dir / "dist").mkdir(parents=True)
    (build_dir / "public" / "models").mkdir(parents=True)
    (build_dir / "public" / "configs").mkdir(parents=True)
    staged = build_dir / notebook.name
    staged.write_text(stamped)
    shutil.copyfile(wheel, build_dir / "dist" / wheel.name)
    for path in files:
        shutil.copyfile(path, build_dir / "public" / "models" / path.name)
    shutil.copyfile(config_path, build_dir / "public" / "configs" / config_path.name)
    return staged


def main(argv: Sequence[str] | None = None) -> int:
    """Stage the build directory from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=PROJECT_ROOT / "marimo_app.py")
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--build-dir", type=Path, default=PROJECT_ROOT / "build" / "site-src")
    args = parser.parse_args(argv)
    staged = stage_site(
        notebook=args.notebook,
        models_dir=args.models,
        config_path=args.config,
        wheel=args.wheel,
        release=args.release,
        model=args.model,
        build_dir=args.build_dir,
    )
    print(f"staged {staged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
