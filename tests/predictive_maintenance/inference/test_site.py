import functools
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import pytest

from predictive_maintenance.inference.site import config_file, models_dir
from predictive_maintenance.training.unit import unit_paths

MODEL = "toy"


def write_unit_files(directory: Path) -> None:
    directory.mkdir(parents=True)
    unit = unit_paths(directory, MODEL)
    for path in (*unit.sidecars, unit.manifest):
        path.write_text(path.name)


@pytest.fixture
def site(tmp_path: Path) -> Iterator[PurePosixPath]:
    """A served site root with the layout the build stages: public/models and public/configs."""
    root = tmp_path / "site"
    write_unit_files(root / "public" / "models")
    (root / "public" / "configs").mkdir()
    (root / "public" / "configs" / "default.toml").write_text("seed = 7\n")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # marimo's URLPath is a PurePosixPath that prints "http://" for "http:/".
        yield PurePosixPath(f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()


def test_a_local_notebook_uses_the_models_and_config_beside_it(tmp_path: Path) -> None:
    assert models_dir(tmp_path, MODEL) == tmp_path / "models"
    assert config_file(tmp_path) == tmp_path / "configs" / "default.toml"


def test_a_site_notebook_downloads_the_five_files_of_the_unit(site: PurePosixPath) -> None:
    downloaded = models_dir(site, MODEL)
    unit = unit_paths(downloaded, MODEL)
    assert {path.read_text() for path in (*unit.sidecars, unit.manifest)} == {
        path.name for path in (*unit.sidecars, unit.manifest)
    }


def test_a_site_notebook_downloads_the_config(site: PurePosixPath) -> None:
    assert config_file(site).read_text() == "seed = 7\n"


def test_a_file_the_site_does_not_serve_is_left_missing_for_the_loader_to_report(
    site: PurePosixPath,
) -> None:
    downloaded = models_dir(site, "absent")
    assert list(downloaded.iterdir()) == []
