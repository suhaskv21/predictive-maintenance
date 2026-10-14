"""Where the notebook finds the unit and the config, on a laptop and on the published site.

Run locally, the notebook's folder holds ``models/`` and ``configs/``. Run in the browser, the
notebook's location is the site's URL and the build stages the same files under ``public/``
(``deploy.stage``); Pyodide has no file access to them, so they are downloaded into a temporary
folder. A unit file that cannot be downloaded is left out, so ``load_unit`` reports it missing and
the app shows the error state; no other model or threshold stands in.
"""

import re
import tempfile
import urllib.request
from pathlib import Path, PurePath
from urllib.error import URLError

from predictive_maintenance.training.unit import unit_paths

CONFIG_NAME = "default.toml"
_TIMEOUT_SECONDS = 60


def _base_url(location: PurePath) -> str:
    """The URL of ``location``; a path object collapses ``http://`` to ``http:/``."""
    return re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*):/+", r"\1://", str(location)).rstrip("/")


def _download(url: str, target: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
            target.write_bytes(response.read())
    except (URLError, OSError):
        return False
    return True


def models_dir(location: PurePath, model: str) -> Path:
    """The folder holding the unit of ``model``, downloaded first when ``location`` is a URL."""
    if isinstance(location, Path):
        return location / "models"
    folder = Path(tempfile.mkdtemp(prefix="unit-"))
    unit = unit_paths(folder, model)
    for path in (*unit.sidecars, unit.manifest):
        _download(f"{_base_url(location)}/public/models/{path.name}", path)
    return folder


def config_file(location: PurePath) -> Path:
    """The config file, downloaded first when ``location`` is a URL.

    Raises:
        OSError: if the config cannot be downloaded; nothing can be assessed without it.
    """
    if isinstance(location, Path):
        return location / "configs" / CONFIG_NAME
    target = Path(tempfile.mkdtemp(prefix="config-")) / CONFIG_NAME
    url = f"{_base_url(location)}/public/configs/{CONFIG_NAME}"
    if not _download(url, target):
        raise OSError(f"cannot download {url}")
    return target
