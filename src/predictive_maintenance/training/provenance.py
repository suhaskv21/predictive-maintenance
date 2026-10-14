"""Git provenance of a training run: train only from a clean commit and record it (ADR 0015).

The commit pins the code, ``configs/``, ``uv.lock`` and the raw dataset, so the manifest stores
the commit SHA and nothing that the commit already determines. An artifact trained with
``--allow-dirty`` records ``None``, which the deploy contract rejects.
"""

import re
import subprocess
from pathlib import Path

_SHA = re.compile(r"[0-9a-f]{40}")


class ProvenanceError(RuntimeError):
    """The Git state does not allow a reproducible training run."""


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        )
    except OSError as error:
        raise ProvenanceError(f"cannot run git: {error}") from error
    except subprocess.CalledProcessError as error:
        raise ProvenanceError(f"git {' '.join(args)} failed: {error.stderr.strip()}") from error
    return result.stdout.rstrip("\n")


def clean_head_commit(root: Path) -> str:
    """The SHA of ``HEAD`` in the repository at ``root``, only if its working tree is clean.

    Modified and untracked files make the tree dirty; git-ignored files (``models/``) do not.

    Raises:
        ProvenanceError: if Git is unavailable, ``root`` is not a checkout, or the tree is dirty.
    """
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise ProvenanceError(
            "the working tree is not clean, so the model could not be reproduced from a commit; "
            "commit or stash these changes (or pass --allow-dirty for a model that will not be "
            f"published):\n{dirty}"
        )
    return _git(root, "rev-parse", "HEAD")


def is_commit_sha(value: object) -> bool:
    """Whether ``value`` is a full 40-character lowercase hex Git commit SHA."""
    return isinstance(value, str) and _SHA.fullmatch(value) is not None
