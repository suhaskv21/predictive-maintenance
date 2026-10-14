import subprocess
from pathlib import Path

import pytest

from predictive_maintenance.training.provenance import (
    ProvenanceError,
    clean_head_commit,
    is_commit_sha,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("models/*\n")
    (tmp_path / "tracked.txt").write_text("one\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "first")
    return tmp_path


def test_a_clean_tree_returns_the_full_head_sha(repo: Path) -> None:
    commit = clean_head_commit(repo)
    assert commit == git(repo, "rev-parse", "HEAD")
    assert is_commit_sha(commit)


def test_a_modified_tracked_file_is_refused(repo: Path) -> None:
    (repo / "tracked.txt").write_text("two\n")
    with pytest.raises(ProvenanceError, match="not clean"):
        clean_head_commit(repo)


def test_an_untracked_file_is_refused(repo: Path) -> None:
    (repo / "new.py").write_text("x = 1\n")
    with pytest.raises(ProvenanceError, match="new.py"):
        clean_head_commit(repo)


def test_a_staged_but_uncommitted_change_is_refused(repo: Path) -> None:
    (repo / "tracked.txt").write_text("two\n")
    git(repo, "add", "tracked.txt")
    with pytest.raises(ProvenanceError, match="not clean"):
        clean_head_commit(repo)


def test_git_ignored_files_such_as_the_models_do_not_make_the_tree_dirty(repo: Path) -> None:
    (repo / "models").mkdir()
    (repo / "models" / "a.pipeline.joblib").write_text("artifact")
    assert is_commit_sha(clean_head_commit(repo))


def test_a_directory_that_is_not_a_checkout_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="failed"):
        clean_head_commit(tmp_path)


def test_a_repository_without_a_commit_is_refused(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    with pytest.raises(ProvenanceError, match="failed"):
        clean_head_commit(tmp_path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0123456789abcdef0123456789abcdef01234567", True),
        ("0123456789ABCDEF0123456789ABCDEF01234567", False),
        ("0123456789abcdef", False),
        ("0123456789abcdef0123456789abcdef01234567\n", False),
        ("", False),
        (None, False),
        (123, False),
    ],
)
def test_only_a_full_lowercase_hex_sha_is_a_commit(value: object, expected: bool) -> None:
    assert is_commit_sha(value) is expected
