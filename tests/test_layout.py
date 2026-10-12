"""Guard: tests must mirror the source tree (src/pkg/x/y.py -> tests/pkg/x/test_y.py)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def test_every_source_module_has_a_mirrored_test() -> None:
    missing = []
    for module in SRC.rglob("*.py"):
        if module.name == "__init__.py":
            continue
        rel = module.relative_to(SRC)
        expected = TESTS / rel.parent / f"test_{rel.name}"
        if not expected.exists():
            missing.append(str(expected.relative_to(ROOT)))
    assert not missing, f"Missing mirrored tests: {missing}"
