"""Keep requirements.txt honest against the code.

An audit on 2026-09-30 found that nine pinned packages were imported nowhere and that one package
the code does import, pymupdf, was missing entirely -- so a clean-environment reproduction would
have installed a CUDA build of torch it never uses and then failed on the bibliography step. Both
directions matter for Gate 9, and neither is visible without checking.

This parses every import in the repository and compares it with the pinned set, so the file cannot
drift from the code again.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"

#: Import name -> distribution name, where they differ.
DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "fitz": "pymupdf",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "cv2": "opencv-python",
}

#: Pinned but not imported, each for a reason recorded in requirements.txt itself. pytest-timeout
#: is a plugin configured through pytest.ini rather than imported.
JUSTIFIED_WITHOUT_IMPORT = {"pytest-timeout"}

SKIP_DIRECTORIES = {".venv", "__pycache__", ".git", "node_modules"}


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-")


def pinned() -> dict[str, str]:
    """Package name -> the requirement line, for the active (uncommented) pins only."""
    out: dict[str, str] = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = re.split(r"[=<>!\[;]", stripped)[0].strip()
        if name:
            out[_normalise(name)] = stripped
    return out


def _guarded_imports(tree: ast.Module) -> set[int]:
    """Line numbers of imports wrapped in a try that catches ImportError.

    Such an import is optional by construction: the module is written to work without the package.
    Requiring it would force a reproducer to install something the code is explicitly prepared to
    do without -- for this repository, a CUDA build of torch, for GPU seeding that no reported
    experiment uses.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = any(
            handler.type is not None
            and ((isinstance(handler.type, ast.Name) and handler.type.id == "ImportError")
                 or (isinstance(handler.type, ast.Tuple)
                     and any(isinstance(e, ast.Name) and e.id == "ImportError"
                             for e in handler.type.elts)))
            for handler in node.handlers)
        if not catches_import_error:
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    guarded.add(inner.lineno)
    return guarded


def imported(include_optional: bool = False) -> dict[str, set[str]]:
    """Third-party top-level import name -> the files that import it."""
    standard = set(sys.stdlib_module_names)
    local = {"src", "scripts", "experiments", "tests", "configs", "data"}
    found: dict[str, set[str]] = {}
    for path in REPO_ROOT.rglob("*.py"):
        if SKIP_DIRECTORIES & set(path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not expected in this repo
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        optional_lines = _guarded_imports(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            if node.lineno in optional_lines and not include_optional:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in standard or top in local:
                    continue
                found.setdefault(top, set()).add(relative)
    return found


def test_every_third_party_import_is_pinned():
    """A clean environment must be able to run everything in the repository."""
    available = pinned()
    missing = {}
    for module, files in sorted(imported().items()):
        distribution = _normalise(DISTRIBUTION.get(module, module))
        if distribution not in available:
            missing[module] = sorted(files)[:3]
    assert not missing, (
        "imported but not pinned in requirements.txt: "
        + "; ".join(m + " (" + ", ".join(f) + ")" for m, f in missing.items())
    )


def test_no_pin_is_unused():
    """An unused pin makes a reproducer install what the results never touched."""
    used = {_normalise(DISTRIBUTION.get(module, module))
            for module in imported(include_optional=True)}
    unused = sorted(name for name in pinned()
                    if name not in used and name not in JUSTIFIED_WITHOUT_IMPORT)
    assert not unused, (
        "pinned in requirements.txt but imported nowhere: " + ", ".join(unused)
        + ". Either the code that used it was removed, or the pin belongs in the commented "
        "optional section with a reason."
    )


@pytest.mark.parametrize("name", sorted(pinned()))
def test_every_pin_has_an_exact_version(name):
    """Reproducibility needs == rather than a range."""
    line = pinned()[name]
    assert "==" in line, "requirement is not pinned to an exact version: " + line
