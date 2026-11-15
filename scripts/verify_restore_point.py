"""Check the working tree's artefacts against the checksums recorded in docs/RESTORE_POINTS.md.

A restore point is only useful if you can tell whether you are at it. This reads the SHA-256 table
out of that document and compares it with the files on disk, so a drifted or partially re-run
artefact is visible immediately -- before it is discovered through a number that no longer matches
the manuscript.

Three outcomes per artefact: matches the restore point, differs from it, or is missing. A
difference is not necessarily wrong -- re-running an experiment is supposed to change its output --
but it does mean the manuscript was not built from the file now on disk.

Usage::

    .venv\\Scripts\\python.exe scripts\\verify_restore_point.py
    .venv\\Scripts\\python.exe scripts\\verify_restore_point.py --update   # after a deliberate re-run
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD = REPO_ROOT / "docs" / "RESTORE_POINTS.md"
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64}|MISSING)`\s*\|\s*$", re.M)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded() -> list[tuple[str, str]]:
    if not RECORD.exists():
        raise SystemExit("no restore-point record at " + str(RECORD))
    return ROW.findall(RECORD.read_text(encoding="utf-8"))


def main(update: bool) -> int:
    rows = recorded()
    if not rows:
        raise SystemExit("no checksum rows found in " + RECORD.name)

    matched, differed, missing = [], [], []
    replacements: dict[str, str] = {}
    for path_text, expected in rows:
        path = REPO_ROOT / path_text
        if not path.exists():
            missing.append(path_text)
            replacements[path_text] = "MISSING"
            continue
        actual = digest(path)
        replacements[path_text] = actual
        (matched if actual == expected else differed).append(path_text)

    for path_text in matched:
        print("  match     " + path_text)
    for path_text in differed:
        print("  DIFFERS   " + path_text)
    for path_text in missing:
        print("  MISSING   " + path_text)

    print("\n" + str(len(matched)) + " match, " + str(len(differed)) + " differ, "
          + str(len(missing)) + " missing, of " + str(len(rows)) + " recorded")

    if update:
        text = RECORD.read_text(encoding="utf-8")
        for path_text, actual in replacements.items():
            text = re.sub(
                r"(\|\s*`" + re.escape(path_text) + r"`\s*\|\s*`)(?:[0-9a-f]{64}|MISSING)(`\s*\|)",
                lambda m: m.group(1) + actual + m.group(2), text)
        RECORD.write_text(text, encoding="utf-8")
        print("\nupdated " + RECORD.name + " to the current artefacts")
        return 0

    if differed or missing:
        print("\nThe working tree is not at the recorded restore point. If that is deliberate --")
        print("an experiment was re-run -- re-record it with --update and say so in the commit.")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="re-record the checksums from the current artefacts")
    raise SystemExit(main(parser.parse_args().update))
