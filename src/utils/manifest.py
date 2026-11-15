"""Provenance manifest for downloaded raw data.

Every raw artefact this project consumes is recorded here with its source URL, byte size,
SHA-256 digest and download timestamp. Nothing downstream is permitted to read a file that
is absent from the manifest, so that every number in the manuscript is traceable to a
specific, checksummed input.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    """True if ``path`` is an unresolved Git LFS pointer rather than real content.

    A failed ``git lfs pull`` leaves behind a ~130 byte text stub. Silently training on
    those would be far worse than crashing, so callers must check.
    """
    try:
        with path.open("rb") as handle:
            return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def load(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"schema": 1, "artifacts": {}}


def record(
    manifest_path: Path,
    key: str,
    path: Path,
    source: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record ``path`` in the manifest under ``key`` and return the entry.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if ``path`` is an unresolved Git LFS pointer.
    """
    if not path.exists():
        raise FileNotFoundError(f"cannot record missing artefact: {path}")
    if is_lfs_pointer(path):
        raise ValueError(
            f"{path} is an unresolved Git LFS pointer, not real data. "
            "Run `git lfs pull` in the dataset clone and retry."
        )

    entry: dict[str, Any] = {
        "path": path.as_posix(),
        "source": source,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "recorded_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        entry.update(extra)

    manifest = load(manifest_path)
    previous = manifest["artifacts"].get(key)
    if previous and previous.get("sha256") != entry["sha256"]:
        logger.warning(
            "checksum for %s changed: %s -> %s (the upstream file was replaced)",
            key,
            previous.get("sha256", "")[:12],
            entry["sha256"][:12],
        )
    manifest["artifacts"][key] = entry
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("recorded %s (%s bytes, sha256 %s)", key, entry["bytes"], entry["sha256"][:12])
    return entry


def verify(manifest_path: Path, key: str) -> bool:
    """Re-hash the recorded artefact for ``key`` and compare against the manifest."""
    manifest = load(manifest_path)
    entry = manifest["artifacts"].get(key)
    if entry is None:
        logger.error("%s is not in the manifest", key)
        return False
    path = Path(entry["path"])
    if not path.exists():
        logger.error("%s is recorded but missing from disk: %s", key, path)
        return False
    actual = sha256_of(path)
    if actual != entry["sha256"]:
        logger.error("%s checksum mismatch: expected %s, got %s", key, entry["sha256"], actual)
        return False
    return True
