"""Download the Elliptic++ dataset from its official GitHub repository.

Source
------
``https://github.com/git-disl/EllipticPlusPlus``, accompanying Elmougy and Liu (2023),
"Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial
Forensics" (ACM SIGKDD). The CSVs are tracked with Git LFS, so a plain HTTP fetch of the
blob returns a ~130 byte pointer stub rather than data; this script clones with LFS and then
asserts that every CSV is real content.

Why this dataset is required
----------------------------
The original Elliptic release has anonymised transaction-level nodes and carries no address
or wallet identity whatsoever. The address-level graphs here -- ``AddrAddr_edgelist.csv``,
``AddrTx_edgelist.csv``, ``TxAddr_edgelist.csv`` -- together with ``wallets_classes.csv`` are
what make the wallet-relationship research question answerable at all.

Licence
-------
The repository carries no LICENSE file, so redistribution terms are unspecified. This project
never commits the data and fetches it from the official repository instead.

Usage
-----
    python data/download_scripts/get_elliptic_plus_plus.py
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.manifest import is_lfs_pointer, record  # noqa: E402

CLONE_URL = "https://github.com/git-disl/EllipticPlusPlus.git"
DEST = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
MANIFEST = REPO_ROOT / "data" / "raw" / "MANIFEST.json"

# Relative paths inside the upstream repository -> manifest keys.
WANTED = {
    "Transactions Dataset/txs_features.csv": "elliptic_pp/txs_features",
    "Transactions Dataset/txs_edgelist.csv": "elliptic_pp/txs_edgelist",
    "Transactions Dataset/txs_classes.csv": "elliptic_pp/txs_classes",
    "Actors Dataset/wallets_features.csv": "elliptic_pp/wallets_features",
    "Actors Dataset/wallets_classes.csv": "elliptic_pp/wallets_classes",
    "Actors Dataset/wallets_features_classes_combined.csv": "elliptic_pp/wallets_combined",
    "Actors Dataset/AddrAddr_edgelist.csv": "elliptic_pp/addr_addr_edgelist",
    "Actors Dataset/AddrTx_edgelist.csv": "elliptic_pp/addr_tx_edgelist",
    "Actors Dataset/TxAddr_edgelist.csv": "elliptic_pp/tx_addr_edgelist",
}

logger = logging.getLogger("get_elliptic_pp")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("stdout: %s", result.stdout[-2000:])
        logger.error("stderr: %s", result.stderr[-2000:])
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if result.stdout.strip():
        logger.debug("stdout: %s", result.stdout[-1000:])


def main(force: bool = False) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if shutil.which("git") is None:
        raise RuntimeError("git is not on PATH")
    if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        raise RuntimeError("git-lfs is not installed; the CSVs cannot be resolved without it")

    if DEST.exists() and force:
        logger.info("removing existing clone at %s", DEST)
        shutil.rmtree(DEST, ignore_errors=True)

    if not (DEST / ".git").exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        # Shallow clone, LFS content fetched explicitly afterwards so failures are visible.
        _run(["git", "clone", "--depth", "1", CLONE_URL, str(DEST)])
    else:
        logger.info("clone already present at %s", DEST)

    _run(["git", "lfs", "install", "--local"], cwd=DEST)
    _run(["git", "lfs", "pull"], cwd=DEST)

    missing: list[str] = []
    pointers: list[str] = []
    for rel, key in WANTED.items():
        path = DEST / rel
        if not path.exists():
            missing.append(rel)
            continue
        if is_lfs_pointer(path):
            pointers.append(rel)
            continue
        record(
            MANIFEST,
            key=key,
            path=path,
            source=f"{CLONE_URL} :: {rel}",
            extra={
                "dataset": "Elliptic++",
                "citation": "Elmougy and Liu 2023, ACM SIGKDD, doi:10.1145/3580305.3599803",
                "licence": "unspecified upstream (no LICENSE file); not redistributed here",
            },
        )

    if missing:
        logger.error("missing from the clone: %s", missing)
    if pointers:
        logger.error("still unresolved LFS pointers: %s", pointers)
    if missing or pointers:
        return 1

    logger.info("Elliptic++ download complete; manifest at %s", MANIFEST)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="delete and re-clone")
    raise SystemExit(main(**vars(parser.parse_args())))
