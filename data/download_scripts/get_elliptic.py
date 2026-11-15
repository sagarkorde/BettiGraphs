"""Download the Elliptic Bitcoin dataset from its public mirror.

Source
------
The dataset was released by Elliptic and accompanies Weber et al. (2019),
"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for
Financial Forensics" (arXiv:1908.02591). The three CSVs are served publicly, zipped, by the
PyTorch Geometric data CDN, which is what ``torch_geometric.datasets.EllipticBitcoinDataset``
itself fetches. No account or credential is required.

Licence
-------
The original release permits research use; redistribution is not granted. This project
therefore never commits the data. ``data/raw/`` is git-ignored and this script re-fetches
from the official mirror on any clean checkout.

Usage
-----
    python data/download_scripts/get_elliptic.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.manifest import record  # noqa: E402

BASE_URL = "https://data.pyg.org/datasets/elliptic"
FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
)

DEST = REPO_ROOT / "data" / "raw" / "elliptic"
MANIFEST = REPO_ROOT / "data" / "raw" / "MANIFEST.json"

logger = logging.getLogger("get_elliptic")


def _download(url: str, target: Path) -> None:
    logger.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https host
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        with target.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                written += len(chunk)
        if total and written != total:
            raise IOError(f"short read for {url}: got {written} of {total} bytes")
    logger.info("  -> %s (%d bytes)", target.name, target.stat().st_size)


def main(force: bool = False) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    DEST.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        csv_path = DEST / name
        if csv_path.exists() and not force:
            logger.info("%s already present, skipping download", name)
        else:
            zip_path = DEST / f"{name}.zip"
            url = f"{BASE_URL}/{name}.zip"
            _download(url, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                members = archive.namelist()
                logger.info("  extracting %s", members)
                archive.extractall(DEST)
            zip_path.unlink()
            # The archive may nest the CSV inside a directory; flatten if so.
            if not csv_path.exists():
                matches = list(DEST.rglob(name))
                if not matches:
                    raise FileNotFoundError(f"{name} not found after extracting {url}")
                matches[0].replace(csv_path)

        record(
            MANIFEST,
            key=f"elliptic/{name}",
            path=csv_path,
            source=f"{BASE_URL}/{name}.zip",
            extra={
                "dataset": "Elliptic Bitcoin Dataset",
                "citation": "Weber et al. 2019, arXiv:1908.02591",
                "licence": "research use permitted; redistribution not granted",
            },
        )

    logger.info("Elliptic download complete; manifest at %s", MANIFEST)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    raise SystemExit(main(**vars(parser.parse_args())))
