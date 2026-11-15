"""Record the temporal split sizes as an artefact.

The split boundaries are frozen in the pre-registration and the resulting counts appear in the
manuscript's Table 3, but they were only ever logged, never saved. That left three numbers in the
prose with no artefact behind them -- the validation row of the split table -- which the manuscript's
literal check correctly refuses to accept.

Cheap: it reads two label files and applies the same split function the experiments use, so the
counts cannot diverge from the ones the runs saw.

Usage::

    .venv\\Scripts\\python.exe scripts\\record_splits.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import splits as sp  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
OUT = REPO_ROOT / "results" / "processed"
TX_CLASS = {1: 1, 2: 0, 3: -1}

logger = logging.getLogger("splits")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv",
                        usecols=["txId", "Time step"])
    classes = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_classes.csv")
    frame = frame.merge(classes, on="txId", how="inner")
    frame["label"] = frame["class"].map(TX_CLASS).astype(np.int8)
    frame = frame[frame.label != -1].reset_index(drop=True)

    steps = frame["Time step"].to_numpy()
    labels = frame["label"].to_numpy()
    split = sp.temporal_split(steps, labels)

    payload: dict = {"n_labelled": int(len(frame))}
    for name, mask in (("train", split.train), ("val", split.val), ("test", split.test)):
        selected = labels[mask]
        payload[name] = {
            "n": int(mask.sum()),
            "n_illicit": int((selected == 1).sum()),
            "n_licit": int((selected == 0).sum()),
            "base_rate": float((selected == 1).mean()) if mask.sum() else 0.0,
            "first_step": int(steps[mask].min()),
            "last_step": int(steps[mask].max()),
        }
        logger.info("%-5s steps %d-%d: n=%d illicit=%d licit=%d (%.2f%%)", name,
                    payload[name]["first_step"], payload[name]["last_step"], payload[name]["n"],
                    payload[name]["n_illicit"], payload[name]["n_licit"],
                    100 * payload[name]["base_rate"])

    # Per-step illicit counts, which the regime-change paragraph quotes.
    per_step = (frame[frame.label == 1].groupby("Time step").size()
                .reindex(range(1, int(steps.max()) + 1), fill_value=0))
    payload["illicit_per_step"] = {str(int(k)): int(v) for k, v in per_step.items()}
    pre = per_step.loc[:42]
    payload["pre_shift_illicit_per_step"] = {
        "min": int(pre.min()), "max": int(pre.max()), "median": float(pre.median()),
        "first_step": 1, "last_step": 42,
    }
    payload["post_shift_illicit_per_step"] = {
        str(step): int(per_step.loc[step]) for step in range(43, int(steps.max()) + 1)
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "splits.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
