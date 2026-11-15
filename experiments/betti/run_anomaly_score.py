"""Contribution 4: the topological anomaly score, evaluated on both graph views.

One-class by design: the scorer sees only **licit training-window entities**. A detector that needs
illicit examples is not solving the problem an analyst has, which is recognising behaviour unlike
the legitimate norm without knowing in advance what the illicit behaviour looks like.

Run on both views, because the rest of this study found they behave differently: the transaction
graph is too sparse for topological descriptors to say much, while the address graph is not. If
that reading is right, the one-class score should fail on the first and do something on the second.
Running it on only the favourable view would be the obvious way to get a misleading answer.

Two comparators, both fitted the same way on the same entities:

* the classical structural signature, so the comparison is between representations rather than
  between a topological detector and no detector at all;
* a shuffled-score floor, because precision at a fixed alert budget on a 5% base rate looks
  impressive before you check what chance gives.

Reported as precision and lift at realistic alert budgets as well as average precision: an analyst
has a finite number of cases to review, and an AUC does not tell them how many of their first
hundred alerts are worth opening.

Usage::

    .venv\\Scripts\\python.exe experiments\\betti\\run_anomaly_score.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import splits as sp  # noqa: E402
from src.features.pipeline import ExtractionConfig, extract_features  # noqa: E402
from src.graph_construction.graphs import build_address_graph, build_tx_graph  # noqa: E402
from src.models.anomaly import SCORERS, evaluate_anomaly_scores, fit_anomaly_scorer  # noqa: E402
from src.utils.seeds import set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
CACHE = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "results" / "processed"
TESTS = REPO_ROOT / "results" / "statistical_tests"
TABLES = REPO_ROOT / "tables"

logger = logging.getLogger("anomaly")
TX_CLASS = {1: 1, 2: 0, 3: -1}
WALLET_CLASS = {1: 1, 2: 0, 3: -1}
BUDGETS = (50, 100, 500, 1000)


def transaction_view(radius: int, filtration: str, n_jobs: int):
    frame = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv",
                        usecols=["txId", "Time step"])
    classes = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_classes.csv")
    frame = frame.merge(classes, on="txId", how="inner")
    frame["label"] = frame["class"].map(TX_CLASS).astype(np.int8)
    frame = frame[frame.label != -1].reset_index(drop=True)

    store = build_tx_graph(RAW_PP)
    config = ExtractionConfig(view="G_tx", radius=radius, filtration=filtration,
                              complex_dim=2, max_dim=1, include_classical=True)
    matrices = extract_features(store, frame["txId"].to_numpy(), config,
                               n_jobs=n_jobs, cache_dir=CACHE)
    split = sp.temporal_split(frame["Time step"].to_numpy(), frame["label"].to_numpy())
    return "G_tx", matrices, frame["label"].to_numpy(), split


def address_view(radius: int, filtration: str, n_jobs: int, n_sample: int, seed: int):
    features = pd.read_csv(RAW_PP / "Actors Dataset" / "wallets_features.csv",
                           usecols=["address", "Time step"])
    classes = pd.read_csv(RAW_PP / "Actors Dataset" / "wallets_classes.csv")
    frame = features.merge(classes, on="address", how="inner")
    frame["label"] = frame["class"].map(WALLET_CLASS).fillna(-1).astype(np.int8)
    frame = frame[frame.label != -1].sort_values("Time step").drop_duplicates("address")
    frame = frame.reset_index(drop=True)

    # The full labelled set is 265,354 addresses; a stratified subsample keeps the run affordable
    # while preserving the base rate the alert-budget metrics depend on.
    rng = np.random.default_rng(seed)
    if len(frame) > n_sample:
        parts = []
        for _, group in frame.groupby("label"):
            take = max(1, int(round(n_sample * len(group) / len(frame))))
            parts.append(group.iloc[rng.choice(len(group), size=min(take, len(group)),
                                               replace=False)])
        frame = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    store = build_address_graph(RAW_PP)
    config = ExtractionConfig(view="G_addr", radius=radius, filtration=filtration,
                              complex_dim=2, max_dim=1, include_classical=True)
    matrices = extract_features(store, frame["address"].to_numpy(), config,
                               n_jobs=n_jobs, cache_dir=CACHE)
    split = sp.temporal_split(frame["Time step"].to_numpy(), frame["label"].to_numpy())
    return "G_addr", matrices, frame["label"].to_numpy(), split


def run_view(name: str, matrices, labels, split, seed: int) -> list[dict]:
    """Fit every scorer on licit training entities and score the held-out test window."""
    reference = split.train & (labels == 0)
    logger.info("%s: %d licit reference entities, %d test entities (%d illicit)",
                name, int(reference.sum()), int(split.test.sum()),
                int((labels[split.test] == 1).sum()))

    blocks = {"T": matrices.topological}
    if matrices.classical is not None:
        blocks["S"] = matrices.classical

    rows: list[dict] = []
    for block_name, block in blocks.items():
        for method in SCORERS:
            set_all_seeds(seed)
            try:
                scorer = fit_anomaly_scorer(block[reference], method=method, seed=seed)
                scores = scorer.score(block[split.test])
                metrics = evaluate_anomaly_scores(scores, labels[split.test], budgets=BUDGETS)
            except Exception as exc:
                logger.error("%s/%s/%s failed: %s", name, block_name, method, exc)
                continue
            rows.append({"view": name, "features": block_name, "method": method, **metrics})
            logger.info("%s %-2s %-16s PR-AUC %.4f  P@100 %.4f  lift@100 %.2f",
                        name, block_name, method, metrics["pr_auc"],
                        metrics.get("precision_at_100", float("nan")),
                        metrics.get("lift_at_100", float("nan")))

    # Shuffled floor: the same labels against random scores, so the alert-budget numbers can be
    # read against what chance delivers at this base rate.
    rng = np.random.default_rng(seed)
    floor = evaluate_anomaly_scores(rng.random(int(split.test.sum())), labels[split.test],
                                    budgets=BUDGETS)
    rows.append({"view": name, "features": "random", "method": "floor", **floor})
    logger.info("%s %-2s %-16s PR-AUC %.4f  P@100 %.4f (floor)", name, "--", "random",
                floor["pr_auc"], floor.get("precision_at_100", float("nan")))
    return rows


def main(radius: int, filtration: str, n_jobs: int, wallet_sample: int, seed: int,
         views: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_all_seeds(seed)
    for directory in (OUT, TESTS, TABLES):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if "tx" in views:
        name, matrices, labels, split = transaction_view(radius, filtration, n_jobs)
        rows += run_view(name, matrices, labels, split, seed)
    if "addr" in views:
        name, matrices, labels, split = address_view(2, filtration, n_jobs, wallet_sample, seed)
        rows += run_view(name, matrices, labels, split, seed)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "anomaly_scores.csv", index=False)

    lines = [
        "# Table 12 — One-class topological anomaly score",
        "",
        "Fitted on **licit training-window entities only**; the scorer never sees an illicit",
        "example. Evaluated on the held-out test window. The classical structural signature is",
        "fitted identically, so the comparison is between representations rather than between a",
        "detector and nothing, and a shuffled-score floor is included because precision at a fixed",
        "alert budget looks impressive on an imbalanced pool before you check what chance gives.",
        "",
        "| View | Features | Method | PR-AUC | ROC-AUC | P@100 | Lift@100 | P@1000 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['view']} | {row['features']} | `{row['method']}` | {row['pr_auc']:.4f} | "
            f"{row['roc_auc']:.4f} | {row.get('precision_at_100', float('nan')):.4f} | "
            f"{row.get('lift_at_100', float('nan')):.2f} | "
            f"{row.get('precision_at_1000', float('nan')):.4f} |"
        )
    lines += ["", f"Base rate in the test window: "
                  f"{frame['base_rate'].iloc[0]:.4f} (transaction view).", ""]
    (TABLES / "table12_anomaly_score.md").write_text("\n".join(lines), encoding="utf-8")
    (TESTS / "anomaly_scores.json").write_text(
        json.dumps({"rows": rows, "budgets": list(BUDGETS), "radius": radius,
                    "filtration": filtration, "seed": seed}, indent=2, default=str),
        encoding="utf-8")

    print("\n=== one-class topological anomaly score ===")
    cols = ["view", "features", "method", "pr_auc", "precision_at_100", "lift_at_100"]
    print(frame[[c for c in cols if c in frame.columns]].round(4).to_string(index=False))
    logger.info("wrote %s", TABLES / "table12_anomaly_score.md")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    parser.add_argument("--wallet-sample", type=int, default=20000, dest="wallet_sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", nargs="+", default=["tx", "addr"], choices=["tx", "addr"])
    raise SystemExit(main(**vars(parser.parse_args())))
