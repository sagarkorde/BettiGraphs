"""R1. Rolling-origin temporal evaluation.

Answers: do the family orderings (P > S > T, P+T ~ P, P > P+S+T) hold at several forecast
origins, or only on the single frozen test window, whose step 42 carries 239 of 408 illicit
transactions?

Each origin c uses train = steps 1..c-8, validation = c-7..c-1, test = c..c+3 (four steps),
so every window is out of sample and the validation width matches the frozen protocol.
Hyper-parameters are re-searched at every origin with the same budget for every family.

Usage:
    .venv/Scripts/python.exe experiments/revision/rolling_origin.py
    .venv/Scripts/python.exe experiments/revision/rolling_origin.py --quick   # smoke test
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import load_tx, logger, search_and_eval, setup_logging, write_json, OUT

ORIGINS = (30, 34, 38, 42, 46)
FAMILIES = ("P", "S", "T", "P+T", "S+T", "P+S+T")


def main(quick: bool, model: str, budget: int, n_seeds: int, n_jobs: int) -> None:
    setup_logging()
    data, _ = load_tx(n_jobs)
    seeds = range(2 if quick else n_seeds)
    origins = ORIGINS[-2:] if quick else ORIGINS
    fams = ("P", "T") if quick else FAMILIES
    rows = []
    for c in origins:
        st, y = data.steps, data.labels
        tr = (st <= c - 8)
        va = (st >= c - 7) & (st <= c - 1)
        te = (st >= c) & (st <= min(c + 3, 49))
        if (y[te] == 1).sum() < 20:
            logger.warning("origin %d: only %d illicit in test; skipped", c, (y[te] == 1).sum())
            continue
        for fam in fams:
            logger.info("origin %d family %s", c, fam)
            for r in search_and_eval(model, data.families[fam], y, tr, va, te, seeds,
                                     3 if quick else budget):
                rows.append({"origin": c, "family": fam, **r})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rolling_origin.csv", index=False)

    summary = df.groupby(["origin", "family"])["pr_auc"].mean().unstack()
    contrasts = {}
    for a, b in [("P", "T"), ("S", "T"), ("P+T", "P"), ("P", "P+S+T"), ("S+T", "S")]:
        if a in summary and b in summary:
            d = (summary[a] - summary[b]).dropna()
            contrasts[f"{a} - {b}"] = {"per_origin": d.round(4).to_dict(),
                                      "n_origins_positive": int((d > 0).sum()),
                                      "n_origins": int(d.size)}
    write_json("rolling_origin.json", {
        "model": model, "budget": budget, "seeds": list(seeds),
        "origins": list(origins), "mean_pr_auc": summary.round(4).to_dict(),
        "contrasts": contrasts})
    print(summary.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    main(a.quick, a.model, a.budget, a.n_seeds, a.n_jobs)
