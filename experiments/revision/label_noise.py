"""R2. Robustness of the family ordering to label noise.

Elliptic labels are heuristic. We flip a fraction of TRAINING labels (symmetric noise,
seeded) and re-run the canonical comparison with the hyper-parameters already chosen on clean
validation data; validation and test labels are untouched. The question is whether the
ordering P > S > T and the null P+T ~ P survive.

Usage:
    .venv/Scripts/python.exe experiments/revision/label_noise.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import load_tx, logger, setup_logging, write_json, OUT

RATES = (0.0, 0.05, 0.10, 0.20)
FAMILIES = ("P", "S", "T", "P+T")


def main(quick: bool, model: str, budget: int, n_seeds: int, n_jobs: int) -> None:
    from src.evaluation import splits as sp
    from src.evaluation.metrics import compute_metrics
    from src.models.tabular import fit_model, random_search
    from src.utils.seeds import set_all_seeds

    setup_logging()
    data, _ = load_tx(n_jobs)
    y = data.labels
    split = sp.temporal_split(data.steps, y)
    fit_idx = split.train | split.val
    rows = []
    fams = ("P", "T") if quick else FAMILIES
    for fam in fams:
        X = data.families[fam]
        params, _ = random_search(model, X[split.train], y[split.train], X[split.val],
                                  y[split.val], budget=3 if quick else budget, seed=0)
        for rate in (RATES[:2] if quick else RATES):
            for seed in range(2 if quick else n_seeds):
                rng = np.random.default_rng(1000 + seed)
                y_noisy = y.copy()
                idx = np.flatnonzero(split.train)  # only training rows are corrupted
                flip = idx[rng.random(idx.size) < rate]
                y_noisy[flip] = 1 - y_noisy[flip]
                set_all_seeds(seed)
                fitted = fit_model(model, X[fit_idx], y_noisy[fit_idx], seed=seed, params=params)
                met = compute_metrics(y[split.test], fitted.predict_proba(X[split.test]))
                rows.append({"family": fam, "noise": rate, "seed": seed, "pr_auc": met.pr_auc,
                             "n_flipped": int(flip.size)})
                logger.info("%s noise=%.2f seed=%d PR-AUC=%.4f", fam, rate, seed, met.pr_auc)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "label_noise.csv", index=False)
    table = df.groupby(["noise", "family"])["pr_auc"].agg(["mean", "std"]).round(4)
    ordering = {}
    for rate, g in df.groupby("noise"):
        means = g.groupby("family")["pr_auc"].mean().sort_values(ascending=False)
        ordering[str(rate)] = list(means.index)
    write_json("label_noise.json", {"model": model, "noise_rates": list(RATES),
                                    "mean_std": {f"{k[0]}|{k[1]}": v for k, v in
                                                 table.to_dict("index").items()},
                                    "ordering_by_rate": ordering})
    print(table.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    main(a.quick, a.model, a.budget, a.n_seeds, a.n_jobs)
