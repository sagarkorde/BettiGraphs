"""R3. Is topology undertuned? T after feature selection, with equal tuning budget.

T has 444 columns against 166 for P, and the ablation found a 32-column block that beat the
full signature. A reviewer can fairly ask whether T lost because a tree learner drowned in
correlated columns. Here the columns are ranked by LightGBM gain on the TRAINING window only,
k is chosen on validation from a fixed grid, and the selected T_k is then tested alone and
appended to P and to S. Selection never sees the test window.

Usage:
    .venv/Scripts/python.exe experiments/revision/feature_selected_T.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import load_tx, logger, search_and_eval, setup_logging, write_json, OUT

K_GRID = (8, 16, 32, 64, 128)


def main(quick: bool, model: str, budget: int, n_seeds: int, n_jobs: int) -> None:
    import lightgbm as lgb
    from src.evaluation import splits as sp

    setup_logging()
    data, _ = load_tx(n_jobs)
    y = data.labels
    split = sp.temporal_split(data.steps, y)
    T = np.nan_to_num(data.families["T"])
    pos = max((y[split.train] == 1).sum(), 1)
    ranker = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                scale_pos_weight=(y[split.train] == 0).sum() / pos,
                                importance_type="gain", random_state=0, verbose=-1)
    ranker.fit(T[split.train], y[split.train])
    order = np.argsort(-ranker.feature_importances_)

    val_scores = {}
    for k in K_GRID:
        cols = order[:k]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=0, verbose=-1,
                               scale_pos_weight=(y[split.train] == 0).sum() / pos)
        m.fit(T[split.train][:, cols], y[split.train])
        val_scores[k] = float(average_precision_score(
            y[split.val], m.predict_proba(T[split.val][:, cols])[:, 1]))
    k_best = max(val_scores, key=val_scores.get)
    cols = order[:k_best]
    logger.info("validation PR-AUC by k: %s -> k=%d", val_scores, k_best)

    fams = {"T_sel": T[:, cols],
            "P+T_sel": np.hstack([data.families["P"], T[:, cols]]),
            "S+T_sel": np.hstack([data.families["S"], T[:, cols]]),
            "P": data.families["P"], "S": data.families["S"], "T": data.families["T"]}
    rows = []
    seeds = range(2 if quick else n_seeds)
    for name, X in fams.items():
        for r in search_and_eval(model, X, y, split.train, split.val, split.test, seeds,
                                 3 if quick else budget):
            rows.append({"family": name, **r})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "feature_selected_T.csv", index=False)
    means = df.groupby("family")["pr_auc"].agg(["mean", "std"]).round(4)
    write_json("feature_selected_T.json", {
        "model": model, "budget_per_family": budget, "k_grid": list(K_GRID),
        "validation_pr_auc_by_k": val_scores, "k_selected": int(k_best),
        "selected_columns": [data.t_names[i] for i in cols],
        "test_pr_auc": means.to_dict("index")})
    print(means.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    main(a.quick, a.model, a.budget, a.n_seeds, a.n_jobs)
