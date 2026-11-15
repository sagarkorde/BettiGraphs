"""Gate 9: which features does a hybrid model actually use?

A feature family can lose a head-to-head comparison and still contribute inside a union. The
ablation answers "is T better than P" and "is T better than two integers"; it does not answer
"when a model is handed P, S and T together, does it reach for anything topological". That is what
this measures, using exact TreeSHAP contributions from the booster itself -- XGBoost and LightGBM
compute them natively, so the ``shap`` package is not a dependency.

Three questions, each answered by a number rather than an impression:

1. **Share of attribution by family.** Summed mean absolute SHAP over P, S and T columns. If T's
   share is near zero the model is ignoring it, whatever the ablation showed.
2. **Rank of the best topological feature.** Where the highest-attribution T column sits among all
   features. A family can hold a small total share and still contain one genuinely useful feature.
3. **A permutation check on the top topological features.** SHAP attributes what the model used, not
   what it needed; a feature can absorb attribution while being replaceable. Permuting the top T
   columns and re-scoring measures what is actually lost.

The third is included because the first two, alone, can overstate. A model will distribute
attribution across correlated columns, and 444 topological columns are heavily correlated by
construction -- adjacent Betti-curve samples are near-duplicates.

Usage::

    .venv\\Scripts\\python.exe experiments\\hybrid\\run_interpretability.py
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
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.features.pipeline import ExtractionConfig, extract_features  # noqa: E402
from src.graph_construction.graphs import build_tx_graph  # noqa: E402
from src.models.tabular import fit_model, tree_shap_values  # noqa: E402
from src.utils.seeds import set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
CACHE = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "results" / "processed"
TESTS = REPO_ROOT / "results" / "statistical_tests"
TABLES = REPO_ROOT / "tables"
FIGURES = REPO_ROOT / "figures"

logger = logging.getLogger("interpretability")
CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}


def main(radius: int, filtration: str, model_name: str, n_permute: int, seed: int) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_all_seeds(seed)
    for directory in (OUT, TESTS, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv")
    classes = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_classes.csv")
    frame = frame.merge(classes, on="txId", how="inner")
    frame["label"] = frame["class"].map(CLASS_TO_LABEL).astype(np.int8)
    labelled = frame[frame.label != -1].reset_index(drop=True)

    store = build_tx_graph(RAW_PP)
    config = ExtractionConfig(view="G_tx", radius=radius, filtration=filtration,
                              complex_dim=2, max_dim=1, include_classical=True)
    matrices = extract_features(store, labelled["txId"].to_numpy(), config, cache_dir=CACHE)

    local = [c for c in labelled.columns if c.startswith("Local_feature_")]
    aggregate = [c for c in labelled.columns if c.startswith("Aggregate_feature_")]
    p_columns = ["Time step"] + local + aggregate
    P = labelled[p_columns].to_numpy(dtype=np.float32)
    S, T = matrices.classical, matrices.topological

    X = np.hstack([P, S, T])
    names = (p_columns
             + [f"S::{n}" for n in matrices.classical_names]
             + [f"T::{n}" for n in matrices.topological_names])
    family_of = np.array(["P"] * P.shape[1] + ["S"] * S.shape[1] + ["T"] * T.shape[1])
    assert len(names) == X.shape[1] == family_of.size

    y = labelled["label"].to_numpy()
    split = sp.temporal_split(labelled["Time step"].to_numpy(), y)
    X_fit = np.vstack([X[split.train], X[split.val]])
    y_fit = np.concatenate([y[split.train], y[split.val]])
    X_test, y_test = X[split.test], y[split.test]

    logger.info("fitting %s on P+S+T: %d features (%d P, %d S, %d T)",
                model_name, X.shape[1], P.shape[1], S.shape[1], T.shape[1])
    fitted = fit_model(model_name, X_fit, y_fit, seed=seed)
    baseline = compute_metrics(y_test, fitted.predict_proba(X_test))
    logger.info("baseline test PR-AUC %.4f", baseline.pr_auc)

    contributions = tree_shap_values(fitted, X_test)
    if contributions is None:
        logger.error("%s exposes no native TreeSHAP", model_name)
        return 1
    # The booster appends a bias column.
    contributions = np.asarray(contributions)[:, : X.shape[1]]
    importance = np.abs(contributions).mean(axis=0)

    total = float(importance.sum()) or 1.0
    shares = {fam: float(importance[family_of == fam].sum() / total) for fam in ("P", "S", "T")}
    order = np.argsort(-importance)
    ranks = {name: int(rank) for rank, name in enumerate(np.array(names)[order], start=1)}

    topological_ranked = [(names[i], float(importance[i])) for i in order if family_of[i] == "T"]
    best_topological = topological_ranked[0] if topological_ranked else ("none", 0.0)

    logger.info("attribution share: P %.3f  S %.3f  T %.3f",
                shares["P"], shares["S"], shares["T"])
    logger.info("best topological feature %s ranks %d of %d",
                best_topological[0], ranks[best_topological[0]], len(names))

    # --- permutation check on the top topological features ---
    rng = np.random.default_rng(seed)
    top_t = [i for i in order if family_of[i] == "T"][:n_permute]
    X_permuted = X_test.copy()
    for column in top_t:
        X_permuted[:, column] = X_permuted[rng.permutation(X_permuted.shape[0]), column]
    permuted = compute_metrics(y_test, fitted.predict_proba(X_permuted))

    # And the same count of top P features, as a scale reference: a drop is only meaningful
    # relative to what permuting the family the model does rely on costs.
    top_p = [i for i in order if family_of[i] == "P"][:n_permute]
    X_permuted_p = X_test.copy()
    for column in top_p:
        X_permuted_p[:, column] = X_permuted_p[rng.permutation(X_permuted_p.shape[0]), column]
    permuted_p = compute_metrics(y_test, fitted.predict_proba(X_permuted_p))

    logger.info("permuting the top %d T features: PR-AUC %.4f -> %.4f (%+.4f)",
                n_permute, baseline.pr_auc, permuted.pr_auc, permuted.pr_auc - baseline.pr_auc)
    logger.info("permuting the top %d P features: PR-AUC %.4f -> %.4f (%+.4f)",
                n_permute, baseline.pr_auc, permuted_p.pr_auc, permuted_p.pr_auc - baseline.pr_auc)

    top_overall = [{"rank": rank, "feature": names[i], "family": str(family_of[i]),
                    "mean_abs_shap": float(importance[i])}
                   for rank, i in enumerate(order[:30], start=1)]

    payload = {
        "model": model_name, "radius": radius, "filtration": filtration,
        "n_features": int(X.shape[1]),
        "family_sizes": {"P": int(P.shape[1]), "S": int(S.shape[1]), "T": int(T.shape[1])},
        "baseline_test_pr_auc": baseline.pr_auc,
        "attribution_share": shares,
        "attribution_share_per_feature": {
            fam: float(importance[family_of == fam].sum() / max((family_of == fam).sum(), 1) / total)
            for fam in ("P", "S", "T")
        },
        "best_topological_feature": {"name": best_topological[0],
                                     "rank": ranks[best_topological[0]],
                                     "mean_abs_shap": best_topological[1]},
        "permutation": {
            "n_permuted": n_permute,
            "topological": {"pr_auc": permuted.pr_auc,
                            "delta": permuted.pr_auc - baseline.pr_auc},
            "provided": {"pr_auc": permuted_p.pr_auc,
                         "delta": permuted_p.pr_auc - baseline.pr_auc},
        },
        "top_30_features": top_overall,
        "note": (
            "SHAP attributes what the model used, not what it needed. The 444 topological columns "
            "are heavily correlated by construction -- adjacent Betti-curve samples are near "
            "duplicates -- so attribution spreads across them and can overstate. The permutation "
            "check measures what is actually lost when they are destroyed."
        ),
    }
    (TESTS / "interpretability.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Table 11 — Interpretability of the hybrid model",
        "",
        f"Exact TreeSHAP from the {model_name} booster, fitted on P+S+T "
        f"({X.shape[1]} features: {P.shape[1]} provided, {S.shape[1]} structural, "
        f"{T.shape[1]} topological), evaluated on the held-out test window.",
        "",
        "## Share of attribution by family",
        "",
        "| Family | Features | Share of total \\|SHAP\\| | Share per feature |",
        "|---|---|---|---|",
    ]
    for fam, size in (("P", P.shape[1]), ("S", S.shape[1]), ("T", T.shape[1])):
        lines.append(
            f"| {fam} | {size} | {shares[fam]:.3f} | "
            f"{payload['attribution_share_per_feature'][fam]:.2e} |")
    lines += [
        "",
        f"Best topological feature: `{best_topological[0]}`, ranked "
        f"**{ranks[best_topological[0]]} of {len(names)}**.",
        "",
        "## Permutation check",
        "",
        "SHAP reports what the model used, not what it needed. Permuting a feature and re-scoring "
        "measures what is actually lost.",
        "",
        "| Permuted | Test PR-AUC | Change |",
        "|---|---|---|",
        f"| nothing (baseline) | {baseline.pr_auc:.4f} | — |",
        f"| top {n_permute} topological | {permuted.pr_auc:.4f} | "
        f"{permuted.pr_auc - baseline.pr_auc:+.4f} |",
        f"| top {n_permute} provided | {permuted_p.pr_auc:.4f} | "
        f"{permuted_p.pr_auc - baseline.pr_auc:+.4f} |",
        "",
        "## Top 15 features overall",
        "",
        "| Rank | Feature | Family | mean \\|SHAP\\| |",
        "|---|---|---|---|",
    ]
    for row in top_overall[:15]:
        lines.append(f"| {row['rank']} | `{row['feature']}` | {row['family']} | "
                     f"{row['mean_abs_shap']:.4g} |")
    lines.append("")
    (TABLES / "table11_interpretability.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== attribution share ===")
    for fam in ("P", "S", "T"):
        print(f"  {fam}: {shares[fam]:.3f} over {int((family_of == fam).sum())} features")
    print(f"\nbest topological feature: {best_topological[0]} at rank "
          f"{ranks[best_topological[0]]} of {len(names)}")
    print(f"\npermuting top {n_permute} T: {baseline.pr_auc:.4f} -> {permuted.pr_auc:.4f} "
          f"({permuted.pr_auc - baseline.pr_auc:+.4f})")
    print(f"permuting top {n_permute} P: {baseline.pr_auc:.4f} -> {permuted_p.pr_auc:.4f} "
          f"({permuted_p.pr_auc - baseline.pr_auc:+.4f})")
    logger.info("wrote %s", TABLES / "table11_interpretability.md")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--model", default="lightgbm", dest="model_name")
    parser.add_argument("--n-permute", type=int, default=20, dest="n_permute")
    parser.add_argument("--seed", type=int, default=0)
    raise SystemExit(main(**vars(parser.parse_args())))
