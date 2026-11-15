"""Gates 5-7: the transaction-level feature-family comparison.

Runs the pre-registered comparison across feature families on the Elliptic++ transaction task,
under the frozen temporal split, and writes every per-(model, family, seed) result to
``results/raw/`` for the statistical analysis to consume.

Protocol, following ``docs/PREREGISTRATION.md``:

1. Random search of an **equal** budget per (model, family), scored on validation illicit PR-AUC.
2. The chosen configuration is then refitted on train+validation for each of the 10 fixed seeds.
3. The test split is scored once per (model, family, seed) and never used for any choice.

Feature families
----------------
``P`` is ``P166`` -- the exact Elliptic-equivalent subset, for comparability with published work.
``Pplus`` adds the 17 named Elliptic++ columns. ``S`` is the classical structural family and ``T``
the topological one, both computed on the same ego-subgraph.

Usage::

    .venv\\Scripts\\python.exe experiments\\baseline\\run_transaction_experiment.py --quick
    .venv\\Scripts\\python.exe experiments\\baseline\\run_transaction_experiment.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import splits as sp  # noqa: E402
from src.evaluation.metrics import compute_metrics, lift_at_k, per_step_pr_auc  # noqa: E402
from src.features.pipeline import ExtractionConfig, extract_features  # noqa: E402
from src.graph_construction.graphs import build_tx_graph  # noqa: E402
from src.models.tabular import SEARCH_BUDGET, fit_model, random_search  # noqa: E402
from src.utils.seeds import SEEDS, RunProvenance, set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
CACHE = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "results" / "raw"

logger = logging.getLogger("tx_experiment")

CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}


def load_transaction_table() -> pd.DataFrame:
    """Load transaction features and labels, returning one row per labelled transaction."""
    features = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv")
    classes = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_classes.csv")
    frame = features.merge(classes, on="txId", how="inner")
    frame["label"] = frame["class"].map(CLASS_TO_LABEL).astype(np.int8)
    logger.info("loaded %d transactions, %d labelled", len(frame), int((frame.label != -1).sum()))
    return frame


def family_matrices(
    frame: pd.DataFrame, topological: np.ndarray, classical: np.ndarray
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Assemble the feature families as (matrix, names) pairs."""
    local = [c for c in frame.columns if c.startswith("Local_feature_")]
    aggregate = [c for c in frame.columns if c.startswith("Aggregate_feature_")]
    named = ["in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
             "num_input_addresses", "num_output_addresses",
             "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
             "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total"]
    named = [c for c in named if c in frame.columns]

    p166_cols = ["Time step"] + local + aggregate
    p166 = frame[p166_cols].to_numpy(dtype=np.float32)
    pplus = frame[p166_cols + named].to_numpy(dtype=np.float32)

    families: dict[str, tuple[np.ndarray, list[str]]] = {
        "P": (p166, p166_cols),
        "Pplus": (pplus, p166_cols + named),
        "S": (classical, [f"S::{i}" for i in range(classical.shape[1])]),
        "T": (topological, [f"T::{i}" for i in range(topological.shape[1])]),
    }
    families["P+S"] = (np.hstack([p166, classical]), families["P"][1] + families["S"][1])
    families["P+T"] = (np.hstack([p166, topological]), families["P"][1] + families["T"][1])
    families["S+T"] = (np.hstack([classical, topological]), families["S"][1] + families["T"][1])
    families["P+S+T"] = (
        np.hstack([p166, classical, topological]),
        families["P"][1] + families["S"][1] + families["T"][1],
    )
    return families


def main(
    quick: bool,
    radius: int,
    filtration: str,
    complex_dim: int,
    budget: int,
    models: list[str],
    n_jobs: int,
    tag: str,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    frame = load_transaction_table()
    labelled = frame[frame.label != -1].reset_index(drop=True)

    store = build_tx_graph(RAW_PP)
    config = ExtractionConfig(
        view="G_tx", radius=radius, filtration=filtration, complex_dim=complex_dim,
        max_dim=1, max_nodes=25_000, include_classical=True,
    )
    matrices = extract_features(
        store, labelled["txId"].to_numpy(), config, n_jobs=n_jobs, cache_dir=CACHE,
    )
    logger.info("feature extraction summary: %s", json.dumps(matrices.summary(), indent=2))

    families = family_matrices(labelled, matrices.topological, matrices.classical)
    time_steps = labelled["Time step"].to_numpy()
    labels = labelled["label"].to_numpy()
    split = sp.temporal_split(time_steps, labels)

    seeds = SEEDS[:3] if quick else SEEDS
    family_names = ["P", "S", "T"] if quick else list(families)
    logger.info("running models=%s families=%s seeds=%s budget=%d",
                models, family_names, seeds, budget)

    suffix = f"{tag}_r{radius}_{filtration}_cd{complex_dim}" + ("_quick" if quick else "")
    results_path = OUT / f"transaction_results_{suffix}.csv"
    per_step_path = OUT / f"transaction_per_step_{suffix}.csv"

    # Resume support. A full run is several hours; writing only at the end means an interruption
    # loses everything, which is not an acceptable design for an unattended job. Completed
    # (model, family) cells are appended as they finish and skipped on a restart.
    records: list[dict] = []
    per_step_records: list[dict] = []
    completed: set[tuple[str, str]] = set()
    if results_path.exists():
        previous = pd.read_csv(results_path)
        records = previous.to_dict("records")
        completed = set(zip(previous["model"], previous["family"]))
        logger.info("resuming: %d cells already complete %s", len(completed), sorted(completed))
        if per_step_path.exists():
            per_step_records = pd.read_csv(per_step_path).to_dict("records")

    def checkpoint() -> None:
        pd.DataFrame(records).to_csv(results_path, index=False)
        if per_step_records:
            pd.DataFrame(per_step_records).to_csv(per_step_path, index=False)

    started = time.perf_counter()

    for family_name in family_names:
        X, names = families[family_name]
        X_train, y_train = X[split.train], labels[split.train]
        X_val, y_val = X[split.val], labels[split.val]
        X_test, y_test = X[split.test], labels[split.test]
        steps_test = time_steps[split.test]

        # Train+validation refit set, used only after the search is complete.
        X_fit = np.vstack([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])

        # H6 needs a balanced view of the regime change, and the primary split cannot give one:
        # its test window starts at step 42, so only a single step sits before the shift at 43
        # against seven after it. A one-step baseline is too thin to carry the primary hypothesis.
        #
        # So H6 is additionally evaluated on a model fitted to the training window ALONE and scored
        # per step across 35-49, which puts 8 unseen pre-shift steps against 7 unseen post-shift
        # steps. Both sides are out-of-sample and come from the same split, so the comparison
        # isolates the regime change rather than confounding it with the split boundary.
        h6_mask = (time_steps >= sp.VAL_STEPS[0]) & (time_steps <= sp.TEST_STEPS[1]) & (labels != -1)
        X_h6, y_h6, steps_h6 = X[h6_mask], labels[h6_mask], time_steps[h6_mask]

        for model_name in models:
            if (model_name, family_name) in completed:
                logger.info("skipping %s / %s: already complete", model_name, family_name)
                continue
            logger.info("=== %s / %s : searching (budget %d) ===", model_name, family_name, budget)
            try:
                best_params, best_val = random_search(
                    model_name, X_train, y_train, X_val, y_val, budget=budget, seed=0,
                )
            except Exception as exc:
                logger.error("search failed for %s/%s: %s", model_name, family_name, exc)
                continue

            for seed in seeds:
                set_all_seeds(seed)
                try:
                    fitted = fit_model(model_name, X_fit, y_fit, seed=seed, params=best_params)
                    scores = fitted.predict_proba(X_test)
                    metrics = compute_metrics(y_test, scores)
                except Exception as exc:
                    logger.error("fit/eval failed for %s/%s seed %d: %s",
                                 model_name, family_name, seed, exc)
                    continue

                record = {
                    "model": model_name,
                    "family": family_name,
                    "seed": seed,
                    "n_features": int(X.shape[1]),
                    "validation_pr_auc_at_search": best_val,
                    "best_params": json.dumps(best_params, default=str),
                    "fit_seconds": fitted.fit_seconds,
                    "lift_at_100": lift_at_k(y_test, scores, 100),
                    "lift_at_1000": lift_at_k(y_test, scores, 1000),
                    **metrics.as_dict(),
                }
                records.append(record)
                logger.info(
                    "%s/%s seed=%d  test PR-AUC=%.4f  ROC-AUC=%.4f  illicit-F1=%.4f",
                    model_name, family_name, seed, metrics.pr_auc, metrics.roc_auc,
                    metrics.illicit_f1,
                )

                for step, stats in per_step_pr_auc(y_test, scores, steps_test).items():
                    per_step_records.append({
                        "model": model_name, "family": family_name, "seed": seed,
                        "time_step": step, "protocol": "train+val", **stats,
                    })

                # H6 protocol: fit on the training window alone, score every step from 35 to 49.
                try:
                    h6_model = fit_model(model_name, X_train, y_train, seed=seed,
                                         params=best_params)
                    h6_scores = h6_model.predict_proba(X_h6)
                    for step, stats in per_step_pr_auc(y_h6, h6_scores, steps_h6).items():
                        per_step_records.append({
                            "model": model_name, "family": family_name, "seed": seed,
                            "time_step": step, "protocol": "train_only", **stats,
                        })
                except Exception as exc:
                    logger.error("H6 fit failed for %s/%s seed %d: %s",
                                 model_name, family_name, seed, exc)

            checkpoint()
            logger.info("checkpointed after %s / %s (%d records so far, %.1f min elapsed)",
                        model_name, family_name, len(records),
                        (time.perf_counter() - started) / 60)

    wall = time.perf_counter() - started
    results = pd.DataFrame(records)
    per_step = pd.DataFrame(per_step_records)
    checkpoint()

    provenance = {
        "wall_seconds": wall,
        "extraction": matrices.summary(),
        "extraction_config": matrices.config,
        "split_counts": split.counts(),
        "seeds": list(seeds),
        "budget": budget,
        "models": models,
        "families": family_names,
        "provenance": RunProvenance(seed=0).as_dict(),
        "preregistration_note": (
            "Hyperparameters chosen on validation only; the test split was scored once per "
            "(model, family, seed) and used for no selection."
        ),
    }
    (OUT / f"transaction_provenance_{suffix}.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )

    if not results.empty:
        pivot = (results.groupby(["family", "model"])["pr_auc"]
                 .agg(["mean", "std", "count"]).round(4))
        print("\n=== test illicit PR-AUC by family and model ===")
        print(pivot.to_string())
        best = results.groupby("family")["pr_auc"].mean().sort_values(ascending=False)
        print("\n=== mean PR-AUC by family (across models and seeds) ===")
        print(best.round(4).to_string())
    logger.info("wrote results to %s (wall %.1f s)", OUT, wall)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="3 seeds, 3 families, small budget: a smoke test, not a result")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--complex-dim", type=int, default=2, dest="complex_dim")
    parser.add_argument("--budget", type=int, default=SEARCH_BUDGET)
    parser.add_argument("--models", nargs="+",
                        default=["logistic", "random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()
    if args.quick:
        args.budget = min(args.budget, 5)
    raise SystemExit(main(**vars(args)))
