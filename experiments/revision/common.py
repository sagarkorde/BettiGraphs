"""Shared plumbing for the revision experiments (BCRA submission, Amendment 5).

Every experiment in this folder is EXPLORATORY and post hoc: none was in the frozen
pre-registration. Each writes a JSON with ``"status": "exploratory (Amendment 5)"`` so the
manuscript build cannot present it as confirmatory.

Design rules carried over from the frozen protocol:
* temporal splits only; nothing from a later step informs an earlier prediction;
* hyper-parameters chosen on validation PR-AUC only, with the same budget for every family;
* the test window is scored once per (family, seed);
* features come from ``src.features.pipeline.extract_features`` with the canonical
  configuration (radius 2, value superlevel, complex dim 2), so the cached ``.npz`` files from
  the r2s20 run are reused and nothing is recomputed unless the config differs.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features.pipeline import ExtractionConfig, extract_features  # noqa: E402
from src.graph_construction.graphs import build_address_graph, build_tx_graph  # noqa: E402,F401
__all__ = ["build_address_graph", "build_tx_graph", "extract_features", "ExtractionConfig"]

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
CACHE = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "results" / "revision"
STATUS = "exploratory (Amendment 5), not pre-registered"
CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}

logger = logging.getLogger("revision")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)


def canonical_config(view: str = "G_tx", **overrides) -> ExtractionConfig:
    base = dict(view=view, radius=2, filtration="value_superlevel", complex_dim=2,
                max_dim=1, max_nodes=25_000, include_classical=True)
    base.update(overrides)
    return ExtractionConfig(**base)


@dataclass
class TxData:
    frame: pd.DataFrame          # labelled transactions, row order = feature-matrix order
    families: dict[str, np.ndarray]
    steps: np.ndarray
    labels: np.ndarray
    t_names: list[str]


def load_tx(n_jobs: int = -1, with_unlabelled_store: bool = False) -> tuple[TxData, object]:
    """Labelled transactions and the eight canonical feature families (cache hit expected)."""
    from experiments.baseline.run_transaction_experiment import (  # noqa: E402
        family_matrices, load_transaction_table,
    )
    frame = load_transaction_table()
    labelled = frame[frame.label != -1].reset_index(drop=True)
    store = build_tx_graph(RAW_PP)
    m = extract_features(store, labelled["txId"].to_numpy(), canonical_config(),
                         n_jobs=n_jobs, cache_dir=CACHE)
    fams = {k: v[0] for k, v in family_matrices(labelled, m.topological, m.classical).items()}
    return TxData(labelled, fams, labelled["Time step"].to_numpy(),
                  labelled["label"].to_numpy(), list(m.topological_names)), store


def search_and_eval(model: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray,
                    te: np.ndarray, seeds, budget: int, refit_on_val: bool = True) -> list[dict]:
    """Validation-only random search, then one test score per seed."""
    from src.evaluation.metrics import compute_metrics, lift_at_k
    from src.models.tabular import fit_model, random_search
    from src.utils.seeds import set_all_seeds

    params, best_val = random_search(model, X[tr], y[tr], X[va], y[va], budget=budget, seed=0)
    fit_idx = tr | va if refit_on_val else tr
    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        fitted = fit_model(model, X[fit_idx], y[fit_idx], seed=seed, params=params)
        scores = fitted.predict_proba(X[te])
        met = compute_metrics(y[te], scores)
        rows.append({"seed": seed, "val_pr_auc": best_val, "params": json.dumps(params),
                     "pr_auc": met.pr_auc, "roc_auc": met.roc_auc,
                     "lift_at_100": lift_at_k(y[te], scores, 100),
                     "n_test": int(te.sum()), "n_test_illicit": int((y[te] == 1).sum())})
    return rows


def write_json(name: str, payload: dict) -> Path:
    payload = {"status": STATUS, **payload}
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("wrote %s", path)
    return path
