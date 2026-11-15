"""Tabular classifiers, with an equal search budget across feature families.

The pre-registration commits to an equal hyperparameter budget per (model, family) pair
(``PREREGISTRATION.md`` §5, §9.8). That commitment matters: reporting a tuned topological model
against an untuned baseline is one of the easiest ways to manufacture an improvement, and it is
explicitly ruled out. :func:`random_search` therefore takes the budget as a required argument and
the caller passes the same value everywhere.

Class imbalance is handled by **class weighting rather than resampling** by default, because
resampling changes the base rate, and the primary hypothesis H6 is a claim about behaviour under a
shifting base rate. Resampling is available as an ablation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

#: Equal random-search budget per (model, family), per PREREGISTRATION.md §5.
SEARCH_BUDGET = 30


@dataclass
class FittedModel:
    """A fitted estimator with the provenance needed to reproduce it."""

    name: str
    estimator: Any
    params: dict
    seed: int
    n_features: int
    fit_seconds: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the illicit class."""
        return self.estimator.predict_proba(X)[:, 1]


def _scaled(estimator) -> Pipeline:
    """Impute then standardise then fit. Fitted on training data only, by construction.

    ``SimpleImputer`` is included because a small number of classical features are deliberately
    returned as NaN when a quantity was refused on cost grounds (the O(n^2) square count above
    3,000 nodes). Imputing at the training median is honest; silently substituting zero would
    assert that the graph has no squares, which is a different and false claim.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _imputed(estimator) -> Pipeline:
    """Impute only; tree ensembles need no scaling."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])


def build_logistic(seed: int, **params) -> Pipeline:
    return _scaled(LogisticRegression(
        class_weight="balanced", max_iter=params.pop("max_iter", 2000),
        C=params.pop("C", 1.0), solver="lbfgs", random_state=seed, n_jobs=1,
    ))


def build_random_forest(seed: int, **params) -> Pipeline:
    return _imputed(RandomForestClassifier(
        n_estimators=params.pop("n_estimators", 400),
        max_depth=params.pop("max_depth", None),
        min_samples_leaf=params.pop("min_samples_leaf", 1),
        max_features=params.pop("max_features", "sqrt"),
        class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1,
    ))


def build_xgboost(seed: int, *, scale_pos_weight: float = 1.0, **params) -> Pipeline:
    from xgboost import XGBClassifier

    return _imputed(XGBClassifier(
        n_estimators=params.pop("n_estimators", 600),
        max_depth=params.pop("max_depth", 6),
        learning_rate=params.pop("learning_rate", 0.1),
        subsample=params.pop("subsample", 0.9),
        colsample_bytree=params.pop("colsample_bytree", 0.9),
        min_child_weight=params.pop("min_child_weight", 1),
        reg_lambda=params.pop("reg_lambda", 1.0),
        scale_pos_weight=scale_pos_weight,
        tree_method="hist", eval_metric="aucpr",
        random_state=seed, n_jobs=-1, verbosity=0,
    ))


def build_lightgbm(seed: int, *, scale_pos_weight: float = 1.0, **params) -> Pipeline:
    from lightgbm import LGBMClassifier

    return _imputed(LGBMClassifier(
        n_estimators=params.pop("n_estimators", 600),
        num_leaves=params.pop("num_leaves", 63),
        learning_rate=params.pop("learning_rate", 0.05),
        subsample=params.pop("subsample", 0.9),
        subsample_freq=1,
        colsample_bytree=params.pop("colsample_bytree", 0.9),
        min_child_samples=params.pop("min_child_samples", 20),
        reg_lambda=params.pop("reg_lambda", 0.0),
        scale_pos_weight=scale_pos_weight,
        random_state=seed, n_jobs=-1, verbose=-1,
    ))


def build_catboost(seed: int, *, scale_pos_weight: float = 1.0, **params) -> Pipeline:
    from catboost import CatBoostClassifier

    return _imputed(CatBoostClassifier(
        iterations=params.pop("iterations", 600),
        depth=params.pop("depth", 6),
        learning_rate=params.pop("learning_rate", 0.05),
        l2_leaf_reg=params.pop("l2_leaf_reg", 3.0),
        scale_pos_weight=scale_pos_weight,
        random_seed=seed, verbose=0, allow_writing_files=False, thread_count=-1,
    ))


BUILDERS: dict[str, Callable[..., Pipeline]] = {
    "logistic": build_logistic,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "catboost": build_catboost,
}

#: Search spaces, identical for every feature family. Ranges are declared here rather than tuned
#: per family so the equal-budget commitment is visible in code.
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "logistic": {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "random_forest": {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 8, 16, 32],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3],
    },
    "xgboost": {
        "n_estimators": [300, 600, 1000],
        "max_depth": [3, 4, 6, 8, 10],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 10],
        "reg_lambda": [0.1, 1.0, 10.0],
    },
    "lightgbm": {
        "n_estimators": [300, 600, 1000],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples": [10, 20, 50],
        "reg_lambda": [0.0, 1.0, 10.0],
    },
    "catboost": {
        "iterations": [300, 600],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [1.0, 3.0, 10.0],
    },
}


def sample_params(model: str, rng: np.random.Generator) -> dict:
    """Draw one hyperparameter setting uniformly from the model's declared space."""
    space = SEARCH_SPACES[model]
    return {key: values[int(rng.integers(len(values)))] for key, values in space.items()}


def fit_model(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    params: dict | None = None,
) -> FittedModel:
    """Fit one model. ``scale_pos_weight`` is derived from the training split only."""
    import time

    params = dict(params or {})
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    ratio = (n_negative / n_positive) if n_positive else 1.0

    builder = BUILDERS[model]
    if model in ("xgboost", "lightgbm", "catboost"):
        estimator = builder(seed, scale_pos_weight=ratio, **params)
    else:
        estimator = builder(seed, **params)

    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    return FittedModel(
        name=model, estimator=estimator, params=params, seed=seed,
        n_features=int(X_train.shape[1]), fit_seconds=elapsed,
    )


def random_search(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    budget: int,
    seed: int = 0,
) -> tuple[dict, float]:
    """Random search scored by **validation** illicit PR-AUC.

    Args:
        budget: number of draws. Required, not defaulted, so that an unequal budget between
            feature families has to be written explicitly and will show up in review.

    Returns:
        ``(best_params, best_validation_pr_auc)``.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    best_params: dict = {}
    best_score = -np.inf
    for draw in range(budget):
        params = sample_params(model, rng)
        try:
            fitted = fit_model(model, X_train, y_train, seed=seed, params=params)
            score = float(average_precision_score(y_val, fitted.predict_proba(X_val)))
        except Exception as exc:  # a bad draw must not abort the search
            logger.warning("%s draw %d failed (%s); skipping", model, draw, exc)
            continue
        if score > best_score:
            best_score, best_params = score, params
    if not np.isfinite(best_score):
        raise RuntimeError(f"every random-search draw for {model} failed")
    logger.info("%s best validation PR-AUC %.4f with %s", model, best_score, best_params)
    return best_params, best_score


def tree_shap_values(fitted: FittedModel, X: np.ndarray) -> np.ndarray | None:
    """Exact TreeSHAP contributions, via the booster's own implementation.

    XGBoost and LightGBM compute TreeSHAP natively, so the ``shap`` package is not a dependency.
    Returns ``None`` for models without native support.
    """
    estimator = fitted.estimator.named_steps["model"]
    prepared = X
    for name, step in fitted.estimator.named_steps.items():
        if name == "model":
            break
        prepared = step.transform(prepared)

    if fitted.name == "xgboost":
        import xgboost as xgb

        return estimator.get_booster().predict(xgb.DMatrix(prepared), pred_contribs=True)
    if fitted.name == "lightgbm":
        return estimator.predict(prepared, pred_contrib=True)
    return None
