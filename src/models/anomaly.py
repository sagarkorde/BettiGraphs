"""Topological anomaly scoring (Contribution 4).

The score measures how far an entity's topological signature sits from the distribution of
signatures of **known-licit entities in the training window**. It is deliberately one-class: an
anomaly detector that needs illicit examples to train is not solving the problem a forensic analyst
actually has, which is finding behaviour unlike the legitimate norm without knowing in advance what
the illicit behaviour looks like.

Three scorers, with different assumptions:

======================  ==================================================================
``mahalanobis``         Distance from the licit mean under a shrunk covariance. Assumes an
                        elliptical reference distribution; cheap and interpretable.
``isolation_forest``    Tree-based, assumption-free about shape.
``lof``                 Local outlier factor; catches entities anomalous relative to their
                        own neighbourhood rather than globally.
======================  ==================================================================

Fitted on training-window licit entities only
---------------------------------------------
Every scorer here is fitted on a subset of the training window and never sees validation or test
data. The covariance, the forest and the neighbourhood structure are all properties of the
reference set, so this is the same leakage discipline the supervised models follow.

Interpretation, stated because it is easy to overclaim
------------------------------------------------------
A high score means "topologically unlike the licit reference distribution". It does **not** mean
"illicit". Many legitimate entities are structurally unusual -- exchanges, mixers used lawfully,
large merchants -- and the evaluation below reports precision at realistic alert budgets precisely
so that the false-positive cost is visible rather than hidden inside an AUC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

SCORERS = ("mahalanobis", "isolation_forest", "lof")


@dataclass
class AnomalyScorer:
    """A fitted one-class scorer over topological signatures."""

    method: str
    model: object = None
    centre: np.ndarray | None = None
    precision: np.ndarray | None = None
    imputer: object = None
    scaler: object = None
    n_reference: int = 0
    meta: dict = field(default_factory=dict)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score, higher meaning more unlike the licit reference."""
        prepared = self._prepare(X)
        if self.method == "mahalanobis":
            centred = prepared - self.centre
            # Quadratic form row-wise, without forming the full n x n product.
            return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centred, self.precision, centred), 0.0))
        if self.method == "isolation_forest":
            return -self.model.score_samples(prepared)
        if self.method == "lof":
            return -self.model.score_samples(prepared)
        raise ValueError(f"unknown method {self.method!r}")

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        prepared = np.asarray(X, dtype=np.float64)
        if self.imputer is not None:
            prepared = self.imputer.transform(prepared)
        if self.scaler is not None:
            prepared = self.scaler.transform(prepared)
        return prepared


def fit_anomaly_scorer(
    X_reference: np.ndarray,
    *,
    method: str = "mahalanobis",
    seed: int = 0,
    shrinkage: float = 0.10,
    contamination: float = 0.05,
) -> AnomalyScorer:
    """Fit a one-class scorer on licit training-window signatures.

    Args:
        X_reference: signatures of known-licit training entities.
        method: one of :data:`SCORERS`.
        shrinkage: ridge added to the covariance diagonal for ``mahalanobis``. Topological
            signatures contain strongly collinear columns -- adjacent Betti-curve samples are
            nearly identical by construction -- so the empirical covariance is near-singular and a
            raw inverse would be dominated by noise directions.
        contamination: assumed outlier fraction for the two scikit-learn detectors.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    if method not in SCORERS:
        raise ValueError(f"unknown method {method!r}; expected one of {SCORERS}")

    imputer = SimpleImputer(strategy="median").fit(X_reference)
    prepared = imputer.transform(X_reference)
    scaler = StandardScaler().fit(prepared)
    prepared = scaler.transform(prepared)

    scorer = AnomalyScorer(method=method, imputer=imputer, scaler=scaler,
                           n_reference=int(prepared.shape[0]))

    if method == "mahalanobis":
        centre = prepared.mean(axis=0)
        covariance = np.cov(prepared, rowvar=False)
        covariance = np.atleast_2d(covariance)
        # Ledoit-Wolf style shrinkage toward a scaled identity, so the inverse is well conditioned.
        trace_mean = float(np.trace(covariance) / covariance.shape[0])
        regularised = (1.0 - shrinkage) * covariance + shrinkage * trace_mean * np.eye(
            covariance.shape[0]
        )
        scorer.centre = centre
        scorer.precision = np.linalg.pinv(regularised)
        scorer.meta = {"shrinkage": shrinkage, "n_features": int(prepared.shape[1]),
                       "condition_number": float(np.linalg.cond(regularised))}
        logger.info("mahalanobis reference: %d entities, %d features, condition number %.3g",
                    prepared.shape[0], prepared.shape[1], scorer.meta["condition_number"])
    elif method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        scorer.model = IsolationForest(
            n_estimators=300, contamination=contamination, random_state=seed, n_jobs=-1
        ).fit(prepared)
    else:
        from sklearn.neighbors import LocalOutlierFactor

        scorer.model = LocalOutlierFactor(
            n_neighbors=20, contamination=contamination, novelty=True, n_jobs=-1
        ).fit(prepared)

    return scorer


def evaluate_anomaly_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    budgets: tuple[int, ...] = (50, 100, 500, 1000),
) -> dict[str, float]:
    """Evaluate an unsupervised score against labels it never saw.

    Reports PR-AUC and ROC-AUC for comparability with the supervised results, and **precision at
    fixed alert budgets**, which is what an investigator actually experiences: a finite number of
    cases they can review. A detector with a respectable AUC and poor precision at 100 alerts is
    not useful in practice, and only the second number shows that.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("labels contain a single class; PR-AUC and ROC-AUC are undefined")

    base_rate = float(labels.mean())
    out: dict[str, float] = {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "base_rate": base_rate,
        "n": int(labels.size),
        "n_illicit": int(labels.sum()),
    }
    order = np.argsort(-scores)
    for budget in budgets:
        if budget > labels.size:
            continue
        top = labels[order[:budget]]
        out[f"precision_at_{budget}"] = float(top.mean())
        out[f"lift_at_{budget}"] = float(top.mean() / base_rate) if base_rate > 0 else float("nan")
        out[f"recall_at_{budget}"] = float(top.sum() / max(labels.sum(), 1))
    return out
