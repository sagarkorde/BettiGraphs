"""Evaluation metrics, with PR-AUC on the illicit class as the primary measure.

The primary metric was fixed in ``docs/PREREGISTRATION.md`` §6 before any model was fitted,
because measured class imbalance is 9.25 : 1 at transaction level and 17.60 : 1 at wallet level.
Accuracy is computed only for completeness and supports no claim in the manuscript.

Average precision is used for PR-AUC rather than trapezoidal interpolation of the
precision-recall curve. Interpolated PR-AUC is optimistically biased under heavy imbalance, which
is exactly the regime this project operates in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

#: Decision threshold for the hard-label metrics. 0.5 is reported, and a
#: validation-selected threshold is reported alongside it where stated.
DEFAULT_THRESHOLD = 0.5


@dataclass
class ClassificationMetrics:
    """All reported metrics for one (model, feature family, seed, split) combination."""

    pr_auc: float
    roc_auc: float
    illicit_precision: float
    illicit_recall: float
    illicit_f1: float
    macro_f1: float
    mcc: float
    balanced_accuracy: float
    accuracy: float
    n: int
    n_illicit: int
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> ClassificationMetrics:
    """Compute every reported metric from labels and scores.

    Args:
        y_true: binary labels, 1 = illicit.
        y_score: predicted probability of the illicit class.
        threshold: cut for the hard-label metrics.

    Raises:
        ValueError: if ``y_true`` has only one class, since PR-AUC and ROC-AUC are undefined and
            returning a placeholder would silently corrupt an aggregate.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, y_score {y_score.shape}")
    n_illicit = int((y_true == 1).sum())
    if n_illicit == 0 or n_illicit == y_true.size:
        raise ValueError(
            f"y_true contains a single class ({n_illicit} illicit of {y_true.size}); "
            "PR-AUC and ROC-AUC are undefined"
        )

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        illicit_precision=float(precision_score(y_true, y_pred, zero_division=0)),
        illicit_recall=float(recall_score(y_true, y_pred, zero_division=0)),
        illicit_f1=float(f1_score(y_true, y_pred, zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        accuracy=float((y_true == y_pred).mean()),
        n=int(y_true.size),
        n_illicit=n_illicit,
        threshold=float(threshold),
        true_positives=int(tp), false_positives=int(fp),
        true_negatives=int(tn), false_negatives=int(fn),
    )


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximising illicit F1. Selected on **validation only**.

    Returned for reporting a tuned operating point beside the fixed 0.5 one. Calling this on test
    scores would be a form of test-set fitting, so callers must pass validation data.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    candidates = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    best, best_f1 = DEFAULT_THRESHOLD, -1.0
    for threshold in candidates:
        score = f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best, best_f1 = float(threshold), float(score)
    return best


def per_step_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    time_steps: np.ndarray,
    *,
    min_illicit: int = 1,
) -> dict[int, dict[str, float]]:
    """PR-AUC within each time step, **always returned with that step's illicit count**.

    Steps 45 and 46 of Elliptic hold only 5 and 2 illicit transactions, so a PR-AUC computed there
    is nearly meaningless alone. The count travels with the value so no downstream table can
    present one without the other, and steps below ``min_illicit`` report ``nan`` rather than a
    fabricated number.
    """
    out: dict[int, dict[str, float]] = {}
    for step in np.unique(time_steps):
        mask = time_steps == step
        labels, scores = y_true[mask], y_score[mask]
        n_illicit = int((labels == 1).sum())
        entry = {"n": int(mask.sum()), "n_illicit": n_illicit,
                 "base_rate": float(n_illicit / max(mask.sum(), 1))}
        if n_illicit < min_illicit or n_illicit == mask.sum():
            entry["pr_auc"] = float("nan")
        else:
            entry["pr_auc"] = float(average_precision_score(labels, scores))
        out[int(step)] = entry
    return out


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision among the top ``k`` scored items, divided by the base rate.

    Reported because it is what an investigator actually experiences: a fixed alert budget.
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(y_score, dtype=float))[:k]
    base_rate = y_true.mean()
    if base_rate <= 0:
        return float("nan")
    return float(y_true[order].mean() / base_rate)
