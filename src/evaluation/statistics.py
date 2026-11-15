"""Statistical procedures, frozen in ``docs/PREREGISTRATION.md`` §7.

Implemented here rather than pulled from ``statsmodels`` so that the dependency surface stays
small and every procedure the manuscript relies on is readable in one place.

The governing rule, from §7: **no superiority claim is made from a point difference whose
confidence interval includes zero**, however large the difference looks. Every comparison function
here therefore returns the interval, not just a *p*-value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_ALPHA = 0.05


@dataclass
class Comparison:
    """Result of a paired comparison between two methods across seeds."""

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    sd_a: float
    sd_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_corrected: float | None
    cohens_d: float
    cliffs_delta: float
    n_pairs: int
    significant: bool
    differences: tuple[float, ...] = ()
    ci_low_corrected: float | None = None
    ci_high_corrected: float | None = None
    decision_rule: str = "uncorrected"

    def as_dict(self) -> dict:
        return asdict(self)

    def verdict(self) -> str:
        """One-line plain reading, deliberately conservative about direction."""
        low = self.ci_low_corrected if self.ci_low_corrected is not None else self.ci_low
        high = self.ci_high_corrected if self.ci_high_corrected is not None else self.ci_high
        if not self.significant:
            return (f"{self.name_a} vs {self.name_b}: no significant difference "
                    f"(diff {self.mean_difference:+.4f}, corrected CI "
                    f"[{low:+.4f}, {high:+.4f}])")
        better = self.name_a if self.mean_difference > 0 else self.name_b
        return (f"{better} is higher (diff {self.mean_difference:+.4f}, corrected CI "
                f"[{low:+.4f}, {high:+.4f}], "
                f"Cliff's delta={self.cliffs_delta:+.3f})")


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic=np.mean,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``statistic`` of ``values``."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_resamples, values.size), replace=True)
    estimates = statistic(draws, axis=1)
    return (float(np.quantile(estimates, alpha / 2.0)),
            float(np.quantile(estimates, 1.0 - alpha / 2.0)))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: ``P(a > b) - P(a < b)``, in ``[-1, 1]``.

    A non-parametric effect size, appropriate here because per-seed metric distributions are small
    and not reliably normal. Computed via the Mann-Whitney U statistic rather than by enumerating
    all pairs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (a.size * b.size) - 1.0)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples, using the standard deviation of the differences."""
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = differences.std(ddof=1)
    if sd == 0:
        return 0.0 if differences.mean() == 0 else float("inf")
    return float(differences.mean() / sd)


def paired_comparison(
    a: np.ndarray,
    b: np.ndarray,
    *,
    name_a: str = "A",
    name_b: str = "B",
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Comparison:
    """Paired Wilcoxon signed-rank comparison of two methods across matched seeds.

    Pairing is by seed: entry *i* of both arrays must come from the same seed, so that
    seed-to-seed variance is removed from the comparison rather than inflating it.

    Raises:
        ValueError: if the arrays differ in length, since silently truncating would break pairing.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs matched arrays, got {a.shape} and {b.shape}")

    differences = a - b
    if np.allclose(differences, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)
        except ValueError:  # all differences zero after tie handling
            p_value = 1.0

    ci_low, ci_high = bootstrap_ci(differences, alpha=alpha, seed=seed)
    return Comparison(
        name_a=name_a, name_b=name_b,
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        sd_a=float(a.std(ddof=1)) if a.size > 1 else 0.0,
        sd_b=float(b.std(ddof=1)) if b.size > 1 else 0.0,
        mean_difference=float(differences.mean()),
        ci_low=ci_low, ci_high=ci_high,
        p_value=p_value, p_value_corrected=None,
        cohens_d=cohens_d_paired(a, b),
        cliffs_delta=cliffs_delta(a, b),
        n_pairs=int(a.size),
        differences=tuple(float(d) for d in differences),
        # Significance requires both the test and the interval to agree. An interval containing
        # zero vetoes the claim regardless of the p-value, per PREREGISTRATION.md §7.
        significant=bool(p_value < alpha and not (ci_low <= 0.0 <= ci_high)),
        decision_rule="wilcoxon+ci (uncorrected)",
    )


def holm_bonferroni(p_values: np.ndarray, *, alpha: float = DEFAULT_ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction.

    Returns ``(rejected, adjusted)``. Adjusted values are made monotone non-decreasing in rank
    and clipped at 1.0, which is what makes them comparable to ``alpha`` directly.

    Holm is used rather than Benjamini-Hochberg because the pre-registration controls the
    family-wise error rate over a small, enumerated family of comparisons, not the false discovery
    rate over a large screen.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    order = np.argsort(p_values)
    ordered = p_values[order]
    # Step-down: multiply the i-th smallest p by (n - i), then enforce monotonicity so a later
    # (larger) p can never receive a smaller adjusted value than an earlier one.
    adjusted_ordered = np.maximum.accumulate((n - np.arange(n)) * ordered)
    adjusted_ordered = np.clip(adjusted_ordered, 0.0, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted < alpha, adjusted


def wilcoxon_p_floor(n_pairs: int) -> float:
    """Smallest two-sided p-value the signed-rank test can return for ``n_pairs`` samples.

    The statistic is discrete: with every difference sharing a sign, the two-sided exact p-value is
    ``2 / 2**n``. At ten pairs that is 0.00195, so any multiple-comparison threshold below it is
    unreachable no matter how large the effect.
    """
    return float(2.0 / (2.0 ** n_pairs)) if n_pairs > 0 else 1.0


def correct_family(comparisons: list[Comparison], *, alpha: float = DEFAULT_ALPHA) -> list[Comparison]:
    """Control the family-wise error rate across a declared family of comparisons, in place.

    The family is enumerated in ``docs/PREREGISTRATION.md`` §7. Exploratory tests are corrected
    separately, or reported uncorrected and labelled as exploratory -- never pooled into this
    family after the fact, which would change the correction applied to the confirmatory tests.

    Why the decision is not taken on the Holm-corrected Wilcoxon p-value
    -------------------------------------------------------------------
    It cannot be, at this sample size. The signed-rank p-value is discrete with a floor of
    ``2 / 2**n``; at the pre-registered ten seeds that floor is 0.00195, while Holm's most
    stringent threshold over 31 comparisons is 0.05/31 = 0.00161. **No comparison can reach
    significance regardless of effect size** -- the procedure as originally specified reports "not
    significant" for a PR-AUC gap of 0.65 with a confidence interval nowhere near zero. That is an
    artefact of test discreteness, not a statement about the data.

    The decision is therefore taken on a **Bonferroni-adjusted bootstrap interval**: each
    difference's interval is recomputed at level ``1 - alpha/m`` and significance requires it to
    exclude zero. This controls the family-wise error rate, is not subject to the discreteness
    floor, and uses the same paired differences. The Wilcoxon p-value and its Holm adjustment are
    still computed and reported, with the floor alongside, so a reader can see both.
    """
    if not comparisons:
        return comparisons

    rejected, adjusted = holm_bonferroni(
        np.array([c.p_value for c in comparisons]), alpha=alpha
    )
    m = len(comparisons)
    adjusted_alpha = alpha / m

    # Use the pre-registered rule when the sample size makes it reachable, and fall back only when
    # it does not. At ten seeds the signed-rank floor of 2/2**10 = 0.00195 sits above Holm's
    # strictest threshold over a 31-member family, so nothing could ever be declared significant;
    # at twenty it is 1.9e-6 and the frozen rule works as intended. Whichever applies is recorded
    # on every comparison, and the interval is reported either way.
    floor = max(wilcoxon_p_floor(c.n_pairs) for c in comparisons)
    holm_is_reachable = floor < (alpha / m)

    for index, (comparison, reject, adjusted_p) in enumerate(zip(comparisons, rejected, adjusted)):
        comparison.p_value_corrected = float(adjusted_p)
        differences = np.asarray(comparison.differences, dtype=float)
        if differences.size:
            low, high = bootstrap_ci(differences, alpha=adjusted_alpha, seed=index)
        else:
            low, high = comparison.ci_low, comparison.ci_high
        comparison.ci_low_corrected = low
        comparison.ci_high_corrected = high
        interval_excludes_zero = bool(np.isfinite(low) and np.isfinite(high)
                                      and not (low <= 0.0 <= high))
        if holm_is_reachable:
            # Pre-registered: Holm-corrected Wilcoxon, with the interval retained as a veto.
            comparison.significant = bool(reject and interval_excludes_zero)
            comparison.decision_rule = (
                f"pre-registered Holm-corrected Wilcoxon at n={comparison.n_pairs}, with the "
                f"interval as a veto; floor {wilcoxon_p_floor(comparison.n_pairs):.2e} is below "
                f"the {alpha / m:.2e} threshold"
            )
        else:
            comparison.significant = interval_excludes_zero
            comparison.decision_rule = (
                f"bonferroni-adjusted bootstrap interval at alpha/{m}; the pre-registered Holm "
                f"rule is unreachable because the wilcoxon floor at n={comparison.n_pairs} is "
                f"{wilcoxon_p_floor(comparison.n_pairs):.2e}, above the {alpha / m:.2e} threshold"
            )
    return comparisons


def mann_whitney_with_effect(
    a: np.ndarray, b: np.ndarray
) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test with Cliff's delta, for hypothesis H1."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    if np.allclose(a.std(), 0.0) and np.allclose(b.std(), 0.0) and np.allclose(a.mean(), b.mean()):
        return 1.0, 0.0
    p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return p_value, cliffs_delta(a, b)


def _fast_delong(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance estimate for one or more score vectors on shared labels.

    Implements the midrank formulation: for ``k`` predictors on the same sample, returns the AUC
    estimates and their ``k x k`` covariance matrix.
    """
    order = np.argsort(-labels, kind="mergesort")
    labels = labels[order]
    scores = scores[:, order]
    n_pos = int((labels == 1).sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("DeLong needs both classes present")

    k = scores.shape[0]
    positive = scores[:, :n_pos]
    negative = scores[:, n_pos:]

    def midrank(values: np.ndarray) -> np.ndarray:
        sorted_index = np.argsort(values)
        sorted_values = values[sorted_index]
        ranks = np.empty(values.size, dtype=float)
        i = 0
        while i < values.size:
            j = i
            while j < values.size and sorted_values[j] == sorted_values[i]:
                j += 1
            ranks[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(values.size, dtype=float)
        out[sorted_index] = ranks
        return out

    tx = np.empty((k, n_pos)); ty = np.empty((k, n_neg)); tz = np.empty((k, labels.size))
    for row in range(k):
        tx[row] = midrank(positive[row])
        ty[row] = midrank(negative[row])
        tz[row] = midrank(scores[row])

    auc = (tz[:, :n_pos].sum(axis=1) / n_pos - (n_pos + 1.0) / 2.0) / n_neg
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    covariance = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    return auc, np.atleast_2d(covariance)


def delong_test(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> dict[str, float]:
    """DeLong's test for two correlated ROC-AUCs on the same test set.

    Used where two models are compared on one shared test split, which the paired-across-seeds
    Wilcoxon procedure does not cover.
    """
    labels = np.asarray(y_true, dtype=float)
    scores = np.vstack([np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)])
    auc, covariance = _fast_delong(scores, labels)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    # Identical (or perfectly correlated) score vectors give zero variance, which is not an error:
    # there is genuinely no difference to test. The return shape stays the same either way so
    # callers never have to guard for a missing key.
    z = 0.0 if variance <= 0 else float((auc[0] - auc[1]) / np.sqrt(variance))
    return {
        "auc_a": float(auc[0]),
        "auc_b": float(auc[1]),
        "difference": float(auc[0] - auc[1]),
        "variance": variance,
        "z": z,
        "p_value": 1.0 if variance <= 0 else float(2.0 * stats.norm.sf(abs(z))),
    }


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Spearman correlation of ``x`` and ``y`` after linearly removing ``controls``.

    Why this exists. Several graph descriptors are **deterministic functions of elementary
    counts**: for a connected graph the cycle rank per node is ``|E|/|V| - 1``, which is mean
    degree rescaled, and ``beta_0`` per node is ``1/|V|`` exactly. A raw correlation between such a
    descriptor and an outcome is therefore a correlation with size or density wearing a topological
    name. Partialling the elementary counts out is what distinguishes a topological finding from a
    density finding.

    Both variables and the controls are rank-transformed first, then the controls are projected out
    by least squares and the residuals correlated. An intercept is included so a constant control
    is harmless.

    Args:
        x, y: the two variables.
        controls: ``(n,)`` or ``(n, k)`` variables to remove.

    Returns:
        ``rho``, a bootstrap 95% interval, and ``n``. A ``rho`` whose interval spans zero means the
        association is explained by the controls.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]

    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    x, y, controls = x[keep], y[keep], controls[keep]
    if x.size < 5:
        return {"rho": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": int(x.size)}

    def residualise(values: np.ndarray, design: np.ndarray) -> np.ndarray:
        ranked = stats.rankdata(values)
        basis = np.column_stack([np.ones(design.shape[0])]
                                + [stats.rankdata(design[:, j]) for j in range(design.shape[1])])
        coefficients, *_ = np.linalg.lstsq(basis, ranked, rcond=None)
        return ranked - basis @ coefficients

    def estimate(idx: np.ndarray) -> float:
        rx = residualise(x[idx], controls[idx])
        ry = residualise(y[idx], controls[idx])
        if np.allclose(rx.std(), 0.0) or np.allclose(ry.std(), 0.0):
            return 0.0
        return float(np.corrcoef(rx, ry)[0, 1])

    observed = estimate(np.arange(x.size))
    rng = np.random.default_rng(seed)
    draws = np.array([estimate(rng.integers(0, x.size, x.size)) for _ in range(n_resamples)])
    draws = draws[np.isfinite(draws)]
    return {
        "rho": observed,
        "ci_low": float(np.quantile(draws, 0.025)) if draws.size else float("nan"),
        "ci_high": float(np.quantile(draws, 0.975)) if draws.size else float("nan"),
        "n": int(x.size),
    }


def spearman_with_ci(
    x: np.ndarray, y: np.ndarray, *, n_resamples: int = DEFAULT_BOOTSTRAP, seed: int = 0
) -> dict[str, float]:
    """Spearman correlation with a bootstrap confidence interval, for hypothesis H4."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return {"rho": float("nan"), "p_value": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": int(x.size)}
    result = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        index = rng.integers(0, x.size, x.size)
        if np.unique(x[index]).size < 3 or np.unique(y[index]).size < 3:
            estimates[i] = np.nan
            continue
        estimates[i] = stats.spearmanr(x[index], y[index]).statistic
    estimates = estimates[np.isfinite(estimates)]
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "ci_low": float(np.quantile(estimates, 0.025)) if estimates.size else float("nan"),
        "ci_high": float(np.quantile(estimates, 0.975)) if estimates.size else float("nan"),
        "n": int(x.size),
    }
