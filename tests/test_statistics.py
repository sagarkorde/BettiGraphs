"""Known-answer tests for the statistical procedures.

These are the procedures every claim in the manuscript rests on, so they are checked against
values that can be verified independently: Holm against hand-computed adjustments, DeLong against
scikit-learn's AUC and against its own symmetry, Cliff's delta against its definition by brute
force, and the significance rule against the pre-registration's requirement that a confidence
interval containing zero vetoes a claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation import statistics as st


class TestHolmBonferroni:
    def test_hand_computed_example(self):
        """p = (0.01, 0.02, 0.03, 0.04), n = 4.

        Step-down multipliers are 4, 3, 2, 1 giving 0.04, 0.06, 0.06, 0.06 after the monotonicity
        constraint lifts the third and fourth up to the running maximum.
        """
        rejected, adjusted = st.holm_bonferroni(np.array([0.01, 0.02, 0.03, 0.04]), alpha=0.05)
        assert np.allclose(adjusted, [0.04, 0.06, 0.06, 0.06])
        assert rejected.tolist() == [True, False, False, False]

    def test_adjusted_values_are_monotone_in_rank(self):
        rng = np.random.default_rng(0)
        p_values = rng.uniform(0, 1, 40)
        _, adjusted = st.holm_bonferroni(p_values)
        order = np.argsort(p_values)
        assert np.all(np.diff(adjusted[order]) >= -1e-12)

    def test_adjusted_never_below_raw_and_never_above_one(self):
        rng = np.random.default_rng(1)
        p_values = rng.uniform(0, 1, 50)
        _, adjusted = st.holm_bonferroni(p_values)
        assert np.all(adjusted >= p_values - 1e-12)
        assert np.all(adjusted <= 1.0)

    def test_single_test_is_unchanged(self):
        _, adjusted = st.holm_bonferroni(np.array([0.03]))
        assert adjusted[0] == pytest.approx(0.03)

    def test_is_more_conservative_than_no_correction(self):
        p_values = np.full(20, 0.04)
        rejected, _ = st.holm_bonferroni(p_values, alpha=0.05)
        assert not rejected.any(), "20 tests at p=0.04 must not all survive correction"

    def test_empty_input(self):
        rejected, adjusted = st.holm_bonferroni(np.array([]))
        assert rejected.size == 0 and adjusted.size == 0


class TestCliffsDelta:
    def _brute_force(self, a, b):
        greater = sum(1 for x in a for y in b if x > y)
        less = sum(1 for x in a for y in b if x < y)
        return (greater - less) / (len(a) * len(b))

    def test_matches_brute_force_definition(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.6, 1.0, 30)
        b = rng.normal(0.0, 1.0, 25)
        assert st.cliffs_delta(a, b) == pytest.approx(self._brute_force(a, b), abs=1e-9)

    def test_complete_separation_gives_plus_one(self):
        assert st.cliffs_delta(np.array([5.0, 6, 7]), np.array([1.0, 2, 3])) == pytest.approx(1.0)

    def test_reversed_separation_gives_minus_one(self):
        assert st.cliffs_delta(np.array([1.0, 2, 3]), np.array([5.0, 6, 7])) == pytest.approx(-1.0)

    def test_identical_samples_give_zero(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert st.cliffs_delta(values, values) == pytest.approx(0.0)


class TestPairedComparison:
    def test_detects_a_consistent_difference(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0.5, 0.02, 10)
        result = st.paired_comparison(base + 0.05, base, name_a="T", name_b="S")
        assert result.significant
        assert result.mean_difference == pytest.approx(0.05, abs=1e-9)
        assert result.ci_low > 0

    def test_reports_no_difference_when_there_is_none(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0.5, 0.05, 10)
        b = rng.normal(0.5, 0.05, 10)
        result = st.paired_comparison(a, b)
        assert not result.significant
        assert "no significant difference" in result.verdict()

    def test_interval_containing_zero_vetoes_significance(self):
        """PREREGISTRATION.md §7: a CI spanning zero blocks the claim regardless of p."""
        a = np.array([0.50, 0.51, 0.49, 0.52, 0.48, 0.90, 0.10, 0.51, 0.49, 0.50])
        b = a - 0.001
        result = st.paired_comparison(a, b)
        if result.ci_low <= 0.0 <= result.ci_high:
            assert not result.significant

    def test_identical_inputs_are_not_significant(self):
        values = np.linspace(0.4, 0.6, 10)
        result = st.paired_comparison(values, values.copy())
        assert not result.significant
        assert result.mean_difference == pytest.approx(0.0)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="matched arrays"):
            st.paired_comparison(np.zeros(5), np.zeros(6))

    def test_correction_across_a_family_weakens_significance(self):
        rng = np.random.default_rng(5)
        comparisons = []
        for i in range(12):
            base = rng.normal(0.5, 0.03, 10)
            comparisons.append(st.paired_comparison(base + 0.005, base, name_a=f"a{i}", name_b="b"))
        before = sum(c.significant for c in comparisons)
        st.correct_family(comparisons)
        after = sum(c.significant for c in comparisons)
        assert after <= before
        assert all(c.p_value_corrected is not None for c in comparisons)
        assert all(c.p_value_corrected >= c.p_value - 1e-12 for c in comparisons)


class TestDeLong:
    def test_auc_matches_sklearn(self):
        rng = np.random.default_rng(6)
        y = rng.integers(0, 2, 400)
        score = y * 0.4 + rng.normal(0, 1, 400)
        result = st.delong_test(y, score, score)
        assert result["auc_a"] == pytest.approx(roc_auc_score(y, score), abs=1e-9)

    def test_identical_scores_give_zero_difference_and_p_one(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 300)
        score = y * 0.5 + rng.normal(0, 1, 300)
        result = st.delong_test(y, score, score)
        assert result["difference"] == pytest.approx(0.0, abs=1e-12)
        assert result["p_value"] == pytest.approx(1.0)

    def test_detects_a_clearly_better_predictor(self):
        rng = np.random.default_rng(8)
        y = rng.integers(0, 2, 600)
        strong = y * 2.0 + rng.normal(0, 1, 600)
        weak = y * 0.1 + rng.normal(0, 1, 600)
        result = st.delong_test(y, strong, weak)
        assert result["difference"] > 0
        assert result["p_value"] < 1e-6

    def test_is_antisymmetric_in_its_arguments(self):
        rng = np.random.default_rng(9)
        y = rng.integers(0, 2, 300)
        a = y * 1.0 + rng.normal(0, 1, 300)
        b = y * 0.3 + rng.normal(0, 1, 300)
        forward = st.delong_test(y, a, b)
        backward = st.delong_test(y, b, a)
        assert forward["difference"] == pytest.approx(-backward["difference"], abs=1e-12)
        assert forward["p_value"] == pytest.approx(backward["p_value"], abs=1e-12)

    def test_single_class_is_rejected(self):
        with pytest.raises(ValueError, match="both classes"):
            st.delong_test(np.ones(10), np.arange(10.0), np.arange(10.0))


class TestBootstrapAndSpearman:
    def test_bootstrap_interval_brackets_the_mean(self):
        rng = np.random.default_rng(10)
        values = rng.normal(0.7, 0.05, 200)
        low, high = st.bootstrap_ci(values, n_resamples=2000)
        assert low < values.mean() < high

    def test_bootstrap_of_a_constant_is_degenerate(self):
        low, high = st.bootstrap_ci(np.full(20, 0.42), n_resamples=500)
        assert low == pytest.approx(0.42) and high == pytest.approx(0.42)

    def test_spearman_detects_a_monotone_relationship(self):
        x = np.arange(49.0)
        y = x ** 1.5
        result = st.spearman_with_ci(x, y, n_resamples=500)
        assert result["rho"] == pytest.approx(1.0)
        assert result["ci_low"] > 0.9

    def test_spearman_on_noise_has_an_interval_spanning_zero(self):
        rng = np.random.default_rng(11)
        result = st.spearman_with_ci(rng.normal(size=49), rng.normal(size=49), n_resamples=1000)
        assert result["ci_low"] < 0 < result["ci_high"]


class TestDiscretenessFloor:
    """The signed-rank test cannot reach a corrected threshold at ten seeds.

    This is why the family-wise decision is taken on Bonferroni-adjusted bootstrap intervals
    rather than on Holm-corrected Wilcoxon p-values. With the pre-registered ten seeds the
    two-sided p-value floor is 2/2^10 = 0.00195, while Holm's most stringent threshold over 31
    comparisons is 0.05/31 = 0.00161. Left as originally specified, the procedure reports "not
    significant" for a PR-AUC gap of 0.65 whose interval is nowhere near zero.
    """

    def test_floor_matches_the_closed_form(self):
        assert st.wilcoxon_p_floor(10) == pytest.approx(2 / 2 ** 10)
        assert st.wilcoxon_p_floor(20) == pytest.approx(2 / 2 ** 20)

    def test_floor_is_unreachable_under_holm_at_ten_seeds(self):
        floor = st.wilcoxon_p_floor(10)
        assert floor > 0.05 / 31, "if this ever fails the workaround is no longer needed"

    def test_an_enormous_effect_is_detected_despite_the_floor(self):
        """The regression this guards: a huge, unambiguous difference must not read as null."""
        rng = np.random.default_rng(0)
        base = rng.normal(0.06, 0.002, 10)
        comparisons = [
            st.paired_comparison(base + 0.65, base, name_a="P", name_b="T")
        ] + [
            st.paired_comparison(rng.normal(0.5, 0.05, 10), rng.normal(0.5, 0.05, 10),
                                 name_a=f"n{i}", name_b="x")
            for i in range(30)
        ]
        st.correct_family(comparisons)
        assert comparisons[0].significant, "a 0.65 gap must survive correction"
        assert comparisons[0].ci_low_corrected > 0

    def test_a_null_difference_is_not_detected(self):
        rng = np.random.default_rng(1)
        comparisons = [
            st.paired_comparison(rng.normal(0.5, 0.05, 10), rng.normal(0.5, 0.05, 10),
                                 name_a=f"a{i}", name_b="b")
            for i in range(31)
        ]
        st.correct_family(comparisons)
        assert sum(c.significant for c in comparisons) <= 2, "null comparisons should mostly fail"

    def test_corrected_interval_is_wider_than_the_uncorrected_one(self):
        rng = np.random.default_rng(2)
        base = rng.normal(0.5, 0.03, 10)
        comparisons = [st.paired_comparison(base + 0.02, base, name_a="a", name_b="b")
                       for _ in range(10)]
        st.correct_family(comparisons)
        first = comparisons[0]
        width_raw = first.ci_high - first.ci_low
        width_corrected = first.ci_high_corrected - first.ci_low_corrected
        assert width_corrected >= width_raw

    def test_decision_rule_falls_back_only_when_holm_is_unreachable(self):
        """A 31-member family at ten seeds cannot use Holm, so the interval rule must take over."""
        rng = np.random.default_rng(3)
        base = rng.normal(0.5, 0.02, 10)
        comparisons = [st.paired_comparison(base + 0.1, base, name_a=f"a{i}", name_b="b")
                       for i in range(31)]
        st.correct_family(comparisons)
        assert "bonferroni" in comparisons[0].decision_rule
        assert "wilcoxon floor" in comparisons[0].decision_rule
        assert "unreachable" in comparisons[0].decision_rule

    def test_decision_rule_uses_holm_when_the_family_is_small_enough(self):
        """With one comparison the threshold is 0.05, which ten seeds can reach, so Holm applies."""
        rng = np.random.default_rng(3)
        base = rng.normal(0.5, 0.02, 10)
        comparisons = [st.paired_comparison(base + 0.1, base, name_a="a", name_b="b")]
        st.correct_family(comparisons)
        assert "Holm" in comparisons[0].decision_rule
        assert comparisons[0].significant

    def test_twenty_seeds_restore_the_preregistered_rule(self):
        """The reason SEEDS was raised: at n=20 the frozen Holm rule works over the real family."""
        rng = np.random.default_rng(4)
        base = rng.normal(0.5, 0.02, 20)
        comparisons = [st.paired_comparison(base + 0.1, base, name_a="a", name_b="b")] + [
            st.paired_comparison(rng.normal(0.5, 0.05, 20), rng.normal(0.5, 0.05, 20),
                                 name_a=f"n{i}", name_b="x")
            for i in range(30)
        ]
        st.correct_family(comparisons)
        assert all("Holm" in c.decision_rule for c in comparisons)
        assert comparisons[0].significant, "a 0.1 gap at twenty seeds must survive the frozen rule"
