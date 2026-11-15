"""Tests for the one-class topological anomaly scorer.

The scorer is fitted only on licit reference entities, so its correctness cannot be checked by a
held-out accuracy number. These tests instead pin the properties it must have: it separates a
planted anomalous population, it is fitted on the reference set alone, and its alert-budget
metrics behave as their definitions require.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.anomaly import SCORERS, evaluate_anomaly_scores, fit_anomaly_scorer


def reference_and_test(seed: int = 0, n_reference: int = 600, n_test: int = 400,
                       n_features: int = 12, shift: float = 3.0):
    rng = np.random.default_rng(seed)
    reference = rng.normal(0.0, 1.0, (n_reference, n_features))
    normal = rng.normal(0.0, 1.0, (n_test, n_features))
    anomalous = rng.normal(shift, 1.0, (n_test // 10, n_features))
    X = np.vstack([normal, anomalous])
    y = np.concatenate([np.zeros(len(normal), int), np.ones(len(anomalous), int)])
    return reference, X, y


class TestScorers:
    @pytest.mark.parametrize("method", SCORERS)
    def test_separates_a_planted_anomalous_population(self, method):
        reference, X, y = reference_and_test()
        scorer = fit_anomaly_scorer(reference, method=method)
        scores = scorer.score(X)
        assert scores.shape == (X.shape[0],)
        assert np.isfinite(scores).all()
        assert scores[y == 1].mean() > scores[y == 0].mean()

    @pytest.mark.parametrize("method", SCORERS)
    def test_reports_a_useful_auc_on_an_easy_case(self, method):
        reference, X, y = reference_and_test()
        scorer = fit_anomaly_scorer(reference, method=method)
        result = evaluate_anomaly_scores(scorer.score(X), y)
        assert result["roc_auc"] > 0.9, f"{method} failed an easy separation"

    def test_mahalanobis_handles_collinear_features(self):
        """Betti-curve samples are near-duplicates by construction, so the covariance is
        near-singular and an unshrunk inverse would be dominated by noise directions."""
        rng = np.random.default_rng(1)
        base = rng.normal(size=(400, 3))
        # Twelve columns spanning only three genuine dimensions.
        reference = np.hstack([base + rng.normal(0, 1e-6, base.shape) for _ in range(4)])
        scorer = fit_anomaly_scorer(reference, method="mahalanobis")
        scores = scorer.score(reference)
        assert np.isfinite(scores).all()
        assert scorer.meta["condition_number"] < 1e12

    def test_unknown_method_is_rejected(self):
        reference, _, _ = reference_and_test()
        with pytest.raises(ValueError, match="unknown method"):
            fit_anomaly_scorer(reference, method="not_a_method")

    def test_scorer_is_deterministic(self):
        reference, X, _ = reference_and_test()
        a = fit_anomaly_scorer(reference, method="mahalanobis", seed=0).score(X)
        b = fit_anomaly_scorer(reference, method="mahalanobis", seed=0).score(X)
        assert np.allclose(a, b)


class TestEvaluation:
    def test_precision_at_budget_matches_its_definition(self):
        scores = np.array([9.0, 8.0, 7.0, 1.0, 0.5])
        labels = np.array([1, 1, 0, 0, 0])
        result = evaluate_anomaly_scores(scores, labels, budgets=(2, 3))
        assert result["precision_at_2"] == pytest.approx(1.0)
        assert result["precision_at_3"] == pytest.approx(2 / 3)
        assert result["recall_at_2"] == pytest.approx(1.0)

    def test_lift_is_precision_over_base_rate(self):
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        labels = np.array([1, 0, 0, 0, 0])
        result = evaluate_anomaly_scores(scores, labels, budgets=(1,))
        assert result["base_rate"] == pytest.approx(0.2)
        assert result["lift_at_1"] == pytest.approx(1.0 / 0.2)

    def test_budget_larger_than_the_sample_is_skipped(self):
        scores = np.array([3.0, 2.0, 1.0])
        labels = np.array([1, 0, 0])
        result = evaluate_anomaly_scores(scores, labels, budgets=(2, 500))
        assert "precision_at_2" in result
        assert "precision_at_500" not in result

    def test_single_class_labels_are_rejected(self):
        with pytest.raises(ValueError, match="single class"):
            evaluate_anomaly_scores(np.arange(5.0), np.zeros(5, int))
