"""Known-answer tests for retrieval evaluation.

H3 rests entirely on these numbers, and retrieval metrics are easy to get subtly wrong: an
off-by-one in the ranking, a query retrieving itself, or the wrong denominator in average precision
all produce plausible values rather than obvious errors. Every case here is small enough to verify
by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.similarity.topological import (
    evaluate_retrieval, pairwise_euclidean, proxy_fidelity, top_k_diagram,
)


class TestPairwiseEuclidean:
    def test_matches_the_definition(self):
        points = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
        distances = pairwise_euclidean(points)
        assert distances[0, 1] == pytest.approx(5.0)
        assert distances[0, 2] == pytest.approx(1.0)
        assert distances[1, 2] == pytest.approx(np.hypot(2.0, 4.0))

    def test_diagonal_is_finite_and_negligible(self):
        """Cancellation makes the squared diagonal slightly negative before the clip.

        The clip keeps the square root real, so no NaN reaches the caller. The residue is around
        1e-7 rather than exactly zero, which is left alone deliberately: zeroing it would only be
        possible in the symmetric call form and would make the two forms disagree. Callers exclude
        a query from its own ranking by index, not by relying on a zero distance.
        """
        rng = np.random.default_rng(0)
        distances = pairwise_euclidean(rng.normal(size=(50, 8)))
        assert np.isfinite(distances).all(), "clip must prevent NaN from a negative square"
        assert np.all(np.diag(distances) < 1e-5)
        assert (distances >= 0.0).all()

    def test_query_form_matches_the_full_matrix(self):
        rng = np.random.default_rng(1)
        pool = rng.normal(size=(30, 5))
        full = pairwise_euclidean(pool)
        subset = pairwise_euclidean(pool, pool[[2, 7, 11]])
        assert np.allclose(subset, full[[2, 7, 11]])

    def test_is_symmetric(self):
        rng = np.random.default_rng(2)
        distances = pairwise_euclidean(rng.normal(size=(20, 4)))
        assert np.allclose(distances, distances.T)


class TestEvaluateRetrieval:
    def test_perfect_ranking_gives_precision_one(self):
        """Two same-class items ranked first for a query with exactly two relevant items."""
        pool_labels = np.array([1, 1, 0, 0, 0])
        # Query is pool item 0; items 1 (same class) then the rest.
        distances = np.array([[0.0, 0.1, 0.5, 0.6, 0.7]])
        result = evaluate_retrieval(distances, np.array([1]), pool_labels, space="t",
                                    ks=(1,), exclude_self=np.array([0]))
        assert result.precision_at[1] == pytest.approx(1.0)

    def test_worst_ranking_gives_precision_zero(self):
        pool_labels = np.array([1, 1, 0, 0, 0])
        # The only other relevant item is ranked last.
        distances = np.array([[0.0, 0.9, 0.1, 0.2, 0.3]])
        result = evaluate_retrieval(distances, np.array([1]), pool_labels, space="t",
                                    ks=(1,), exclude_self=np.array([0]))
        assert result.precision_at[1] == pytest.approx(0.0)

    def test_self_is_excluded_from_its_own_ranking(self):
        """Without exclusion a query retrieves itself first and precision is inflated.

        Query is pool item 0, illicit. The only other illicit item is ranked last. With self
        excluded, precision@1 must be 0; without exclusion the query retrieves itself at distance
        zero and precision@1 would be 1.
        """
        pool_labels = np.array([1, 0, 0, 1])
        distances = np.array([[0.0, 0.1, 0.2, 0.9]])

        excluded = evaluate_retrieval(distances, np.array([1]), pool_labels, space="t",
                                      ks=(1,), exclude_self=np.array([0]))
        assert excluded.precision_at[1] == pytest.approx(0.0)

        not_excluded = evaluate_retrieval(distances, np.array([1]), pool_labels, space="t",
                                          ks=(1,))
        assert not_excluded.precision_at[1] == pytest.approx(1.0)

    def test_query_with_no_relevant_item_is_skipped(self):
        pool_labels = np.array([1, 0, 0, 0])
        distances = np.array([[0.0, 0.1, 0.2, 0.3], [0.3, 0.0, 0.1, 0.2]])
        query_labels = np.array([1, 0])
        result = evaluate_retrieval(distances, query_labels, pool_labels, space="t",
                                    ks=(1,), exclude_self=np.array([0, 1]))
        # The illicit query has no other illicit item, so only the licit query is scored.
        assert result.n_queries == 1

    def test_recall_at_k_uses_the_relevant_total(self):
        pool_labels = np.array([1, 1, 1, 0, 0])
        distances = np.array([[0.0, 0.1, 0.2, 0.9, 0.9]])
        result = evaluate_retrieval(distances, np.array([1]), pool_labels, space="t",
                                    ks=(1, 2), exclude_self=np.array([0]))
        # Two relevant items remain after self-exclusion.
        assert result.recall_at[1] == pytest.approx(0.5)
        assert result.recall_at[2] == pytest.approx(1.0)

    def test_ndcg_is_one_for_a_perfect_ranking(self):
        pool_labels = np.array([1, 1, 1, 0, 0])
        distances = np.array([[0.0, 0.1, 0.2, 0.9, 0.95]])
        result = evaluate_retrieval(distances, np.array([1]), pool_labels, space="t",
                                    ks=(2,), exclude_self=np.array([0]))
        assert result.ndcg_at[2] == pytest.approx(1.0)

    def test_ndcg_is_lower_for_a_worse_ranking(self):
        pool_labels = np.array([1, 1, 1, 0, 0])
        good = np.array([[0.0, 0.1, 0.2, 0.9, 0.95]])
        bad = np.array([[0.0, 0.9, 0.95, 0.1, 0.2]])
        a = evaluate_retrieval(good, np.array([1]), pool_labels, space="t", ks=(2,),
                               exclude_self=np.array([0]))
        b = evaluate_retrieval(bad, np.array([1]), pool_labels, space="t", ks=(2,),
                               exclude_self=np.array([0]))
        assert a.ndcg_at[2] > b.ndcg_at[2]

    def test_random_distances_land_near_the_base_rate(self):
        """The floor the H3 interpretation depends on: shuffled distances score the base rate."""
        rng = np.random.default_rng(3)
        n, base = 4000, 0.05
        pool_labels = (rng.random(n) < base).astype(int)
        queries = np.flatnonzero(pool_labels == 1)[:200]
        distances = rng.random((queries.size, n))
        result = evaluate_retrieval(distances, pool_labels[queries], pool_labels, space="random",
                                    ks=(10,), exclude_self=queries)
        assert result.precision_at[10] == pytest.approx(base, abs=0.03)

    def test_informative_distances_beat_the_base_rate(self):
        rng = np.random.default_rng(4)
        n, base = 4000, 0.05
        pool_labels = (rng.random(n) < base).astype(int)
        # Place same-class items close together along one axis.
        coordinates = pool_labels.astype(float) + rng.normal(0, 0.1, n)
        queries = np.flatnonzero(pool_labels == 1)[:200]
        distances = np.abs(coordinates[None, :] - coordinates[queries][:, None])
        result = evaluate_retrieval(distances, pool_labels[queries], pool_labels, space="t",
                                    ks=(10,), exclude_self=queries)
        assert result.precision_at[10] > 0.8

    def test_all_queries_unscorable_is_an_error(self):
        pool_labels = np.array([0, 0, 0])
        distances = np.array([[0.0, 0.1, 0.2]])
        with pytest.raises(ValueError, match="no query"):
            evaluate_retrieval(distances, np.array([1]), pool_labels, space="t", ks=(1,))


class TestDiagramTruncation:
    def test_keeps_the_most_persistent_classes(self):
        pairs = np.array([[0.0, 0.1], [0.0, 0.9], [0.0, 0.5], [0.0, 0.2]])
        kept = top_k_diagram(pairs, k=2)
        lifetimes = sorted((kept[:, 1] - kept[:, 0]).tolist())
        assert lifetimes == pytest.approx([0.5, 0.9])

    def test_short_diagrams_pass_through_unchanged(self):
        pairs = np.array([[0.0, 0.3], [0.1, 0.4]])
        assert np.array_equal(top_k_diagram(pairs, k=10), pairs)

    def test_empty_diagram_gives_an_empty_result(self):
        assert top_k_diagram(np.empty((0, 2)), k=5).shape == (0, 2)


class TestProxyFidelity:
    def test_perfect_agreement(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = proxy_fidelity(values, values * 3.0)
        assert result["spearman"] == pytest.approx(1.0)

    def test_reversed_agreement_is_negative(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert proxy_fidelity(values, -values)["spearman"] == pytest.approx(-1.0)

    def test_too_few_points_gives_nan(self):
        result = proxy_fidelity(np.array([1.0]), np.array([2.0]))
        assert np.isnan(result["spearman"])
