"""Leakage guards.

Train/test leakage is the failure mode most likely to produce an impressive, wrong result that
survives review. These tests assert the four properties the pre-registration commits to, so a
change that breaks one fails the suite rather than quietly inflating a score.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import splits as sp
from src.features import topological as topo
from src.topology.persistence import persistence_of


def _synthetic(n_per_step: int = 40, n_steps: int = 49, seed: int = 0):
    rng = np.random.default_rng(seed)
    time_steps = np.repeat(np.arange(1, n_steps + 1), n_per_step)
    labels = rng.choice([sp.ILLICIT, sp.LICIT, sp.UNKNOWN], size=time_steps.size,
                        p=[0.1, 0.3, 0.6])
    # Guarantee illicit examples in every step so the split is constructible.
    for step in range(1, n_steps + 1):
        index = np.flatnonzero(time_steps == step)[:4]
        labels[index] = sp.ILLICIT
    return time_steps, labels


class TestTemporalSplit:
    def test_splits_are_disjoint_in_rows_and_in_time(self):
        time_steps, labels = _synthetic()
        split = sp.temporal_split(time_steps, labels)
        split.assert_disjoint()  # raises on failure
        assert not (split.train & split.val).any()
        assert not (split.train & split.test).any()
        assert not (split.val & split.test).any()

    def test_split_windows_match_the_preregistration(self):
        time_steps, labels = _synthetic()
        split = sp.temporal_split(time_steps, labels)
        assert time_steps[split.train].min() == 1
        assert time_steps[split.train].max() == 34
        assert time_steps[split.val].min() == 35
        assert time_steps[split.val].max() == 41
        assert time_steps[split.test].min() == 42
        assert time_steps[split.test].max() == 49

    def test_no_train_step_is_at_or_after_any_test_step(self):
        """The defining property of a temporal split: the past never reaches the future."""
        time_steps, labels = _synthetic()
        split = sp.temporal_split(time_steps, labels)
        assert time_steps[split.train].max() < time_steps[split.test].min()
        assert time_steps[split.val].max() < time_steps[split.test].min()

    def test_unlabelled_rows_are_excluded_from_supervised_masks(self):
        time_steps, labels = _synthetic()
        split = sp.temporal_split(time_steps, labels, labelled_only=True)
        for mask in (split.train, split.val, split.test):
            assert not (labels[mask] == sp.UNKNOWN).any()

    def test_overlapping_windows_are_rejected(self):
        time_steps, labels = _synthetic()
        with pytest.raises(AssertionError):
            sp.temporal_split(time_steps, labels, train=(1, 40), val=(35, 41), test=(42, 49))

    def test_window_without_illicit_examples_is_rejected(self):
        time_steps = np.repeat(np.arange(1, 50), 10)
        labels = np.full(time_steps.size, sp.LICIT, dtype=int)
        labels[time_steps <= 34] = sp.ILLICIT  # only the train window has positives
        with pytest.raises(ValueError, match="no illicit examples"):
            sp.temporal_split(time_steps, labels)

    def test_regime_shift_sits_inside_the_test_window(self):
        assert sp.TEST_STEPS[0] <= sp.REGIME_SHIFT_STEP <= sp.TEST_STEPS[1]

    def test_pre_post_shift_masks_partition_the_test_window(self):
        time_steps, labels = _synthetic()
        split = sp.temporal_split(time_steps, labels)
        pre, post = sp.pre_post_shift_masks(time_steps, split.test)
        assert not (pre & post).any()
        assert (pre | post).sum() == split.test.sum()
        assert time_steps[post].min() >= sp.REGIME_SHIFT_STEP
        assert time_steps[pre].max() < sp.REGIME_SHIFT_STEP


class TestTopologicalFeaturesCannotLeak:
    """The topological descriptors are fitted to nothing, which is the strongest guarantee.

    Under rank normalisation the Betti-curve grid and persistence-image range are fixed constants
    on [0, 1], so no descriptor depends on any statistic pooled across a split boundary. These
    tests pin that property rather than trusting the docstring.
    """

    def _graph(self, seed: int, scale: float):
        import networkx as nx

        graph = nx.gnm_random_graph(60, 140, seed=seed)
        rng = np.random.default_rng(seed)
        for u, v in graph.edges():
            graph[u][v]["weight"] = float(rng.exponential(scale))
        return graph

    def test_signature_is_invariant_to_monotone_rescaling_of_weights(self):
        """Rank normalisation must make the signature depend on edge *order*, not on units.

        This is why a scaler fitted on training data is unnecessary for T, and therefore why T
        cannot leak distributional information across the split.
        """
        graph = self._graph(seed=3, scale=1.0)
        scaled = graph.copy()
        for u, v in scaled.edges():
            scaled[u][v]["weight"] *= 1000.0  # strictly monotone, order-preserving

        a, names_a = topo.betti_signature(
            persistence_of(graph, "value_superlevel", max_dim=1), max_dim=1
        )
        b, names_b = topo.betti_signature(
            persistence_of(scaled, "value_superlevel", max_dim=1), max_dim=1
        )
        assert names_a == names_b
        assert np.allclose(a, b), "rank-normalised signature changed under a monotone rescaling"

    def test_signature_of_one_graph_does_not_depend_on_any_other_graph(self):
        """Computed independently per entity, so no cross-entity statistic can creep in."""
        first = persistence_of(self._graph(1, 1.0), "value_superlevel", max_dim=1)
        alone, _ = topo.betti_signature(first, max_dim=1)

        # Compute a batch in a different order; the target's vector must be identical.
        vectors = {}
        for seed in (5, 1, 9):
            result = persistence_of(self._graph(seed, 1.0 + seed), "value_superlevel", max_dim=1)
            vectors[seed], _ = topo.betti_signature(result, max_dim=1)
        assert np.allclose(alone, vectors[1])

    def test_non_rank_normalisation_refuses_a_default_grid(self):
        """A fitted grid is a leakage hazard, so it must be passed explicitly, never inferred."""
        result = persistence_of(self._graph(2, 1.0), "value_superlevel", max_dim=1,
                                normalise="minmax")
        with pytest.raises(ValueError, match="leakage"):
            topo.betti_signature(result, max_dim=1)
