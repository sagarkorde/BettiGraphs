"""Tests for Betti-curve construction.

These pin two corrections that together decide whether a Betti curve carries any information at
all on graph data:

1. **Essential classes are counted from their birth, not from the start of the filtration.** In a
   one-dimensional complex no cycle can ever die, so every H1 class is essential. Counting them
   from t = 0 collapses beta_1 to a constant equal to the cycle rank -- the very scalar that sits
   in the classical baseline, making the curve worthless.

2. **Vertices are present from the start; only edges are filtered.** Letting a vertex enter with
   its first incident edge makes beta_0 identically 1 for any subgraph that ends up connected,
   discarding the merge structure that makes beta_0 informative.

Both bugs produced flat curves that looked plausible on a plot.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.features.topological import betti_curve, default_grid
from src.topology.persistence import flag_persistence


def ranked(graph: nx.Graph) -> dict[tuple, float]:
    """Edges entering in order, spread evenly over [0, 1]."""
    edges = [tuple(sorted(e)) for e in graph.edges()]
    return {e: i / max(len(edges) - 1, 1) for i, e in enumerate(edges)}


def constant(graph: nx.Graph, value: float) -> dict[tuple, float]:
    return {tuple(sorted(e)): value for e in graph.edges()}


class TestEssentialClassBirths:
    def test_late_cycle_is_absent_before_its_birth(self):
        graph = nx.cycle_graph(6)
        result = flag_persistence(graph, constant(graph, 0.9), max_dim=1, complex_dim=1)
        curve = betti_curve(result, 1, default_grid(11))
        assert curve[0] == 0, "a cycle born at 0.9 must not be counted at t=0"
        assert curve[-1] == 1

    def test_essential_births_are_recorded(self):
        graph = nx.cycle_graph(5)
        result = flag_persistence(graph, constant(graph, 0.7), max_dim=1, complex_dim=1)
        assert result.essential_births[1].size == 1
        assert result.essential_births[1][0] == pytest.approx(0.7)

    def test_one_dimensional_complex_has_only_essential_h1(self):
        """No 2-simplices means no cycle can die, so every H1 class is essential."""
        graph = nx.gnm_random_graph(30, 60, seed=4)
        result = flag_persistence(graph, ranked(graph), max_dim=1, complex_dim=1)
        assert result.diagrams[1].shape[0] == 0
        components = nx.number_connected_components(graph)
        cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + components
        assert result.essential[1] == cycle_rank

    def test_beta1_curve_is_not_constant_under_a_ranked_filtration(self):
        """The regression this guards: a flat curve carries nothing the cycle rank does not."""
        graph = nx.gnm_random_graph(40, 90, seed=5)
        result = flag_persistence(graph, ranked(graph), max_dim=1, complex_dim=1)
        curve = betti_curve(result, 1, default_grid(32))
        assert curve.std() > 0, "beta_1 curve must vary across the filtration"
        assert curve[0] < curve[-1]

    def test_beta1_curve_is_non_decreasing_for_a_one_dimensional_complex(self):
        """With no 2-simplices, cycles accumulate and never disappear."""
        graph = nx.gnm_random_graph(35, 80, seed=6)
        result = flag_persistence(graph, ranked(graph), max_dim=1, complex_dim=1)
        curve = betti_curve(result, 1, default_grid(48))
        assert np.all(np.diff(curve) >= -1e-9)


class TestVertexEntryTime:
    def test_beta0_starts_at_the_vertex_count_and_merges_to_component_count(self):
        graph = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
        result = flag_persistence(graph, constant(graph, 0.5), max_dim=1, complex_dim=2)
        curve = betti_curve(result, 0, default_grid(11))
        assert curve[0] == graph.number_of_nodes() == 6
        assert curve[-1] == nx.number_connected_components(graph) == 2

    def test_beta0_curve_is_non_increasing(self):
        """Components only ever merge as edges are added."""
        graph = nx.gnm_random_graph(40, 70, seed=7)
        result = flag_persistence(graph, ranked(graph), max_dim=1, complex_dim=1)
        curve = betti_curve(result, 0, default_grid(48))
        assert np.all(np.diff(curve) <= 1e-9)

    def test_beta0_curve_is_not_constant(self):
        graph = nx.gnm_random_graph(40, 70, seed=8)
        result = flag_persistence(graph, ranked(graph), max_dim=1, complex_dim=1)
        curve = betti_curve(result, 0, default_grid(32))
        assert curve.std() > 0

    def test_legacy_vertex_entry_makes_beta0_constant(self):
        """Documents precisely what the default avoids."""
        graph = nx.path_graph(12)
        result = flag_persistence(graph, ranked(graph), max_dim=1, complex_dim=1,
                                  vertices_at_zero=False)
        curve = betti_curve(result, 0, default_grid(16))
        assert curve.max() - curve.min() <= 1, "legacy mode collapses the beta_0 curve"

    def test_final_betti_numbers_are_unaffected_by_vertex_entry_time(self):
        """Vertex entry time changes the curve, never the homology of the final complex."""
        graph = nx.complete_multipartite_graph(2, 2, 2)
        values = constant(graph, 0.5)
        modern = flag_persistence(graph, values, max_dim=2, complex_dim=3)
        legacy = flag_persistence(graph, values, max_dim=2, complex_dim=3,
                                  vertices_at_zero=False)
        for dim in (0, 1, 2):
            assert modern.betti_at(1.0, dim) == legacy.betti_at(1.0, dim)
