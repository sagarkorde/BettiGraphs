"""Known-answer tests for the classical feature family.

The 4-cycle count uses the trace identity
``#C4 = (1/8)[tr(A^4) - 2m - 2*sum_i d_i(d_i-1)]``, which replaced an O(n^2) pair enumeration
that had become the dominant cost of the whole feature pipeline. It is exact, so it is pinned
here against graphs whose 4-cycle counts are known in closed form.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.features.classical import FEATURE_NAMES, _count_squares, classical_features


class TestSquareCount:
    @pytest.mark.parametrize("graph,expected", [
        (nx.cycle_graph(4), 1),
        (nx.complete_graph(4), 3),
        (nx.complete_bipartite_graph(2, 3), 3),
        (nx.complete_bipartite_graph(3, 3), 9),
        (nx.complete_graph(5), 15),
        (nx.cycle_graph(6), 0),
        (nx.path_graph(10), 0),
        (nx.star_graph(8), 0),
    ])
    def test_matches_closed_form(self, graph, expected):
        assert _count_squares(graph) == pytest.approx(float(expected))

    def test_matches_brute_force_on_random_graphs(self):
        """Independent check against the common-neighbour enumeration it replaced."""
        for seed in range(6):
            graph = nx.gnm_random_graph(24, 60, seed=seed)
            adjacency = {n: set(graph.neighbors(n)) for n in graph.nodes()}
            nodes = list(graph.nodes())
            brute = 0
            for i, u in enumerate(nodes):
                for v in nodes[i + 1:]:
                    common = len(adjacency[u] & adjacency[v])
                    brute += common * (common - 1) // 2
            brute /= 2.0
            assert _count_squares(graph) == pytest.approx(brute, abs=1e-6)

    def test_too_small_graphs_give_zero(self):
        assert _count_squares(nx.complete_graph(3)) == 0.0
        assert _count_squares(nx.empty_graph(5)) == 0.0


class TestClassicalFeatures:
    def _ego(self, graph, seed=0):
        for node in graph.nodes():
            graph.nodes[node]["in_degree"] = graph.degree(node)
            graph.nodes[node]["out_degree"] = graph.degree(node)
        for u, v in graph.edges():
            graph[u][v]["weight"] = 1.0
            graph[u][v]["timestamp"] = 5.0
        graph.graph["seed"] = seed
        return graph

    def test_vector_length_matches_names_and_is_finite(self):
        vector, names = classical_features(self._ego(nx.gnm_random_graph(50, 120, seed=1)))
        assert len(vector) == len(names) == len(FEATURE_NAMES)
        assert np.isfinite(vector).all()

    def test_cycle_rank_is_correct(self):
        graph = self._ego(nx.cycle_graph(7))
        vector, names = classical_features(graph)
        assert vector[names.index("ego_cycle_rank")] == pytest.approx(1.0)

    def test_cycle_rank_on_a_tree_is_zero(self):
        vector, names = classical_features(self._ego(nx.balanced_tree(2, 3)))
        assert vector[names.index("ego_cycle_rank")] == pytest.approx(0.0)

    def test_component_count_on_a_disjoint_union(self):
        graph = self._ego(nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(5)))
        vector, names = classical_features(graph)
        assert vector[names.index("ego_n_components")] == pytest.approx(2.0)

    def test_triangle_count_on_a_complete_graph(self):
        graph = self._ego(nx.complete_graph(5))
        vector, names = classical_features(graph)
        assert vector[names.index("ego_triangles")] == pytest.approx(10.0)

    def test_empty_graph_returns_the_zero_vector(self):
        vector, names = classical_features(nx.Graph())
        assert len(vector) == len(FEATURE_NAMES)
        assert np.allclose(vector, 0.0)

    def test_algebraic_connectivity_of_a_disconnected_graph_is_zero(self):
        graph = self._ego(nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5)))
        vector, names = classical_features(graph)
        assert vector[names.index("ego_algebraic_connectivity")] == pytest.approx(0.0, abs=1e-6)

    def test_spectral_radius_of_a_complete_graph(self):
        """The largest Laplacian eigenvalue of K_n is n."""
        graph = self._ego(nx.complete_graph(8))
        vector, names = classical_features(graph)
        assert vector[names.index("ego_spectral_radius")] == pytest.approx(8.0, rel=1e-4)

    def test_seed_closeness_is_in_unit_interval(self):
        for n, m, seed in ((40, 90, 2), (100, 300, 3)):
            vector, names = classical_features(self._ego(nx.gnm_random_graph(n, m, seed=seed)))
            value = vector[names.index("seed_closeness")]
            assert 0.0 <= value <= 1.0

    def test_feature_names_are_unique(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


class TestDerivedClusteringMatchesNetworkX:
    """Clustering and transitivity are derived from triangle counts, not read from networkx.

    nx.clustering and nx.transitivity each took 23 seconds on a 24,000-node address ego-subgraph,
    against 0.04 s for nx.triangles on the same graph, and together were 86% of the cost of the
    whole classical feature family. The closed forms are exact, so these tests hold them to
    networkx's own values rather than to a tolerance chosen for convenience.
    """

    def _prepared(self, graph):
        for node in graph.nodes():
            graph.nodes[node]["in_degree"] = graph.degree(node)
            graph.nodes[node]["out_degree"] = graph.degree(node)
        for u, v in graph.edges():
            graph[u][v]["weight"] = 1.0
            graph[u][v]["timestamp"] = 1.0
        graph.graph["seed"] = next(iter(graph.nodes()))
        return graph

    @pytest.mark.parametrize("graph", [
        nx.gnm_random_graph(40, 90, seed=1),
        nx.complete_graph(6),
        nx.cycle_graph(8),
        nx.barabasi_albert_graph(200, 3, seed=2),
        nx.star_graph(20),
        nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)),
        nx.path_graph(15),
        nx.complete_bipartite_graph(4, 5),
    ])
    def test_matches_networkx_exactly(self, graph):
        graph = self._prepared(graph)
        vector, names = classical_features(graph)
        assert vector[names.index("ego_clustering_mean")] == pytest.approx(
            nx.average_clustering(graph), abs=1e-12)
        assert vector[names.index("ego_transitivity")] == pytest.approx(
            nx.transitivity(graph), abs=1e-12)
        assert vector[names.index("seed_clustering")] == pytest.approx(
            nx.clustering(graph, graph.graph["seed"]), abs=1e-12)

    def test_isolated_and_degree_one_nodes_give_zero_clustering(self):
        graph = self._prepared(nx.star_graph(6))
        vector, names = classical_features(graph)
        assert vector[names.index("ego_clustering_mean")] == pytest.approx(0.0)
