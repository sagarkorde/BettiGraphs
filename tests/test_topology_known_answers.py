"""Known-answer tests for the topology layer.

Every assertion here is a homology fact that can be checked by hand, so a regression in the
filtration or persistence code fails loudly rather than producing plausible-looking rubbish.
These run in seconds and gate every commit that touches ``src/topology`` or
``src/features/topological.py``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.features import topological as topo
from src.topology import filtrations as filt
from src.topology.persistence import flag_persistence, persistence_of, rips_persistence


def _constant_values(graph: nx.Graph) -> dict[tuple, float]:
    """All edges at filtration time 1.0, so the final complex is the whole flag complex."""
    return {tuple(sorted(e)): 1.0 for e in graph.edges()}


def _final_betti(graph: nx.Graph, max_dim: int = 2) -> list[int]:
    """Betti numbers of the fully-built flag complex, read off the persistence result."""
    result = flag_persistence(graph, _constant_values(graph), max_dim=max_dim)
    # At a filtration time past every death, only essential classes survive.
    return [result.betti_at(1.0, dim) for dim in range(max_dim + 1)]


class TestBettiNumbers:
    def test_cycle_graph_has_one_loop(self):
        assert _final_betti(nx.cycle_graph(5)) == [1, 1, 0]

    def test_tree_is_contractible(self):
        assert _final_betti(nx.path_graph(10)) == [1, 0, 0]

    def test_disjoint_triangles_count_components_and_fill_in(self):
        graph = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
        # Each triangle becomes a filled 2-simplex, so no loop survives.
        assert _final_betti(graph) == [2, 0, 0]

    def test_complete_graph_clique_complex_is_contractible(self):
        assert _final_betti(nx.complete_graph(4)) == [1, 0, 0]
        assert _final_betti(nx.complete_graph(6)) == [1, 0, 0]

    def test_wedge_of_two_cycles_has_two_loops(self):
        left = nx.cycle_graph(4)
        right = nx.relabel_nodes(nx.cycle_graph(4), {i: (0 if i == 0 else i + 3) for i in range(4)})
        assert _final_betti(nx.compose(left, right)) == [1, 2, 0]

    def test_octahedron_clique_complex_is_a_two_sphere(self):
        """K(2,2,2) has eight triangular faces and no K4, so its clique complex is S^2."""
        assert _final_betti(nx.complete_multipartite_graph(2, 2, 2)) == [1, 0, 1]

    @pytest.mark.parametrize("n_nodes,n_edges,seed", [(40, 70, 7), (60, 120, 11), (25, 40, 3)])
    def test_cycle_rank_identity_on_one_dimensional_complex(self, n_nodes, n_edges, seed):
        """With ``complex_dim=1`` the complex is the graph itself, so beta_1 = |E| - |V| + beta_0.

        This is the identity that motivates using the whole Betti *curve* rather than a single
        beta_1: one value carries no more information than the classical cycle rank, which is
        already in the baseline feature family.
        """
        graph = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
        result = flag_persistence(graph, _constant_values(graph), max_dim=1, complex_dim=1)
        beta0 = result.betti_at(1.0, 0)
        beta1 = result.betti_at(1.0, 1)
        assert beta1 == graph.number_of_edges() - graph.number_of_nodes() + beta0

    def test_filling_triangles_kills_triangular_cycles(self):
        """``complex_dim`` is a real modelling choice: it changes what beta_1 counts.

        A single triangle has one cycle in its 1-skeleton and none once the face is filled.
        """
        triangle = nx.complete_graph(3)
        values = _constant_values(triangle)
        graph_only = flag_persistence(triangle, values, max_dim=1, complex_dim=1)
        filled = flag_persistence(triangle, values, max_dim=1, complex_dim=2)
        assert graph_only.betti_at(1.0, 1) == 1
        assert filled.betti_at(1.0, 1) == 0

    def test_complex_dim_below_max_dim_is_rejected(self):
        graph = nx.complete_graph(4)
        with pytest.raises(ValueError, match="cannot be below max_dim"):
            flag_persistence(graph, _constant_values(graph), max_dim=2, complex_dim=1)

    def test_beta0_equals_component_count(self):
        graph = nx.disjoint_union_all([nx.cycle_graph(4), nx.path_graph(3), nx.complete_graph(5)])
        result = flag_persistence(graph, _constant_values(graph), max_dim=1)
        assert result.essential[0] == nx.number_connected_components(graph) == 3


class TestBipartiteDegeneracy:
    def test_bipartite_flag_complex_has_no_high_homology(self):
        """A bipartite graph has no triangles, so beta_p == 0 for p >= 2 by construction.

        The address-transaction view is bipartite, which is why beta_2 is not reported there.
        """
        graph = nx.complete_bipartite_graph(4, 5)
        assert nx.is_bipartite(graph)
        betti = _final_betti(graph, max_dim=2)
        assert betti[2] == 0

    def test_bipartite_input_warns_when_asked_for_dimension_two(self, caplog):
        graph = nx.complete_bipartite_graph(3, 3)
        with caplog.at_level("WARNING"):
            flag_persistence(graph, _constant_values(graph), max_dim=2)
        assert any("bipartite" in record.message for record in caplog.records)


class TestFiltrations:
    def test_rank_normalisation_lands_in_unit_interval(self):
        graph = nx.gnm_random_graph(30, 60, seed=5)
        rng = np.random.default_rng(0)
        for u, v in graph.edges():
            graph[u][v]["weight"] = float(rng.exponential(10.0))
        values = np.array(list(filt.edge_filtration(graph, "value_superlevel").values()))
        assert values.min() >= 0.0 and values.max() <= 1.0

    def test_value_superlevel_puts_high_value_edges_first(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=100.0)
        graph.add_edge(1, 2, weight=1.0)
        values = filt.edge_filtration(graph, "value_superlevel", normalise="rank")
        assert values[(0, 1)] < values[(1, 2)]

    def test_temporal_sublevel_orders_by_time(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, timestamp=10.0)
        graph.add_edge(1, 2, timestamp=5.0)
        values = filt.edge_filtration(graph, "temporal_sublevel", normalise="rank")
        assert values[(1, 2)] < values[(0, 1)]

    def test_missing_attribute_on_every_edge_is_an_error(self):
        graph = nx.cycle_graph(4)
        with pytest.raises(ValueError, match="absent from every edge"):
            filt.edge_filtration(graph, "value_superlevel")

    def test_geodesic_distance_matrix_is_a_metric_shape(self):
        graph = nx.cycle_graph(6)
        matrix, nodes = filt.distance_matrix(graph, "geodesic_rips", normalise="none")
        assert matrix.shape == (6, 6)
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.allclose(matrix, matrix.T)
        assert matrix[nodes.index(0), nodes.index(3)] == pytest.approx(3.0)

    def test_disconnected_pairs_get_finite_sentinel_not_infinity(self):
        graph = nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))
        matrix, _ = filt.distance_matrix(graph, "geodesic_rips", normalise="none")
        assert np.isfinite(matrix).all()
        assert matrix[0, 3] > matrix[0, 2]

    def test_effective_resistance_on_a_path_is_additive(self):
        """On a path, effective resistance equals hop count (unit resistors in series)."""
        graph = nx.path_graph(4)
        matrix, nodes = filt.distance_matrix(graph, "resistance_rips", normalise="none")
        assert matrix[nodes.index(0), nodes.index(3)] == pytest.approx(3.0, abs=1e-8)

    def test_resistance_refuses_oversized_graphs(self):
        graph = nx.path_graph(filt.RESISTANCE_MAX_NODES + 2)
        with pytest.raises(MemoryError, match="effective resistance"):
            filt.distance_matrix(graph, "resistance_rips")

    def test_rips_on_cycle_recovers_the_loop(self):
        graph = nx.cycle_graph(10)
        matrix, _ = filt.distance_matrix(graph, "geodesic_rips", normalise="none")
        result = rips_persistence(matrix, max_dim=1)
        # A discretised circle must carry at least one 1-cycle somewhere in the filtration.
        assert result.diagrams[1].shape[0] >= 1

    def test_unknown_filtration_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown filtration"):
            persistence_of(nx.cycle_graph(4), "not_a_filtration")


class TestBettiSignature:
    def _result(self):
        graph = nx.gnm_random_graph(40, 90, seed=13)
        rng = np.random.default_rng(1)
        for u, v in graph.edges():
            graph[u][v]["weight"] = float(rng.exponential(5.0))
        return persistence_of(graph, "value_superlevel", max_dim=1)

    def test_signature_length_matches_names(self):
        vector, names = topo.betti_signature(self._result(), max_dim=1)
        assert vector.size == len(names) > 0
        assert len(set(names)) == len(names), "feature names must be unique"

    def test_signature_is_finite(self):
        vector, _ = topo.betti_signature(self._result(), max_dim=1)
        assert np.isfinite(vector).all()

    def test_signature_length_is_stable_across_graphs(self):
        """Fixed length regardless of graph order -- the whole point of vectorisation."""
        sizes = []
        for n, m, seed in [(20, 40, 1), (80, 200, 2), (300, 900, 3)]:
            graph = nx.gnm_random_graph(n, m, seed=seed)
            rng = np.random.default_rng(seed)
            for u, v in graph.edges():
                graph[u][v]["weight"] = float(rng.exponential(5.0))
            result = persistence_of(graph, "value_superlevel", max_dim=1)
            vector, _ = topo.betti_signature(result, max_dim=1)
            sizes.append(vector.size)
        assert len(set(sizes)) == 1

    def test_betti_curve_of_a_cycle_reaches_one(self):
        graph = nx.cycle_graph(8)
        result = flag_persistence(graph, _constant_values(graph), max_dim=1)
        curve = topo.betti_curve(result, 1, topo.default_grid(16))
        assert curve.max() >= 1.0

    def test_entropy_is_zero_without_finite_classes(self):
        result = flag_persistence(nx.path_graph(5), _constant_values(nx.path_graph(5)), max_dim=1)
        assert topo.persistence_entropy(result, 1) == 0.0

    def test_entropy_is_higher_for_spread_lifetimes(self):
        from src.topology.persistence import PersistenceResult

        concentrated = PersistenceResult({1: np.array([[0.0, 1.0], [0.0, 0.01]])}, {1: 0}, 1.0, 0, 0)
        spread = PersistenceResult({1: np.array([[0.0, 0.5], [0.0, 0.5]])}, {1: 0}, 1.0, 0, 0)
        assert topo.persistence_entropy(spread, 1) > topo.persistence_entropy(concentrated, 1)

    def test_euler_curve_matches_alternating_sum(self):
        graph = nx.complete_multipartite_graph(2, 2, 2)
        result = flag_persistence(graph, _constant_values(graph), max_dim=2)
        grid = topo.default_grid(8)
        chi = topo.euler_curve(result, grid, max_dim=2)
        manual = (
            topo.betti_curve(result, 0, grid)
            - topo.betti_curve(result, 1, grid)
            + topo.betti_curve(result, 2, grid)
        )
        assert np.allclose(chi, manual)

    def test_persistence_image_is_nonnegative_and_sized(self):
        image = topo.persistence_image(self._result(), 1, resolution=8)
        assert image.size == 64
        assert (image >= 0.0).all()

    def test_landscapes_are_ordered_pointwise(self):
        result = self._result()
        flat = topo.persistence_landscape(result, 1, num_landscapes=3, resolution=16)
        first, second, third = flat[:16], flat[16:32], flat[32:48]
        assert (first >= second - 1e-12).all()
        assert (second >= third - 1e-12).all()

    def test_explicit_grid_required_for_non_rank_normalisation(self):
        graph = nx.gnm_random_graph(20, 40, seed=4)
        rng = np.random.default_rng(2)
        for u, v in graph.edges():
            graph[u][v]["weight"] = float(rng.exponential(5.0))
        result = persistence_of(graph, "value_superlevel", max_dim=1, normalise="minmax")
        with pytest.raises(ValueError, match="leakage"):
            topo.betti_signature(result, max_dim=1)
