"""The classical graph-structural feature family (family **S**).

Computed on **exactly the same ego-subgraph** that the topological signature is computed on. That
is deliberate and it is what makes the S-versus-T comparison meaningful: if the two families saw
different neighbourhoods, any difference between them would confound the representation with the
region of the graph it describes.

Family S is intended to be a *strong* baseline, not a straw man. In particular it includes:

* **cycle rank** ``|E| - |V| + beta_0`` -- which for a one-dimensional complex is exactly beta_1.
  If the Betti *curve* carries nothing beyond this single number, the comparison will show it.
* **triangle and wedge counts**, the structural quantities filled 2-simplices encode.
* **degree asymmetry and reciprocity**, which recover the flow direction that simplicial homology
  necessarily discards.

Anything cheap that a reviewer could reasonably ask "wouldn't a simple graph statistic have found
this?" about belongs here, so that the question is answered by the ablation rather than argued.
"""

from __future__ import annotations

import logging
from typing import Sequence

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

#: Above this order, exact betweenness is replaced by a sampled estimate. Betweenness is
#: O(|V||E|), which is untenable on a 23,000-node ego-subgraph.
BETWEENNESS_EXACT_MAX = 300
#: Pivot count for the sampled betweenness estimate.
BETWEENNESS_PIVOTS = 64
#: Above this order, the diameter is estimated by double sweep rather than computed exactly.
DIAMETER_EXACT_MAX = 500
#: Above this order the two extreme Laplacian eigenvalues are not computed and are reported as
#: NaN, with ``ego_spectral_computed`` set to 0. A dense symmetric eigensolve is O(n^3); the
#: ego-subgraph order distribution is heavily right-tailed, so a cap keeps the tail from
#: dominating total runtime.
SPECTRAL_MAX_ORDER = 600

FEATURE_NAMES: tuple[str, ...] = (
    # seed-centric
    "seed_degree", "seed_in_degree", "seed_out_degree", "seed_degree_asymmetry",
    "seed_weighted_degree", "seed_weight_mean", "seed_weight_max", "seed_weight_min",
    "seed_clustering", "seed_triangles", "seed_core_number",
    "seed_pagerank", "seed_betweenness", "seed_closeness", "seed_eigenvector",
    "seed_neighbour_degree_mean", "seed_neighbour_degree_max",
    # ego-net global
    "ego_order", "ego_size", "ego_density", "ego_n_components", "ego_cycle_rank",
    "ego_degree_mean", "ego_degree_std", "ego_degree_max", "ego_degree_gini",
    "ego_clustering_mean", "ego_transitivity", "ego_assortativity",
    "ego_diameter", "ego_radius_est", "ego_triangles", "ego_wedges", "ego_squares",
    "ego_max_core", "ego_spectral_computed", "ego_weight_sum", "ego_weight_mean", "ego_weight_std",
    "ego_weight_max", "ego_weight_gini", "ego_timestep_span",
    "ego_reciprocity_proxy", "ego_leaf_fraction", "ego_degree_two_fraction",
    "ego_algebraic_connectivity", "ego_spectral_radius",
)


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array; 0.0 for degenerate input."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    values = np.sort(np.clip(values, 0.0, None))
    total = values.sum()
    if total <= 0.0:
        return 0.0
    n = values.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * values)) / (n * total) - (n + 1.0) / n)


def _double_sweep_diameter(graph: nx.Graph) -> float:
    """Lower-bound diameter estimate by double sweep, on the largest component."""
    if graph.number_of_nodes() == 0:
        return 0.0
    component = max(nx.connected_components(graph), key=len)
    sub = graph.subgraph(component)
    start = next(iter(sub.nodes()))
    far = max(nx.single_source_shortest_path_length(sub, start).items(), key=lambda kv: kv[1])
    lengths = nx.single_source_shortest_path_length(sub, far[0])
    return float(max(lengths.values()) if lengths else 0.0)


def _count_squares(graph: nx.Graph, *, block_bytes: int = 64 << 20) -> float:
    """Number of 4-cycles, via the trace identity.

    ``#C4 = (1/8) * [ tr(A^4) - 2m - 2 * sum_i d_i (d_i - 1) ]``

    ``tr(A^4)`` equals the squared Frobenius norm of ``A^2``, which is accumulated **in row
    blocks** so that ``A^2`` is never materialised. That is not a micro-optimisation: on a
    25,000-node hub-heavy ego-subgraph from the address graph, ``A @ A`` has 307 million nonzeros
    and needs 2.29 GiB, and across 32 parallel workers that exhausts memory outright. It did.

    The identity is verified in the test suite against closed forms -- a 4-cycle gives 1, ``K4``
    gives 3, ``K3,3`` gives 9, ``K5`` gives 15 -- and against brute-force enumeration.

    An even earlier implementation enumerated all node pairs and intersected neighbour sets, which
    is O(n^2) in Python and was the single dominant cost of the whole pipeline.

    Args:
        block_bytes: soft memory budget per row block.
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n < 4 or m < 4:
        return 0.0

    adjacency = nx.to_scipy_sparse_array(graph, format="csr", dtype=np.float64, weight=None)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()

    # Rows per block, from the budget and the worst-case width of a block of A^2.
    rows_per_block = max(1, min(n, int(block_bytes / max(8 * n, 1))))
    trace_a4 = 0.0
    for start in range(0, n, rows_per_block):
        block = adjacency[start:start + rows_per_block] @ adjacency
        trace_a4 += float(block.multiply(block).sum())
        del block

    correction = 2.0 * m + 2.0 * float(np.sum(degrees * (degrees - 1.0)))
    return max((trace_a4 - correction) / 8.0, 0.0)


def classical_features(
    graph: nx.Graph,
    *,
    seed: int | None = None,
    weight_attr: str = "weight",
    time_attr: str = "timestamp",
) -> tuple[np.ndarray, list[str]]:
    """Compute family S on an ego-subgraph.

    Args:
        graph: the ego-subgraph, with ``weight`` and ``timestamp`` edge attributes and
            ``in_degree`` / ``out_degree`` node attributes.
        seed: the focal node. Defaults to ``graph.graph["seed"]``.

    Returns:
        ``(vector, names)``, with ``names == list(FEATURE_NAMES)``. Undefined quantities become
        0.0 rather than NaN, except where a quantity was *refused* on cost grounds, which yields
        NaN so that an imputation step can see it.
    """
    if seed is None:
        seed = graph.graph.get("seed")
    values: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}

    order = graph.number_of_nodes()
    size = graph.number_of_edges()
    values["ego_order"] = float(order)
    values["ego_size"] = float(size)
    if order == 0:
        return np.array([values[n] for n in FEATURE_NAMES], dtype=float), list(FEATURE_NAMES)

    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    values["ego_density"] = float(nx.density(graph))
    n_components = nx.number_connected_components(graph)
    values["ego_n_components"] = float(n_components)
    values["ego_cycle_rank"] = float(size - order + n_components)
    values["ego_degree_mean"] = float(degrees.mean())
    values["ego_degree_std"] = float(degrees.std())
    values["ego_degree_max"] = float(degrees.max())
    values["ego_degree_gini"] = _gini(degrees)
    values["ego_leaf_fraction"] = float(np.mean(degrees == 1))
    values["ego_degree_two_fraction"] = float(np.mean(degrees <= 2))

    triangles = nx.triangles(graph)
    values["ego_triangles"] = float(sum(triangles.values()) / 3.0)
    wedges_per_node = degrees * (degrees - 1.0) / 2.0
    values["ego_wedges"] = float(np.sum(wedges_per_node))
    values["ego_squares"] = _count_squares(graph)

    # Clustering and transitivity are derived from the triangle counts already computed, rather
    # than by calling nx.clustering and nx.transitivity. Those two took 23 seconds EACH on a
    # 24,000-node address ego-subgraph -- against 0.04 s for nx.triangles on the same graph -- and
    # together were 86% of the cost of the whole classical feature family. The formulas are exact:
    #   c_i          = 2 t_i / (d_i (d_i - 1))
    #   transitivity = 3T / (number of connected triples) = sum(t_i) / sum(wedges_i)
    # Both are checked against networkx in the test suite.
    node_order = list(graph.nodes())
    triangle_counts = np.array([triangles[node] for node in node_order], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_node_clustering = np.where(degrees > 1, triangle_counts / wedges_per_node, 0.0)
    clustering = dict(zip(node_order, per_node_clustering))
    values["ego_clustering_mean"] = float(np.mean(per_node_clustering))
    total_wedges = float(wedges_per_node.sum())
    values["ego_transitivity"] = (float(triangle_counts.sum() / total_wedges)
                                  if total_wedges > 0 else 0.0)
    try:
        # Regular graphs give 0/0 here; networkx returns NaN via a numpy warning rather than
        # raising, so the error state is suppressed and the NaN handled explicitly below.
        with np.errstate(invalid="ignore", divide="ignore"):
            assortativity = nx.degree_assortativity_coefficient(graph)
        values["ego_assortativity"] = 0.0 if not np.isfinite(assortativity) else float(assortativity)
    except (ZeroDivisionError, ValueError, nx.NetworkXError):
        values["ego_assortativity"] = 0.0

    core = nx.core_number(graph)
    values["ego_max_core"] = float(max(core.values()))

    if order <= DIAMETER_EXACT_MAX and n_components == 1:
        values["ego_diameter"] = float(nx.diameter(graph))
        values["ego_radius_est"] = float(nx.radius(graph))
    else:
        estimate = _double_sweep_diameter(graph)
        values["ego_diameter"] = estimate
        values["ego_radius_est"] = estimate / 2.0

    weights = np.array([d.get(weight_attr, 0.0) for _, _, d in graph.edges(data=True)], dtype=float)
    if weights.size:
        values["ego_weight_sum"] = float(weights.sum())
        values["ego_weight_mean"] = float(weights.mean())
        values["ego_weight_std"] = float(weights.std())
        values["ego_weight_max"] = float(weights.max())
        values["ego_weight_gini"] = _gini(weights)
    times = np.array([d.get(time_attr, np.nan) for _, _, d in graph.edges(data=True)], dtype=float)
    finite_times = times[np.isfinite(times)]
    if finite_times.size:
        values["ego_timestep_span"] = float(finite_times.max() - finite_times.min())

    # --- spectral: the two extreme Laplacian eigenvalues ---
    # A dense symmetric eigensolve is used rather than sparse ARPACK. ARPACK with which="SM" is
    # notoriously slow to converge without shift-invert, and measurement showed it dominating the
    # whole feature pipeline; a dense solve on the sizes that actually occur here (median
    # ego-subgraph order is in the single digits) is far cheaper.
    #
    # Above the cap the pair is left as NaN rather than 0.0, and the indicator below records that.
    # Substituting 0.0 would assert "disconnected, no expansion", which is a specific and usually
    # false claim, and would silently encode ego-subgraph size into two feature columns.
    if 3 <= order <= SPECTRAL_MAX_ORDER:
        try:
            laplacian = nx.laplacian_matrix(graph).toarray().astype(np.float64)
            spectrum = np.linalg.eigvalsh(laplacian)
            values["ego_algebraic_connectivity"] = float(spectrum[1])
            values["ego_spectral_radius"] = float(spectrum[-1])
            values["ego_spectral_computed"] = 1.0
        except (nx.NetworkXError, np.linalg.LinAlgError, ValueError, MemoryError):
            values["ego_algebraic_connectivity"] = float("nan")
            values["ego_spectral_radius"] = float("nan")
            values["ego_spectral_computed"] = 0.0
    elif order > SPECTRAL_MAX_ORDER:
        values["ego_algebraic_connectivity"] = float("nan")
        values["ego_spectral_radius"] = float("nan")
        values["ego_spectral_computed"] = 0.0
    else:
        values["ego_spectral_computed"] = 1.0  # trivially, for an order below 3

    # --- seed-centric ---
    if seed is not None and graph.has_node(seed):
        values["seed_degree"] = float(graph.degree(seed))
        in_degree = float(graph.nodes[seed].get("in_degree", 0.0))
        out_degree = float(graph.nodes[seed].get("out_degree", 0.0))
        values["seed_in_degree"] = in_degree
        values["seed_out_degree"] = out_degree
        denominator = in_degree + out_degree
        values["seed_degree_asymmetry"] = ((out_degree - in_degree) / denominator
                                          if denominator > 0 else 0.0)
        # A crude reciprocity proxy: how balanced the ego-net's directed degrees are overall.
        ins = np.array([graph.nodes[n].get("in_degree", 0.0) for n in graph.nodes()], dtype=float)
        outs = np.array([graph.nodes[n].get("out_degree", 0.0) for n in graph.nodes()], dtype=float)
        totals = ins + outs
        with np.errstate(invalid="ignore", divide="ignore"):
            balance = np.where(totals > 0, np.minimum(ins, outs) / np.maximum(totals, 1e-12), 0.0)
        values["ego_reciprocity_proxy"] = float(np.mean(balance))

        incident = np.array([d.get(weight_attr, 0.0) for _, _, d in graph.edges(seed, data=True)],
                            dtype=float)
        if incident.size:
            values["seed_weighted_degree"] = float(incident.sum())
            values["seed_weight_mean"] = float(incident.mean())
            values["seed_weight_max"] = float(incident.max())
            values["seed_weight_min"] = float(incident.min())
        values["seed_clustering"] = float(clustering.get(seed, 0.0))
        values["seed_triangles"] = float(triangles.get(seed, 0))
        values["seed_core_number"] = float(core.get(seed, 0))

        neighbour_degrees = np.array([graph.degree(n) for n in graph.neighbors(seed)], dtype=float)
        if neighbour_degrees.size:
            values["seed_neighbour_degree_mean"] = float(neighbour_degrees.mean())
            values["seed_neighbour_degree_max"] = float(neighbour_degrees.max())

        try:
            pagerank = nx.pagerank(graph, max_iter=100, tol=1e-6)
            values["seed_pagerank"] = float(pagerank.get(seed, 0.0))
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            values["seed_pagerank"] = 1.0 / order

        if order <= BETWEENNESS_EXACT_MAX:
            betweenness = nx.betweenness_centrality(graph)
        else:
            betweenness = nx.betweenness_centrality(
                graph, k=min(BETWEENNESS_PIVOTS, order), seed=0
            )
        values["seed_betweenness"] = float(betweenness.get(seed, 0.0))

        # Closeness of the seed alone, from one BFS. nx.closeness_centrality computes it for
        # every node, which is O(V*E) when a single source suffices.
        lengths = nx.single_source_shortest_path_length(graph, seed)
        reachable = len(lengths) - 1
        total_distance = float(sum(lengths.values()))
        if reachable > 0 and total_distance > 0:
            # Wasserman-Faust normalisation, so the value is comparable across components of
            # different size within a disconnected ego-subgraph.
            values["seed_closeness"] = float(
                (reachable / total_distance) * (reachable / max(order - 1, 1))
            )

        component = nx.node_connected_component(graph, seed)
        if len(component) >= 3:
            try:
                eigenvector = nx.eigenvector_centrality_numpy(
                    graph.subgraph(component), max_iter=200
                )
                values["seed_eigenvector"] = float(eigenvector.get(seed, 0.0))
            except (nx.NetworkXError, np.linalg.LinAlgError, TypeError, ValueError, RuntimeError):
                values["seed_eigenvector"] = 0.0

    return np.array([values[name] for name in FEATURE_NAMES], dtype=float), list(FEATURE_NAMES)


def feature_names() -> Sequence[str]:
    return FEATURE_NAMES
