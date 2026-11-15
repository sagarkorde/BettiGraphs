"""Filtration functions on transaction graphs.

A filtration assigns a real value to every edge (and vertex) of a graph, inducing a nested
family of subgraphs ``G_0 subset G_1 subset ... subset G_m`` and, via flag-complex expansion,
a nested family of simplicial complexes. The choice of filtration is the single most
consequential modelling decision in a topological pipeline: it determines what "scale" means
and therefore what the persistence diagram is measuring.

Five candidates are implemented, each with a forensic rationale. None is privileged in code.
Which one the study uses is decided empirically on the validation split only, by the procedure
in ``docs/PREREGISTRATION.md`` -- never by how the diagrams look.

============================  ==========================================================
Filtration                    Rationale
============================  ==========================================================
``value_superlevel``          Edges enter in decreasing transaction value, so the
                              high-value backbone appears first. Layering structure in
                              laundering is value-stratified.
``temporal_sublevel``         Edges enter in increasing timestamp. Encodes flow order
                              directly; the basis of the temporal Betti experiment.
``geodesic_rips``             Vietoris-Rips on shortest-path distance. The standard
                              network-persistence construction (Horak et al. 2009).
``resistance_rips``           Vietoris-Rips on effective-resistance (commute-time)
                              distance. Reflects flow capacity rather than hop count and
                              is less brittle than geodesic distance on sparse graphs.
``core_sublevel``             Vertices enter by k-core number. Cheap; separates periphery
                              from core.
============================  ==========================================================

Normalisation
-------------
Ego-subgraphs vary in order by orders of magnitude, so raw filtration values are not
comparable across graphs. ``rank`` normalisation (the default) replaces each edge value by
its within-graph quantile, which makes the resulting Betti curves purely shape descriptors and
-- importantly -- introduces no cross-split dependence, so it cannot leak test information into
training. Absolute scale is not discarded; it is carried separately as scalar features by
:mod:`src.features.classical`, which keeps shape and scale statistically separable.
"""

from __future__ import annotations

import logging
from typing import Literal

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

Normalisation = Literal["none", "minmax", "rank"]
FiltrationName = Literal[
    "value_superlevel",
    "temporal_sublevel",
    "geodesic_rips",
    "resistance_rips",
    "core_sublevel",
]

#: Filtrations expressed as edge values, consumed by flag-complex expansion.
EDGE_FILTRATIONS: tuple[str, ...] = ("value_superlevel", "temporal_sublevel", "core_sublevel")
#: Filtrations expressed as a dense pairwise distance matrix, consumed by Vietoris-Rips.
MATRIX_FILTRATIONS: tuple[str, ...] = ("geodesic_rips", "resistance_rips")

#: Above this order, effective resistance (a dense pseudo-inverse) is refused rather than
#: silently costing O(n^3) time and O(n^2) memory. Callers should sample or use geodesic.
RESISTANCE_MAX_NODES = 1500


def _normalise(values: np.ndarray, how: Normalisation) -> np.ndarray:
    """Map raw filtration values onto a comparable scale.

    ``rank`` assigns each value its within-array quantile in ``[0, 1]``, averaging ties so
    that equal values receive equal filtration times.
    """
    if values.size == 0:
        return values.astype(float)
    values = np.asarray(values, dtype=float)
    if how == "none":
        return values
    if how == "minmax":
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)
    if how == "rank":
        order = values.argsort(kind="stable")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(values.size, dtype=float)
        # Average ranks within tied groups so ties share a filtration time.
        unique, inverse = np.unique(values, return_inverse=True)
        if unique.size < values.size:
            sums = np.bincount(inverse, weights=ranks, minlength=unique.size)
            counts = np.bincount(inverse, minlength=unique.size)
            ranks = (sums / counts)[inverse]
        denom = max(values.size - 1, 1)
        return ranks / denom
    raise ValueError(f"unknown normalisation: {how!r}")


def edge_filtration(
    graph: nx.Graph,
    name: str,
    *,
    weight: str = "weight",
    timestamp: str = "timestamp",
    normalise: Normalisation = "rank",
) -> dict[tuple[int, int], float]:
    """Return ``{edge: filtration_value}`` for an edge-valued filtration.

    The graph is treated as undirected for homology purposes; see the module note in
    :mod:`src.topology.persistence` on directionality.

    Args:
        graph: an undirected graph (or a view of one).
        name: one of :data:`EDGE_FILTRATIONS`.
        weight: edge attribute holding transaction value, for ``value_superlevel``.
        timestamp: edge attribute holding time, for ``temporal_sublevel``.
        normalise: see :data:`Normalisation`.

    Raises:
        ValueError: if ``name`` is not an edge filtration, or a required attribute is absent
            from every edge (a silent fallback to constant values would produce a meaningless
            filtration, so this is an error rather than a warning).
    """
    if name not in EDGE_FILTRATIONS:
        raise ValueError(f"{name!r} is not an edge filtration; expected one of {EDGE_FILTRATIONS}")

    edges = [tuple(sorted(e)) for e in graph.edges()]
    if not edges:
        return {}

    if name == "value_superlevel":
        raw = _attribute_array(graph, edges, weight)
        # Superlevel: large values enter first, so negate to get an ascending filtration.
        raw = -raw
    elif name == "temporal_sublevel":
        raw = _attribute_array(graph, edges, timestamp)
    else:  # core_sublevel
        core = nx.core_number(graph)
        # An edge exists once both endpoints do; a high-core edge appears late under a
        # sublevel filtration on -core, i.e. dense cores first.
        raw = np.array([-min(core[u], core[v]) for u, v in edges], dtype=float)

    return dict(zip(edges, _normalise(raw, normalise)))


def _attribute_array(graph: nx.Graph, edges: list[tuple[int, int]], attr: str) -> np.ndarray:
    values = []
    missing = 0
    for u, v in edges:
        data = graph.get_edge_data(u, v) or {}
        if attr in data and data[attr] is not None:
            values.append(float(data[attr]))
        else:
            missing += 1
            values.append(np.nan)
    array = np.asarray(values, dtype=float)
    if missing == array.size:
        raise ValueError(
            f"edge attribute {attr!r} is absent from every edge; the filtration would be "
            "constant and meaningless"
        )
    if missing:
        # Impute missing values at the median so a handful of gaps do not reorder the
        # filtration, and say so loudly.
        logger.warning("edge attribute %r missing on %d/%d edges; imputing median", attr, missing, array.size)
        array = np.where(np.isnan(array), np.nanmedian(array), array)
    return array


def distance_matrix(
    graph: nx.Graph,
    name: str,
    *,
    weight: str | None = None,
    normalise: Normalisation = "rank",
) -> tuple[np.ndarray, list]:
    """Return ``(D, nodes)`` for a Vietoris-Rips filtration.

    Disconnected pairs receive a distance strictly greater than every finite distance, so they
    never merge within the filtration. ``numpy.inf`` is avoided because GUDHI treats an
    infinite edge length as "include always" in some code paths.

    Args:
        graph: an undirected graph.
        name: one of :data:`MATRIX_FILTRATIONS`.
        weight: edge attribute to use as a length, or ``None`` for hop distance.
        normalise: see :data:`Normalisation`.

    Raises:
        ValueError: if ``name`` is not a matrix filtration.
        MemoryError: if ``resistance_rips`` is requested on a graph above
            :data:`RESISTANCE_MAX_NODES`.
    """
    if name not in MATRIX_FILTRATIONS:
        raise ValueError(f"{name!r} is not a matrix filtration; expected one of {MATRIX_FILTRATIONS}")

    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0)), nodes
    index = {node: i for i, node in enumerate(nodes)}

    if name == "geodesic_rips":
        matrix = np.full((n, n), np.nan, dtype=float)
        for source, lengths in nx.all_pairs_dijkstra_path_length(graph, weight=weight):
            i = index[source]
            for target, length in lengths.items():
                matrix[i, index[target]] = length
    else:
        if n > RESISTANCE_MAX_NODES:
            raise MemoryError(
                f"effective resistance on {n} nodes exceeds the {RESISTANCE_MAX_NODES}-node "
                "guard; sample the subgraph or use geodesic_rips"
            )
        matrix = _effective_resistance(graph, nodes, weight)

    finite = matrix[np.isfinite(matrix)]
    # Disconnected pairs sit just beyond the largest finite distance: never merged, but not
    # infinite. The factor is arbitrary but fixed and reported; it affects only the tail.
    sentinel = (float(finite.max()) * 1.10 + 1.0) if finite.size else 1.0
    matrix = np.where(np.isfinite(matrix), matrix, sentinel)
    np.fill_diagonal(matrix, 0.0)
    matrix = 0.5 * (matrix + matrix.T)  # symmetrise; Dijkstra on an undirected graph is already symmetric

    if normalise != "none":
        off = matrix[~np.eye(n, dtype=bool)]
        flat = _normalise(off, normalise)
        out = np.zeros_like(matrix)
        out[~np.eye(n, dtype=bool)] = flat
        out = 0.5 * (out + out.T)
        matrix = out

    return matrix, nodes


def _effective_resistance(graph: nx.Graph, nodes: list, weight: str | None) -> np.ndarray:
    """Effective-resistance distance via the Laplacian pseudo-inverse.

    ``R(i, j) = L+_ii + L+_jj - 2 L+_ij``. Computed per connected component; cross-component
    pairs are returned as ``inf`` for the caller to replace with a sentinel.
    """
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.full((n, n), np.inf, dtype=float)
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        sub_nodes = list(sub.nodes())
        laplacian = nx.laplacian_matrix(sub, nodelist=sub_nodes, weight=weight).todense()
        pinv = np.linalg.pinv(np.asarray(laplacian, dtype=float))
        diag = np.diag(pinv)
        block = diag[:, None] + diag[None, :] - 2.0 * pinv
        np.fill_diagonal(block, 0.0)
        ids = [index[node] for node in sub_nodes]
        matrix[np.ix_(ids, ids)] = np.clip(block, 0.0, None)
    return matrix


ALL_FILTRATIONS: tuple[str, ...] = EDGE_FILTRATIONS + MATRIX_FILTRATIONS
