"""Persistent homology of transaction graphs.

Two constructions are supported, matching the two kinds of filtration in
:mod:`src.topology.filtrations`:

* **Flag (clique) complex.** Vertices and edges are inserted with their filtration values and
  the complex is expanded to dimension ``d``; each higher simplex inherits the maximum value
  of its faces. This is the natural construction when the filtration is defined on edges
  (transaction value, time, coreness).
* **Vietoris-Rips complex.** Built from a pairwise distance matrix. Used for the geodesic and
  effective-resistance filtrations.

Directionality
--------------
Bitcoin transaction graphs are directed, and ordinary simplicial homology is not. Both
constructions here therefore act on the underlying undirected graph, which is the standard
choice in network persistence (Horak et al. 2009) but does discard flow direction. Direction is
not thrown away project-wide: it is retained as explicit asymmetry features in
:mod:`src.features.classical` (in/out degree ratio, reciprocity, and so on), and the loss is
stated as a limitation in the manuscript rather than glossed over. Directed flag complexes are
noted as future work.

Dimension and bipartiteness
---------------------------
A bipartite graph has no triangles, so its flag complex is one-dimensional and
``beta_p == 0`` for all ``p >= 2`` *by construction, not as a finding*. The
address-transaction view ``G_bip`` is bipartite, so :func:`flag_persistence` warns if asked
for ``max_dim >= 2`` on a bipartite input rather than returning a column of decorative zeros.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import gudhi
import networkx as nx
import numpy as np

from src.topology import filtrations as filt

logger = logging.getLogger(__name__)

#: Guard on flag-complex expansion. A dense ego-subgraph can blow up combinatorially; above
#: this many simplices the expansion is aborted so a single pathological wallet cannot stall
#: a whole run. Aborted cases are counted and reported, never silently dropped.
MAX_SIMPLICES = 4_000_000


@dataclass
class PersistenceResult:
    """Persistence diagrams and the bookkeeping needed to interpret them.

    Attributes:
        diagrams: ``{homology_dimension: array of shape (k, 2)}`` of (birth, death) pairs.
            Essential classes (infinite death) are **excluded** here and counted in
            ``essential`` instead, so that downstream statistics never mix finite lifetimes
            with an arbitrary infinity replacement.
        essential: ``{homology_dimension: count}`` of classes that never die. For dimension 0
            this is the number of connected components of the final complex.
        max_filtration: the largest finite filtration value reached, needed to place Betti
            curves on a common grid.
        n_simplices: size of the final complex, recorded for the scalability analysis.
        n_vertices: order of the input graph.
        truncated: True if expansion hit :data:`MAX_SIMPLICES` and was abandoned.
    """

    diagrams: dict[int, np.ndarray]
    essential: dict[int, int]
    max_filtration: float
    n_simplices: int
    n_vertices: int
    truncated: bool = False
    meta: dict = field(default_factory=dict)
    essential_births: dict[int, np.ndarray] = field(default_factory=dict)

    def betti_at(self, t: float, dim: int) -> int:
        """Betti number in dimension ``dim`` at filtration value ``t``.

        Essential classes are counted **from their birth**, not from the start of the filtration.
        That distinction is not a detail here: in a one-dimensional complex no cycle can ever die,
        so every H1 class is essential, and treating them all as present from t = 0 would flatten
        the entire beta_1 curve into a constant equal to the cycle rank -- discarding precisely the
        multiscale information the curve exists to capture.
        """
        births = self.essential_births.get(dim)
        if births is None or births.size == 0:
            count = self.essential.get(dim, 0) if t >= self.meta.get("min_filtration", 0.0) else 0
        else:
            count = int(np.count_nonzero(births <= t))
        pairs = self.diagrams.get(dim)
        if pairs is None or pairs.size == 0:
            return count
        return int(count + np.count_nonzero((pairs[:, 0] <= t) & (pairs[:, 1] > t)))


def _split_diagrams(
    simplex_tree: "gudhi.SimplexTree", max_dim: int
) -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, np.ndarray], float]:
    """Separate finite from essential classes, **keeping the essential classes' birth times**.

    Discarding those births would be a silent correctness bug rather than a simplification: in a
    one-dimensional complex every H1 class is essential, so the birth times are the entire
    multiscale content of beta_1.
    """
    finite: dict[int, list[tuple[float, float]]] = {d: [] for d in range(max_dim + 1)}
    essential_births: dict[int, list[float]] = {d: [] for d in range(max_dim + 1)}
    largest = 0.0
    for dim in range(max_dim + 1):
        for birth, death in simplex_tree.persistence_intervals_in_dimension(dim):
            if np.isinf(death):
                essential_births[dim].append(float(birth))
                largest = max(largest, float(birth))
            else:
                finite[dim].append((float(birth), float(death)))
                largest = max(largest, float(death))
    diagrams = {
        dim: (np.asarray(pairs, dtype=float) if pairs else np.empty((0, 2), dtype=float))
        for dim, pairs in finite.items()
    }
    births = {
        dim: (np.asarray(values, dtype=float) if values else np.empty(0, dtype=float))
        for dim, values in essential_births.items()
    }
    counts = {dim: int(values.size) for dim, values in births.items()}
    return diagrams, counts, births, largest


def flag_persistence(
    graph: nx.Graph,
    edge_values: Mapping[tuple, float],
    *,
    max_dim: int = 2,
    complex_dim: int | None = None,
    max_simplices: int = MAX_SIMPLICES,
    vertices_at_zero: bool = True,
) -> PersistenceResult:
    """Persistent homology of the flag-complex filtration induced by ``edge_values``.

    Vertex entry time
    -----------------
    ``vertices_at_zero=True`` (the default) places every vertex at filtration value 0 and filters
    only the **edges**. This is the standard weight-rank clique filtration for networks, and it is
    the faithful model here: the entities in an ego-subgraph all exist at every scale, and what is
    being thresholded is the strength of the relationships between them. Under it ``beta_0``
    starts at the vertex count and merges downward, which is single-linkage clustering by edge
    value and is genuinely informative.

    The alternative -- vertices entering with their first incident edge -- makes ``beta_0``
    identically 1 for any subgraph that ends up connected, because a vertex can never be isolated.
    That collapses the whole ``beta_0`` curve to a constant and discards the merge structure, so it
    is available but is not the default.

    ``complex_dim`` is a genuine modelling choice, not an implementation detail, because it
    changes what ``H_1`` means:

    * ``complex_dim=1`` keeps the graph's 1-skeleton. Then ``H_1`` is the full cycle space and
      ``beta_1 = |E| - |V| + beta_0`` exactly -- the classical cycle rank.
    * ``complex_dim>=2`` fills in triangles (and higher cliques). Then ``H_1`` counts cycles
      *modulo* those filled cliques, so a triangular transaction loop no longer registers.

    Neither is more correct; they are the homology of different complexes and they measure
    different things. Which one carries forensic signal is an empirical question, so both are
    exercised in the ablation rather than one being assumed.

    Args:
        graph: undirected graph.
        edge_values: ``{(u, v): filtration value}``, as returned by
            :func:`src.topology.filtrations.edge_filtration`. Keys are order-insensitive.
        max_dim: highest homology dimension to compute.
        complex_dim: dimension to expand the flag complex to. Defaults to ``max_dim + 1``,
            which is what exact ``H_{max_dim}`` requires, since killing a ``p``-cycle needs
            ``(p+1)``-simplices.
        max_simplices: abort guard; see :data:`MAX_SIMPLICES`.

    Returns:
        A :class:`PersistenceResult`.
    """
    if complex_dim is None:
        complex_dim = max_dim + 1
    if complex_dim < max_dim:
        raise ValueError(
            f"complex_dim={complex_dim} cannot be below max_dim={max_dim}: H_{max_dim} is "
            "identically zero in a complex of lower dimension"
        )
    if max_dim >= 2 and graph.number_of_edges() > 0 and nx.is_bipartite(graph):
        logger.warning(
            "graph is bipartite (%d nodes): its flag complex is 1-dimensional, so beta_p is "
            "identically zero for p >= 2 by construction, not as a result",
            graph.number_of_nodes(),
        )

    tree = gudhi.SimplexTree()
    normalised: dict[tuple, float] = {}
    for (u, v), value in edge_values.items():
        normalised[(u, v) if u <= v else (v, u)] = float(value)

    if vertices_at_zero:
        for node in graph.nodes():
            tree.insert([node], 0.0)
    else:
        vertex_value: dict = {}
        for (u, v), value in normalised.items():
            for node in (u, v):
                if node not in vertex_value or value < vertex_value[node]:
                    vertex_value[node] = value
        for node in graph.nodes():
            tree.insert([node], float(vertex_value.get(node, 0.0)))
    for (u, v), value in normalised.items():
        tree.insert([u, v], value)

    truncated = False
    if complex_dim >= 2:
        tree.expansion(complex_dim)
        if tree.num_simplices() > max_simplices:
            logger.warning(
                "flag complex reached %d simplices on a %d-node graph, exceeding the %d guard; "
                "recording as truncated",
                tree.num_simplices(),
                graph.number_of_nodes(),
                max_simplices,
            )
            truncated = True

    tree.compute_persistence(persistence_dim_max=True)
    diagrams, essential, births, largest = _split_diagrams(tree, max_dim)
    return PersistenceResult(
        diagrams=diagrams,
        essential=essential,
        essential_births=births,
        max_filtration=largest,
        n_simplices=tree.num_simplices(),
        n_vertices=graph.number_of_nodes(),
        truncated=truncated,
        meta={"construction": "flag", "complex_dim": complex_dim},
    )


def rips_persistence(
    matrix: np.ndarray,
    *,
    max_dim: int = 2,
    complex_dim: int | None = None,
    max_edge_length: float | None = None,
    max_simplices: int = MAX_SIMPLICES,
) -> PersistenceResult:
    """Persistent homology of the Vietoris-Rips filtration of a distance matrix.

    Args:
        matrix: square, symmetric, zero-diagonal distance matrix.
        max_dim: highest homology dimension to compute.
        complex_dim: dimension to build the complex to; defaults to ``max_dim + 1``. See
            :func:`flag_persistence` for why this is a modelling choice.
        max_edge_length: truncate the filtration here; defaults to the matrix maximum.
        max_simplices: abort guard; see :data:`MAX_SIMPLICES`.
    """
    if complex_dim is None:
        complex_dim = max_dim + 1
    if complex_dim < max_dim:
        raise ValueError(f"complex_dim={complex_dim} cannot be below max_dim={max_dim}")
    n = matrix.shape[0]
    if n == 0:
        return PersistenceResult({d: np.empty((0, 2)) for d in range(max_dim + 1)},
                                 {d: 0 for d in range(max_dim + 1)}, 0.0, 0, 0,
                                 meta={"construction": "rips"},
                                 essential_births={d: np.empty(0) for d in range(max_dim + 1)})
    threshold = float(matrix.max()) if max_edge_length is None else float(max_edge_length)

    # Pre-emptive size guard. The MAX_SIMPLICES check below fires only after the complex is built,
    # which is too late: a Vietoris-Rips complex on a few hundred points can reach tens of millions
    # of simplices, and the memory is committed during construction. Observed in practice --
    # 10,194,159 simplices from 394 points -- and with 32 workers that exhausts the machine before
    # any guard downstream can speak.
    #
    # The binomial bound C(n, d+1) is the worst case at the requested dimension. Refusing here
    # raises MemoryError, which the extraction layer records as a per-entity refusal and reports as
    # a rate, so an unaffordable filtration shows up in the selection table as poor coverage rather
    # than as a dead run.
    if complex_dim >= 2:
        from math import comb

        worst_case = comb(n, min(complex_dim + 1, n))
        if worst_case > max_simplices:
            raise MemoryError(
                f"Vietoris-Rips on {n} points at dimension {complex_dim} admits up to "
                f"{worst_case:.3g} simplices, above the {max_simplices:.3g} guard; refusing "
                "before construction rather than exhausting memory during it"
            )

    rips = gudhi.RipsComplex(distance_matrix=matrix, max_edge_length=threshold)
    tree = rips.create_simplex_tree(max_dimension=complex_dim)

    truncated = tree.num_simplices() > max_simplices
    if truncated:
        logger.warning(
            "Rips complex reached %d simplices on %d points, exceeding the %d guard",
            tree.num_simplices(), n, max_simplices,
        )

    tree.compute_persistence(persistence_dim_max=True)
    diagrams, essential, births, largest = _split_diagrams(tree, max_dim)
    return PersistenceResult(
        diagrams=diagrams,
        essential=essential,
        essential_births=births,
        max_filtration=max(largest, 0.0),
        n_simplices=tree.num_simplices(),
        n_vertices=n,
        truncated=truncated,
        meta={"construction": "rips", "threshold": threshold, "complex_dim": complex_dim},
    )


def persistence_of(
    graph: nx.Graph,
    filtration: str,
    *,
    max_dim: int = 2,
    complex_dim: int | None = None,
    weight: str = "weight",
    timestamp: str = "timestamp",
    normalise: filt.Normalisation = "rank",
    rips_weight: str | None = None,
) -> PersistenceResult:
    """Compute persistent homology of ``graph`` under the named filtration.

    Dispatches to :func:`flag_persistence` or :func:`rips_persistence` according to whether
    ``filtration`` is edge-valued or matrix-valued.

    Raises:
        ValueError: if ``filtration`` is not in :data:`src.topology.filtrations.ALL_FILTRATIONS`.
    """
    undirected = graph.to_undirected(as_view=False) if graph.is_directed() else graph

    if filtration in filt.EDGE_FILTRATIONS:
        values = filt.edge_filtration(
            undirected, filtration, weight=weight, timestamp=timestamp, normalise=normalise
        )
        result = flag_persistence(undirected, values, max_dim=max_dim, complex_dim=complex_dim)
    elif filtration in filt.MATRIX_FILTRATIONS:
        matrix, _nodes = filt.distance_matrix(
            undirected, filtration, weight=rips_weight, normalise=normalise
        )
        result = rips_persistence(matrix, max_dim=max_dim, complex_dim=complex_dim)
    else:
        raise ValueError(
            f"unknown filtration {filtration!r}; expected one of {filt.ALL_FILTRATIONS}"
        )

    result.meta["filtration"] = filtration
    result.meta["normalise"] = normalise
    return result
