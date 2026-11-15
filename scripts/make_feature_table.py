"""Generate the feature-definitions table, checked against the code that produces the features.

A replication needs to know what every column is, and the brief asks for the table. The risk with
such a table is that it drifts: a feature is renamed or added and the table still describes the old
set. So the names are not typed here. They are obtained by running the two extractors on a small
graph and read off the output, and a definition that has no matching feature -- or a feature that
has no definition -- fails the build rather than being quietly dropped.

The topological columns are described by group rather than individually. There are 444 of them and
they are generated systematically: a name like ``pl_h1_07`` is the seventh sample of the first
landscape function in dimension 1, and enumerating all 444 would be less informative than stating
the construction once.

Usage::

    .venv\\Scripts\\python.exe scripts\\make_feature_table.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.classical import classical_features  # noqa: E402
from src.features.topological import betti_signature, default_grid  # noqa: E402
from src.topology.persistence import flag_persistence  # noqa: E402

TABLES = REPO_ROOT / "tables"
logger = logging.getLogger("features")

#: Definitions for the classical structural family. "Seed" means the entity itself; "ego" means its
#: radius-r neighbourhood as an induced undirected subgraph.
CLASSICAL: dict[str, str] = {
    "seed_degree": "Degree of the seed node in the undirected ego-subgraph.",
    "seed_in_degree": "In-degree of the seed in the original directed graph.",
    "seed_out_degree": "Out-degree of the seed in the original directed graph.",
    "seed_degree_asymmetry": "(out − in) / (out + in) for the seed; 0 when both are zero.",
    "seed_weighted_degree": "Sum of BTC values on the seed's incident edges.",
    "seed_weight_mean": "Mean BTC value over the seed's incident edges.",
    "seed_weight_max": "Largest BTC value on an edge incident to the seed.",
    "seed_weight_min": "Smallest BTC value on an edge incident to the seed.",
    "seed_clustering": "Local clustering coefficient of the seed.",
    "seed_triangles": "Number of triangles through the seed.",
    "seed_core_number": "k-core index of the seed.",
    "seed_pagerank": "PageRank of the seed within the ego-subgraph.",
    "seed_betweenness": "Betweenness centrality of the seed within the ego-subgraph.",
    "seed_closeness": "Closeness centrality of the seed, Wasserman–Faust normalised, from a "
                      "single-source shortest-path pass.",
    "seed_eigenvector": "Eigenvector centrality of the seed within the ego-subgraph.",
    "seed_neighbour_degree_mean": "Mean degree of the seed's neighbours.",
    "seed_neighbour_degree_max": "Largest degree among the seed's neighbours.",
    "ego_order": "Number of nodes in the ego-subgraph, |V|.",
    "ego_size": "Number of undirected edges in the ego-subgraph, |E|.",
    "ego_density": "2|E| / (|V|(|V| − 1)).",
    "ego_n_components": "Number of connected components of the ego-subgraph.",
    "ego_cycle_rank": "First Betti number of the 1-skeleton, |E| − |V| + components. Exact, not "
                      "an estimate, and the single-integer baseline the ablation uses.",
    "ego_degree_mean": "Mean node degree in the ego-subgraph.",
    "ego_degree_std": "Standard deviation of node degree.",
    "ego_degree_max": "Largest node degree.",
    "ego_degree_gini": "Gini coefficient of the degree distribution.",
    "ego_clustering_mean": "Mean local clustering coefficient, derived from triangle counts rather "
                           "than recomputed per node.",
    "ego_transitivity": "Global transitivity, 3 × triangles / wedges.",
    "ego_assortativity": "Degree assortativity coefficient.",
    "ego_diameter": "Longest shortest path in the largest component.",
    "ego_radius_est": "Eccentricity of the seed, an upper-bounded estimate of the graph radius.",
    "ego_triangles": "Total number of triangles.",
    "ego_wedges": "Total number of paths of length two.",
    "ego_squares": "Number of 4-cycles, from the trace identity "
                   "(tr A⁴ − 2|E| − 2Σd(d−1))/8, computed in row blocks.",
    "ego_max_core": "Largest k-core index present.",
    "ego_spectral_computed": "Indicator: 1 when the Laplacian spectrum was computed, 0 when the "
                             "subgraph exceeded the order cap and the spectral columns are NaN.",
    "ego_weight_sum": "Sum of BTC values over all edges.",
    "ego_weight_mean": "Mean BTC value per edge.",
    "ego_weight_std": "Standard deviation of edge BTC value.",
    "ego_weight_max": "Largest edge BTC value.",
    "ego_weight_gini": "Gini coefficient of the edge-value distribution.",
    "ego_timestep_span": "Range of time steps spanned by the edges.",
    "ego_reciprocity_proxy": "Share of adjacent pairs joined in both directions in the original "
                             "directed graph.",
    "ego_leaf_fraction": "Share of nodes with degree one.",
    "ego_degree_two_fraction": "Share of nodes with degree two.",
    "ego_algebraic_connectivity": "Second-smallest Laplacian eigenvalue, from a dense solve capped "
                                  "at order 600; NaN above the cap.",
    "ego_spectral_radius": "Largest Laplacian eigenvalue, under the same cap.",
}

#: Definitions for the topological family, by name group. The trailing index of a generated column
#: selects a sample point, so the group is the unit a reader needs.
TOPOLOGICAL: dict[str, str] = {
    "bc_h0": "Betti curve in dimension 0: the number of homology classes alive at each of the 32 "
             "filtration values. Essential classes are counted from their birth.",
    "bc_h1": "Betti curve in dimension 1, sampled the same way. On a graph this is the number of "
             "independent cycles present at each threshold.",
    "euler": "Euler characteristic curve, β₀ − β₁ at each filtration value.",
    "pi_h0": "Persistence image in dimension 0: the diagram smoothed by a Gaussian and rasterised "
             "on an 8×8 grid in birth–lifetime coordinates.",
    "pi_h1": "Persistence image in dimension 1, same construction.",
    "pl_h0": "Persistence landscape in dimension 0: the first three landscape functions, each "
             "sampled at 32 points.",
    "pl_h1": "Persistence landscape in dimension 1, same construction.",
    "h0_n_finite": "Count of finite (born-and-died) classes in dimension 0.",
    "h0_n_essential": "Count of essential classes in dimension 0 — components that never merge.",
    "h0_life_mean": "Mean lifetime (death − birth) of finite classes in dimension 0.",
    "h0_life_std": "Standard deviation of those lifetimes.",
    "h0_life_max": "Longest lifetime in dimension 0.",
    "h0_life_sum": "Total persistence: the sum of all finite lifetimes.",
    "h0_life_l2": "L2 norm of the lifetime vector.",
    "h0_life_q25": "First quartile of the lifetime distribution.",
    "h0_life_q50": "Median lifetime.",
    "h0_life_q75": "Third quartile of the lifetime distribution.",
    "h0_life_q90": "Ninetieth percentile of the lifetime distribution.",
    "h0_birth_mean": "Mean birth value of finite classes.",
    "h0_death_mean": "Mean death value of finite classes.",
    "h0_entropy": "Persistence entropy: the Shannon entropy of lifetimes normalised by their sum.",
}
TOPOLOGICAL.update({
    key.replace("h0", "h1"): value.replace("dimension 0", "dimension 1").replace(
        " — components that never merge", " — cycles that never fill in")
    for key, value in TOPOLOGICAL.items() if key.startswith("h0_")
})


def classical_names() -> list[str]:
    graph = nx.karate_club_graph()
    for u, v in graph.edges():
        graph[u][v]["weight"], graph[u][v]["timestamp"] = 1.0, 1.0
    for node in graph.nodes():
        graph.nodes[node].update(out_degree=1, in_degree=1, time_step=1)
    graph.graph["seed"] = 0
    return list(classical_features(graph)[1])


def topological_groups() -> dict[str, int]:
    graph = nx.karate_club_graph()
    values = {edge: float(index + 1) for index, edge in enumerate(graph.edges())}
    result = flag_persistence(graph, values, max_dim=1, complex_dim=2)
    names = betti_signature(result, max_dim=1, grid=default_grid(32))[1]
    groups: dict[str, int] = {}
    for name in names:
        groups[re.sub(r"_\d+$", "", name)] = groups.get(re.sub(r"_\d+$", "", name), 0) + 1
    return groups


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    names = classical_names()
    groups = topological_groups()

    problems = []
    problems += ["classical feature with no definition: " + n for n in names if n not in CLASSICAL]
    problems += ["definition with no classical feature: " + n for n in CLASSICAL if n not in names]
    problems += ["topological group with no definition: " + g for g in groups
                 if g not in TOPOLOGICAL]
    problems += ["definition with no topological group: " + g for g in TOPOLOGICAL
                 if g not in groups]
    for problem in problems:
        logger.error("%s", problem)
    if problems:
        return 1

    total_topological = sum(groups.values())
    lines = [
        "# Table 13 — Feature definitions",
        "",
        "Generated by `scripts/make_feature_table.py`, which reads the names from the extractors",
        "themselves and fails if a definition has no feature or a feature has no definition. The",
        "table cannot drift from the code.",
        "",
        "**Seed** means the entity being described; **ego** means its radius-*r* neighbourhood as",
        "an induced undirected subgraph. Directed quantities are read from the original graph,",
        "since homology requires an undirected complex but direction is still informative.",
        "",
        "## Classical structural family (S) — " + str(len(names)) + " features",
        "",
        "| Feature | Definition |",
        "|---|---|",
    ]
    lines += ["| `" + name + "` | " + CLASSICAL[name] + " |" for name in names]

    lines += [
        "",
        "## Topological family (T) — " + str(total_topological) + " columns in "
        + str(len(groups)) + " groups",
        "",
        "Columns are generated systematically: a trailing index selects a sample point, so",
        "`pl_h1_07` is the seventh sample of the dimension-1 landscape stack. The group is the unit",
        "a reader needs, and the count is how many columns it contributes.",
        "",
        "| Group | Columns | Definition |",
        "|---|---|---|",
    ]
    order = sorted(groups, key=lambda g: (-groups[g], g))
    lines += ["| `" + g + "` | " + str(groups[g]) + " | " + TOPOLOGICAL[g] + " |" for g in order]
    lines += [
        "",
        "The provided families **P** (the dataset's own transaction features) and **Pplus** (those",
        "plus the extended Elliptic++ set) are anonymised by their publishers and have no public",
        "definitions; they are used as supplied and are not redefined here.",
        "",
    ]

    path = TABLES / "table13_feature_definitions.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s: %d classical features, %d topological columns in %d groups",
                path.name, len(names), total_topological, len(groups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
