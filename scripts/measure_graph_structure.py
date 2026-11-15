"""Measure the structural properties that decide whether topology can carry information here.

Section 8 of DATA_FINDINGS reports these numbers and they carry the paper's central explanation --
that the transaction and address views of the same chain behave oppositely because of density, not
method. They were originally computed ad hoc and never written to an artefact, so nothing could
check them and a clean-environment reproduction could not produce them. This script closes that
gap: it recomputes the global census and the ego-subgraph sparsity profile and writes them to
``results/processed/graph_structure.json``.

Two deliberate choices:

* **Cycle rank is computed by the Euler identity** ``beta_1 = E - V + C`` rather than by building a
  complex. On a graph the first Betti number of the 1-skeleton is exactly the cycle rank, so the
  identity is not an approximation, and it gives an independent check on the edge, node and
  component counts at the same time -- if any of the three is wrong, the identity fails.
* **Ego-subgraphs are bounded at 25,000 nodes**, the bound frozen in pre-registration Amendment 1,
  and the truncated fraction is reported beside every row. An unbounded radius-3 neighbourhood of a
  22,996-degree hub is not affordable, and silently truncating would bias cycle rank downward
  without saying so.

Usage::

    .venv\\Scripts\\python.exe scripts\\measure_graph_structure.py --sections global
    .venv\\Scripts\\python.exe scripts\\measure_graph_structure.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.graph_construction.graphs import (  # noqa: E402
    GraphStore,
    build_address_graph,
    build_tx_graph,
)
from src.utils.seeds import set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
OUT = REPO_ROOT / "results" / "processed"

#: The node bound frozen in pre-registration Amendment 1.
NODE_BOUND = 25_000
#: Sample sizes as originally reported, so the rows are comparable rather than merely recomputed.
SAMPLES = {"G_tx": 3_000, "G_addr": 1_500}
RADII = {"G_tx": (1, 2, 3, 4), "G_addr": (1, 2, 3)}

logger = logging.getLogger("structure")


def global_census(store: GraphStore) -> dict:
    """Node, edge, degree and component counts, plus the cycle rank they imply."""
    n_nodes = store.n_nodes
    degree = np.diff(store.indptr)

    # The stored adjacency is symmetric, so a non-loop undirected edge appears twice. Self-loops
    # appear once, which makes the entry count odd and makes naive halving wrong: it charges half
    # an edge for each loop. They are excluded rather than halved or counted, because
    # GraphStore.ego_subgraph admits an edge only when ``other > internal`` -- so no self-loop ever
    # reaches the complex, and a census that counted them would describe a different graph from the
    # one the persistence is computed on. The loop count is reported so the choice is visible.
    rows = np.repeat(np.arange(n_nodes, dtype=store.indices.dtype), degree)
    self_loops = int((rows == store.indices).sum())
    n_undirected = int((len(store.indices) - self_loops) // 2)

    matrix = csr_matrix((np.ones(len(store.indices), dtype=np.int8), store.indices, store.indptr),
                        shape=(n_nodes, n_nodes))
    n_components, labels = connected_components(matrix, directed=False)
    _, sizes = np.unique(labels, return_counts=True)

    cycle_rank = n_undirected - n_nodes + n_components
    return {
        "view": store.name,
        "n_nodes": n_nodes,
        "n_undirected_edges": n_undirected,
        "n_self_loops": self_loops,
        "n_adjacency_entries": int(len(store.indices)),
        "mean_degree": float(2.0 * n_undirected / n_nodes),
        "mean_degree_including_self_loops": float(len(store.indices) / n_nodes),
        "frac_degree_at_most_2": float((degree <= 2).mean()),
        "max_degree": int(degree.max()),
        "n_components": int(n_components),
        "largest_component_nodes": int(sizes.max()),
        "largest_component_share": float(sizes.max() / n_nodes),
        "n_singleton_components": int((sizes == 1).sum()),
        "global_cycle_rank": int(cycle_rank),
        "cycle_rank_identity": "beta_1 = E - V + C = "
                               f"{n_undirected} - {n_nodes} + {n_components} = {cycle_rank}",
    }


def sparsity_profile(store: GraphStore, node_ids: np.ndarray, radii, seed: int) -> list[dict]:
    """Ego-subgraph order, size and cycle rank by radius, over a fixed random sample."""
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    sample = rng.choice(node_ids, size=min(len(node_ids), SAMPLES[store.name]), replace=False)

    rows: list[dict] = []
    for radius in radii:
        orders, sizes, ranks, truncated = [], [], [], 0
        started = time.perf_counter()
        for node_id in sample:
            graph = store.ego_subgraph(node_id, radius, max_nodes=NODE_BOUND, relabel=True)
            order = graph.number_of_nodes()
            size = graph.number_of_edges()
            # Components of the ego-subgraph itself: a bounded traversal can disconnect it.
            components = 1
            if order:
                import networkx as nx
                components = nx.number_connected_components(graph)
            orders.append(order)
            sizes.append(size)
            ranks.append(size - order + components)
            truncated += int(graph.graph.get("truncated", False))
        orders_a, sizes_a, ranks_a = np.array(orders), np.array(sizes), np.array(ranks)
        rows.append({
            "view": store.name,
            "radius": radius,
            "n_sampled": int(len(sample)),
            "median_nodes": float(np.median(orders_a)),
            "median_edges": float(np.median(sizes_a)),
            "frac_with_any_cycle": float((ranks_a > 0).mean()),
            "mean_cycle_rank": float(ranks_a.mean()),
            "median_cycle_rank": float(np.median(ranks_a)),
            "frac_truncated_at_bound": float(truncated / len(sample)),
            "node_bound": NODE_BOUND,
            "seconds": float(time.perf_counter() - started),
        })
        logger.info("%s r=%d: median %d nodes / %d edges, %.1f%% with a cycle, "
                    "mean cycle rank %.2f, %.1f%% truncated (%.1fs)",
                    store.name, radius, rows[-1]["median_nodes"], rows[-1]["median_edges"],
                    100 * rows[-1]["frac_with_any_cycle"], rows[-1]["mean_cycle_rank"],
                    100 * rows[-1]["frac_truncated_at_bound"], rows[-1]["seconds"])
    return rows


def value_distribution() -> dict:
    """The BTC value distribution the value filtration is built on.

    Measured over the transactions that actually carry an address mapping, since those are the ones
    whose ``total_BTC`` becomes an edge weight in ``G_addr``. The shortfall against the full
    transaction set is reported too, because it is the figure the stability argument quotes.
    """
    actors = RAW_PP / "Actors Dataset"
    txs = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv",
                      usecols=["txId", "total_BTC"])
    addr_tx = pd.read_csv(actors / "AddrTx_edgelist.csv", usecols=["txId"])
    tx_addr = pd.read_csv(actors / "TxAddr_edgelist.csv", usecols=["txId"])
    mapped = np.intersect1d(addr_tx["txId"].unique(), tx_addr["txId"].unique())
    used = txs[txs["txId"].isin(mapped)]

    values = pd.to_numeric(used["total_BTC"], errors="coerce").dropna()
    quantiles = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "available": True,
        "column": "total_BTC",
        "n": int(values.size),
        "n_all_transactions": int(len(txs)),
        "n_without_address_mapping": int(len(txs) - len(used)),
        "frac_without_address_mapping": float((len(txs) - len(used)) / len(txs)),
        "min": float(quantiles.loc[0.0]),
        "q25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "q75": float(quantiles.loc[0.75]),
        "max": float(quantiles.loc[1.0]),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "orders_of_magnitude": float(np.log10(quantiles.loc[1.0] / quantiles.loc[0.0]))
        if quantiles.loc[0.0] > 0 else None,
    }


def address_appearance_histogram() -> dict:
    """How many time steps each address appears in, which decides whether trajectories exist."""
    frame = pd.read_csv(RAW_PP / "Actors Dataset" / "wallets_features.csv",
                        usecols=["address", "Time step"])
    per_address = frame.groupby("address")["Time step"].nunique()
    counts = per_address.value_counts().sort_index()
    total = int(per_address.size)
    return {
        "n_addresses": total,
        "n_feature_rows": int(len(frame)),
        "steps_to_addresses": {str(int(k)): int(v) for k, v in counts.items()},
        "frac_single_step": float(counts.get(1, 0) / total),
        "frac_multi_step": float(1.0 - counts.get(1, 0) / total),
        "n_multi_step": int(total - counts.get(1, 0)),
        # Cumulative tails, because the prose quotes "five or more" and an arithmetic expression
        # over the histogram would have to name every one of its keys.
        "n_at_least_3": int(counts[counts.index >= 3].sum()),
        "n_at_least_5": int(counts[counts.index >= 5].sum()),
        "n_at_least_10": int(counts[counts.index >= 10].sum()),
        "max_steps_observed": int(counts.index.max()),
    }


def main(sections: tuple[str, ...], seed: int, radii: tuple[int, ...] | None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "graph_structure.json"
    payload: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    if "global" in sections or "sparsity" in sections:
        stores = {"G_tx": build_tx_graph(RAW_PP), "G_addr": build_address_graph(RAW_PP)}
        if "global" in sections:
            payload["global"] = {name: global_census(store) for name, store in stores.items()}
            for name, census in payload["global"].items():
                logger.info("%s: %s", name, census["cycle_rank_identity"])
        if "sparsity" in sections:
            # Rows accumulate rather than replace, so the profile can be filled in over several
            # runs: the cheap radii first, and the deep address-graph neighbourhoods -- where a
            # 22,996-degree hub makes a single subgraph expensive -- when the machine is free.
            rows: list[dict] = [row for row in payload.get("sparsity", [])]
            for name, store in stores.items():
                wanted = tuple(r for r in RADII[name] if radii is None or r in radii)
                if not wanted:
                    continue
                fresh = sparsity_profile(store, store.node_ids, wanted, seed)
                done = {(row["view"], row["radius"]) for row in fresh}
                rows = [row for row in rows if (row["view"], row["radius"]) not in done] + fresh
            payload["sparsity"] = sorted(rows, key=lambda row: (row["view"], row["radius"]))
    if "values" in sections:
        payload["value_distribution"] = value_distribution()
    if "appearances" in sections:
        payload["address_appearances"] = address_appearance_histogram()

    payload["seed"] = seed
    payload["node_bound"] = NODE_BOUND
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+",
                        default=["global", "sparsity", "values", "appearances"],
                        choices=["global", "sparsity", "values", "appearances"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radii", type=int, nargs="+", default=None,
                        help="restrict the sparsity profile to these radii; omit for all")
    args = parser.parse_args()
    raise SystemExit(main(tuple(args.sections), args.seed,
                          tuple(args.radii) if args.radii else None))
