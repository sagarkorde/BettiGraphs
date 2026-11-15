"""Graph views of the Elliptic++ data, and fast ego-subgraph extraction.

Three views are built, matching ``docs/DATA_FINDINGS.md``:

===========  ===================================  =========  ==========
View         Nodes / edges                        Directed   Bipartite
===========  ===================================  =========  ==========
``G_tx``     203,769 transactions / 234,355       yes        no
``G_addr``   822,942 addresses / 2,868,964        yes        no
``G_bip``    addresses + transactions / 1,314,241 yes        **yes**
===========  ===================================  =========  ==========

``G_addr`` is not read from ``AddrAddr_edgelist.csv`` directly, because that file carries no time
and no value. It is reconstructed as ``AddrTx ⋈ TxAddr`` on ``txId``, which was verified to
reproduce 34,481 of 34,481 sampled AddrAddr edges exactly, and which gives every address-address
edge a **time step** and a **real BTC value** inherited from the transaction that created it.
Those two attributes are what make the value and temporal filtrations meaningful rather than
arbitrary.

Why CSR rather than networkx for storage
----------------------------------------
Extracting k-hop neighbourhoods for tens of thousands of target nodes is the hot path of the
whole project. A networkx graph of 823k nodes costs several gigabytes and its traversal is
per-node Python dictionary work. Structure is therefore held in SciPy CSR arrays and traversed
with an explicit frontier loop; a small networkx graph is materialised only for the individual
ego-subgraph about to be handed to GUDHI. This keeps the persistent-homology step -- which is
genuinely expensive -- as the bottleneck, rather than the graph plumbing around it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


@dataclass
class GraphStore:
    """A graph held as CSR arrays with per-edge attributes.

    The stored adjacency is **symmetric** (each original edge appears in both directions) so that
    neighbourhood traversal is direction-agnostic, which is what homology requires. Direction is
    not lost: ``out_degree`` and ``in_degree`` are precomputed from the original directed edges
    and exposed as node attributes for the classical feature family.

    Attributes:
        node_ids: original identifiers, indexed by internal node index.
        indptr, indices: CSR structure of the symmetric adjacency.
        weight: per-adjacency-entry BTC value, aligned with ``indices``.
        timestep: per-adjacency-entry time step, aligned with ``indices``.
        out_degree, in_degree: directed degrees of each node.
        node_timestep: time step of each node where defined, else -1.
        name: view name, for logging and result provenance.
    """

    node_ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weight: np.ndarray
    timestep: np.ndarray
    out_degree: np.ndarray
    in_degree: np.ndarray
    node_timestep: np.ndarray
    name: str
    bipartite_part: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._index = {int(node): i for i, node in enumerate(self.node_ids)} \
            if self.node_ids.dtype.kind in "iu" else \
            {node: i for i, node in enumerate(self.node_ids)}

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_adjacency_entries(self) -> int:
        return len(self.indices)

    def index_of(self, node_id) -> int:
        """Internal index of ``node_id``.

        Raises:
            KeyError: if the node is absent, rather than returning a default that would silently
                produce an empty subgraph.
        """
        try:
            return self._index[node_id]
        except KeyError as exc:
            raise KeyError(f"node {node_id!r} is not in view {self.name!r}") from exc

    def k_hop(self, seed: int, radius: int, *, max_nodes: int | None = None) -> np.ndarray:
        """Internal indices within ``radius`` hops of internal index ``seed``, seed included.

        Truncation policy
        -----------------
        A hub address can reach tens of thousands of nodes at radius 2 (the largest observed
        degree in ``G_addr`` is 22,996), so a size bound is necessary. When the bound bites, the
        surviving frontier is chosen by a **deterministic uniform random subsample** seeded from
        the seed node, not by taking the first entries.

        That distinction matters. Taking a prefix would select neighbours by internal array
        order, which reflects nothing but the order rows appeared in the source CSV -- an
        arbitrary criterion that could correlate with anything. A uniform subsample is unbiased
        with respect to node properties, and seeding it from the seed node keeps it exactly
        reproducible across runs and machines.

        Args:
            max_nodes: bound on the collected node count. Callers learn whether it bit by
                comparing the result size against it, and ``ego_subgraph`` records it in
                ``graph.graph["truncated"]``.
        """
        visited: set[int] = {int(seed)}
        frontier = np.array([seed], dtype=np.int64)
        for _ in range(radius):
            if frontier.size == 0:
                break
            slices = [self.indices[self.indptr[node]: self.indptr[node + 1]] for node in frontier]
            neighbours = np.concatenate(slices) if slices else np.empty(0, dtype=np.int64)
            if neighbours.size == 0:
                break
            fresh = np.unique(neighbours)
            seen = np.fromiter(visited, dtype=np.int64, count=len(visited))
            fresh = fresh[~np.isin(fresh, seen, assume_unique=True)]
            if fresh.size == 0:
                break
            if max_nodes is not None and len(visited) + fresh.size > max_nodes:
                room = max_nodes - len(visited)
                if room <= 0:
                    break
                # Deterministic uniform subsample, reproducible from the seed node.
                rng = np.random.default_rng(int(seed))
                fresh = rng.choice(fresh, size=room, replace=False)
            visited.update(int(x) for x in fresh)
            frontier = fresh
            if max_nodes is not None and len(visited) >= max_nodes:
                break
        return np.fromiter(visited, dtype=np.int64, count=len(visited))

    def ego_subgraph(
        self,
        node_id,
        radius: int,
        *,
        max_nodes: int | None = 2000,
        relabel: bool = True,
    ) -> nx.Graph:
        """Induced undirected ego-subgraph around ``node_id`` as a networkx graph.

        Edge attributes ``weight`` (BTC) and ``timestamp`` (time step) are attached, which is what
        :mod:`src.topology.filtrations` consumes. Node attributes carry the directed degrees so
        the classical feature family can use direction even though homology cannot.
        """
        seed = self.index_of(node_id)
        members = self.k_hop(seed, radius, max_nodes=max_nodes)
        member_set = set(members.tolist())

        graph = nx.Graph()
        mapping = {m: (i if relabel else int(self.node_ids[m])) for i, m in enumerate(members)}
        for internal, label in mapping.items():
            graph.add_node(
                label,
                original_id=self.node_ids[internal],
                out_degree=int(self.out_degree[internal]),
                in_degree=int(self.in_degree[internal]),
                time_step=int(self.node_timestep[internal]),
                is_seed=(internal == seed),
            )
        for internal in members:
            start, end = self.indptr[internal], self.indptr[internal + 1]
            for position in range(start, end):
                other = self.indices[position]
                if other in member_set and other > internal:
                    graph.add_edge(
                        mapping[internal],
                        mapping[int(other)],
                        weight=float(self.weight[position]),
                        timestamp=float(self.timestep[position]),
                    )
        graph.graph["seed"] = mapping[seed]
        graph.graph["view"] = self.name
        graph.graph["radius"] = radius
        graph.graph["truncated"] = bool(max_nodes is not None and members.size >= max_nodes)
        return graph


def _build_store(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    timesteps: np.ndarray,
    node_ids: np.ndarray,
    name: str,
    node_timestep: np.ndarray | None = None,
    bipartite_part: np.ndarray | None = None,
) -> GraphStore:
    """Assemble a :class:`GraphStore` from directed edge arrays."""
    n = len(node_ids)
    out_degree = np.bincount(sources, minlength=n).astype(np.int32)
    in_degree = np.bincount(targets, minlength=n).astype(np.int32)

    # Symmetrise for traversal. Duplicate (u,v) pairs are summed for weight and reduced to the
    # earliest time step, because a pair of addresses can transact many times.
    both_src = np.concatenate([sources, targets])
    both_dst = np.concatenate([targets, sources])
    both_w = np.concatenate([weights, weights])
    both_t = np.concatenate([timesteps, timesteps])

    weight_matrix = sparse.coo_matrix((both_w, (both_src, both_dst)), shape=(n, n)).tocsr()
    weight_matrix.sum_duplicates()
    # Minimum time step per pair: negate, take max via maximum-reduction, negate back.
    time_matrix = sparse.coo_matrix((-both_t, (both_src, both_dst)), shape=(n, n)).tocsr()
    time_matrix = -time_matrix.maximum(
        sparse.coo_matrix((-both_t, (both_src, both_dst)), shape=(n, n)).tocsr()
    )
    time_matrix.sort_indices()
    weight_matrix.sort_indices()
    if not np.array_equal(weight_matrix.indices, time_matrix.indices):
        raise AssertionError("weight and time CSR structures diverged; cannot align attributes")

    return GraphStore(
        node_ids=node_ids,
        indptr=weight_matrix.indptr.astype(np.int64),
        indices=weight_matrix.indices.astype(np.int64),
        weight=weight_matrix.data.astype(np.float32),
        timestep=time_matrix.data.astype(np.float32),
        out_degree=out_degree,
        in_degree=in_degree,
        node_timestep=(node_timestep if node_timestep is not None
                       else np.full(n, -1, dtype=np.int32)),
        name=name,
        bipartite_part=bipartite_part,
    )


def build_tx_graph(raw_pp: Path) -> GraphStore:
    """Build ``G_tx``: transaction to transaction money flow.

    Edge weight is the ``total_BTC`` of the *source* transaction, which is the value actually
    moving out of it. Edge time step is the source transaction's time step; since the graph has
    zero cross-time-step edges, source and target agree anyway.
    """
    txs = pd.read_csv(raw_pp / "Transactions Dataset" / "txs_features.csv",
                      usecols=["txId", "Time step", "total_BTC"])
    edges = pd.read_csv(raw_pp / "Transactions Dataset" / "txs_edgelist.csv")
    src_col, dst_col = edges.columns[0], edges.columns[1]

    node_ids = txs["txId"].to_numpy(dtype=np.int64)
    index = pd.Series(np.arange(len(node_ids), dtype=np.int64), index=node_ids)

    src = edges[src_col].map(index)
    dst = edges[dst_col].map(index)
    keep = src.notna() & dst.notna()
    dropped = int((~keep).sum())
    if dropped:
        logger.warning("dropping %d tx edges with an endpoint absent from the feature table", dropped)
    src = src[keep].to_numpy(dtype=np.int64)
    dst = dst[keep].to_numpy(dtype=np.int64)

    btc = txs.set_index("txId")["total_BTC"].fillna(0.0)
    step = txs.set_index("txId")["Time step"]
    edge_w = btc.reindex(node_ids).to_numpy(dtype=np.float32)[src]
    edge_t = step.reindex(node_ids).to_numpy(dtype=np.float32)[src]

    store = _build_store(
        src, dst, edge_w, edge_t, node_ids, "G_tx",
        node_timestep=step.reindex(node_ids).to_numpy(dtype=np.int32),
    )
    logger.info("G_tx: %d nodes, %d directed edges, %d adjacency entries",
                store.n_nodes, len(src), store.n_adjacency_entries)
    return store


def build_address_graph(raw_pp: Path) -> GraphStore:
    """Build ``G_addr``: address to address, reconstructed with time and value.

    ``AddrAddr_edgelist.csv`` has no time or value columns, so it is rebuilt as
    ``AddrTx ⋈ TxAddr`` on ``txId`` -- verified to reproduce sampled AddrAddr edges exactly --
    which lets each edge inherit the connecting transaction's time step and ``total_BTC``.
    """
    actors = raw_pp / "Actors Dataset"
    addr_tx = pd.read_csv(actors / "AddrTx_edgelist.csv")
    tx_addr = pd.read_csv(actors / "TxAddr_edgelist.csv")
    txs = pd.read_csv(raw_pp / "Transactions Dataset" / "txs_features.csv",
                      usecols=["txId", "Time step", "total_BTC"])

    joined = addr_tx.merge(txs, on="txId", how="inner").merge(tx_addr, on="txId", how="inner")
    logger.info("reconstructed %d address-address edges from the bipartite files", len(joined))

    addresses = pd.unique(
        np.concatenate([joined["input_address"].to_numpy(), joined["output_address"].to_numpy()])
    )
    index = pd.Series(np.arange(len(addresses), dtype=np.int64), index=addresses)
    src = joined["input_address"].map(index).to_numpy(dtype=np.int64)
    dst = joined["output_address"].map(index).to_numpy(dtype=np.int64)

    # Each address's own time step is the earliest transaction it participates in.
    first_step = (
        pd.concat([
            pd.DataFrame({"a": joined["input_address"], "t": joined["Time step"]}),
            pd.DataFrame({"a": joined["output_address"], "t": joined["Time step"]}),
        ])
        .groupby("a")["t"].min()
    )

    store = _build_store(
        src, dst,
        joined["total_BTC"].to_numpy(dtype=np.float32),
        joined["Time step"].to_numpy(dtype=np.float32),
        addresses, "G_addr",
        node_timestep=first_step.reindex(addresses).fillna(-1).to_numpy(dtype=np.int32),
    )
    logger.info("G_addr: %d nodes, %d directed edges, %d adjacency entries",
                store.n_nodes, len(src), store.n_adjacency_entries)
    return store


def build_bipartite_graph(raw_pp: Path) -> GraphStore:
    """Build ``G_bip``: the address-transaction bipartite view.

    This is the most faithful representation of UTXO semantics. It is bipartite, so its flag
    complex is one-dimensional and β_p is identically zero for p ≥ 2 **by construction**;
    ``bipartite_part`` records the side of each node so that fact can be asserted rather than
    rediscovered.
    """
    actors = raw_pp / "Actors Dataset"
    addr_tx = pd.read_csv(actors / "AddrTx_edgelist.csv")
    tx_addr = pd.read_csv(actors / "TxAddr_edgelist.csv")
    txs = pd.read_csv(raw_pp / "Transactions Dataset" / "txs_features.csv",
                      usecols=["txId", "Time step", "total_BTC"])
    tx_info = txs.set_index("txId")

    addresses = pd.unique(
        np.concatenate([addr_tx["input_address"].to_numpy(), tx_addr["output_address"].to_numpy()])
    )
    tx_ids = pd.unique(np.concatenate([addr_tx["txId"].to_numpy(), tx_addr["txId"].to_numpy()]))

    node_ids = np.concatenate([addresses.astype(object), tx_ids.astype(object)])
    part = np.concatenate([np.zeros(len(addresses), np.int8), np.ones(len(tx_ids), np.int8)])
    index = pd.Series(np.arange(len(node_ids), dtype=np.int64), index=node_ids)

    src = np.concatenate([
        addr_tx["input_address"].map(index).to_numpy(dtype=np.int64),
        tx_addr["txId"].map(index).to_numpy(dtype=np.int64),
    ])
    dst = np.concatenate([
        addr_tx["txId"].map(index).to_numpy(dtype=np.int64),
        tx_addr["output_address"].map(index).to_numpy(dtype=np.int64),
    ])
    edge_tx = np.concatenate([addr_tx["txId"].to_numpy(), tx_addr["txId"].to_numpy()])
    weights = tx_info["total_BTC"].reindex(edge_tx).fillna(0.0).to_numpy(dtype=np.float32)
    steps = tx_info["Time step"].reindex(edge_tx).fillna(-1).to_numpy(dtype=np.float32)

    node_step = np.full(len(node_ids), -1, dtype=np.int32)
    node_step[len(addresses):] = tx_info["Time step"].reindex(tx_ids).fillna(-1).to_numpy(np.int32)

    store = _build_store(src, dst, weights, steps, node_ids, "G_bip",
                         node_timestep=node_step, bipartite_part=part)
    logger.info("G_bip: %d nodes (%d addresses + %d transactions), %d directed edges",
                store.n_nodes, len(addresses), len(tx_ids), len(src))
    return store
