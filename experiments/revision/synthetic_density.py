"""R5. Controlled synthetic test of the density account.

The density account says cycle-based descriptors can only separate classes when the
neighbourhoods contain cycles. On Elliptic the two views also differ in task, so the account is
confounded. Here the task is held fixed and cycle density is the only thing that moves.

Generator (see ``make_entity``). Each entity is a value-weighted random tree, like a UTXO
transaction neighbourhood, plus Binomial(n, p_cycle) reconvergence edges; p_cycle is the
density knob. The class signal lives ONLY in where those cycles close (among high-value or
low-value paths). Node count, edge count, edge-value distribution and triangle count are the
same in distribution for both classes, so neither a size-only baseline nor value histograms can
separate them, and at p_cycle = 0 nothing can.

Expected if the density account is right: every family is at the base rate when p_cycle = 0,
and T's PR-AUC, and its margin over S, rise with the share of cyclic neighbourhoods. A result
where T does NOT rise would count against the account.

Usage:
    .venv/Scripts/python.exe experiments/revision/synthetic_density.py
"""
from __future__ import annotations

import argparse

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import logger, setup_logging, write_json, OUT

P_CYCLE = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)


def make_entity(rng, n: int, p_cycle: float, positive: bool, n_candidates: int = 10) -> nx.Graph:
    """A random value-weighted tree with Binomial(n, p_cycle) reconvergence edges.

    Both classes get the same NUMBER of extra edges, drawn from the same value distribution, and
    every extra edge joins two nodes 3-6 hops apart on the tree (so no triangles are created).
    The classes differ only in WHERE the cycle closes: among ``n_candidates`` admissible pairs,
    a positive closes the one whose tree path has the largest minimum value (a cycle of
    high-value edges, which a superlevel filtration sees early and keeps long), a negative the
    one with the smallest. With no extra edges the two classes are identically distributed.
    """
    tree = nx.Graph(nx.random_labeled_tree(n, seed=int(rng.integers(1 << 31))) if hasattr(
        nx, "random_labeled_tree") else nx.random_tree(n, seed=int(rng.integers(1 << 31))))
    for u, v in tree.edges():
        tree[u][v]["weight"] = float(rng.lognormal(0, 1))
        tree[u][v]["timestamp"] = 1.0
    g = tree.copy()
    nodes = np.array(tree.nodes())
    for _ in range(rng.binomial(n, p_cycle)):
        best, best_key = None, None
        for _ in range(n_candidates * 5):
            u, v = rng.choice(nodes, 2, replace=False)
            if g.has_edge(u, v):
                continue
            path = nx.shortest_path(tree, u, v)
            if not 4 <= len(path) <= 7:
                continue
            key = min(tree[a][b]["weight"] for a, b in zip(path, path[1:]))
            key = key if positive else -key
            if best_key is None or key > best_key:
                best, best_key = (int(u), int(v)), key
        if best is not None:
            g.add_edge(*best, weight=float(rng.lognormal(0, 1)), timestamp=1.0)
    g.graph["seed"] = 0
    for node in g.nodes():
        g.nodes[node]["in_degree"] = g.degree(node)
        g.nodes[node]["out_degree"] = g.degree(node)
    return g


def featurise(g):
    from src.features.classical import classical_features
    from src.features.topological import betti_signature
    from src.topology.persistence import persistence_of
    res = persistence_of(g, "value_superlevel", max_dim=1, complex_dim=2)
    t, _ = betti_signature(res, max_dim=1)
    s, _ = classical_features(g)
    size = np.array([g.number_of_nodes(), g.number_of_edges()], dtype=float)
    cyc = nx.number_of_edges(g) - g.number_of_nodes() + nx.number_connected_components(g)
    return t, s, size, cyc


def score(X, y, seed):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    p = cross_val_predict(clf, np.nan_to_num(X), y, cv=StratifiedKFold(5, shuffle=True,
                          random_state=seed), method="predict_proba")[:, 1]
    return float(average_precision_score(y, p))


def main(n_entities: int, n_nodes: int, base_rate: float, n_reps: int) -> None:
    setup_logging()
    rows = []
    for p in P_CYCLE:
        for rep in range(n_reps):
            rng = np.random.default_rng(10_000 * rep + int(p * 1000))
            y = (rng.random(n_entities) < base_rate).astype(int)
            feats = [featurise(make_entity(rng, n_nodes, p, bool(lab))) for lab in y]
            T = np.array([f[0] for f in feats]); S = np.array([f[1] for f in feats])
            Z = np.array([f[2] for f in feats]); C = np.array([f[3] for f in feats])
            r = {"p_cycle": p, "rep": rep, "base_rate": float(y.mean()),
                 "cyclic_share": float(np.mean(C > 0)),
                 "median_cycle_rank": float(np.median(C)),
                 "T": score(T, y, rep), "S": score(S, y, rep), "size_only": score(Z, y, rep),
                 "S+T": score(np.hstack([S, T]), y, rep)}
            rows.append(r)
            logger.info("%s", r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_density.csv", index=False)
    summary = df.groupby("p_cycle")[["cyclic_share", "median_cycle_rank", "size_only", "S", "T", "S+T"]].mean()
    summary["T_minus_S"] = summary["T"] - summary["S"]
    rho = summary["median_cycle_rank"].rank().corr(summary["T_minus_S"].rank())
    write_json("synthetic_density.json", {
        "n_entities": n_entities, "n_nodes": n_nodes, "reps": n_reps,
        "by_p_cycle": summary.round(4).to_dict("index"),
        "spearman_median_cycle_rank_vs_T_minus_S": float(rho)})
    print(summary.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-entities", type=int, default=2000)
    ap.add_argument("--n-nodes", type=int, default=40)
    ap.add_argument("--base-rate", type=float, default=0.1)
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_entities, a.n_reps = 300, 1
    main(a.n_entities, a.n_nodes, a.base_rate, a.n_reps)
