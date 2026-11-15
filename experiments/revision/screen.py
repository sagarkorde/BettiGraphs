"""R9. The label-free pre-check, as a standalone tool, and its prospective test.

Computes, on a sample of target nodes of ANY graph, before any label is read:
  1. cyclic share   -- fraction of radius-r ego-subgraphs with at least one cycle;
  2. median cycle rank (|E| - |V| + c) of those ego-subgraphs;
  3. bridge fraction -- share of ego-subgraph edges that are bridges. This is what drives the
     geodesic-Rips instability: deleting a bridge disconnects vertex pairs, which jump to the
     sentinel distance, whereas an edge-valued filtration loses only the deleted edge;
  4. edge-deletion drift -- mean relative L2 change of the Betti signature when 10% of the
     ego-subgraph's edges are deleted at random.

and turns them into a stated prediction using thresholds fixed in THIS file (below), so the
prediction can be frozen and hashed before the labelled experiment is run.

Prospective protocol (what makes the screen a validated predictor instead of a post hoc story):
  a) pick a second labelled graph not used in this study (e.g. Elliptic2, an Ethereum phishing
     transaction graph, or a later Bitcoin period);
  b) run ``screen.py --edges <file> --predict`` -- it writes results/revision/screen_<name>.json
     with the prediction and prints its SHA-256; register that hash (OSF) BEFORE step c;
  c) run the topological vs classical comparison on that graph and compare with the prediction.

Usage:
    .venv/Scripts/python.exe experiments/revision/screen.py --view G_tx
    .venv/Scripts/python.exe experiments/revision/screen.py --view G_addr
    .venv/Scripts/python.exe experiments/revision/screen.py --edges path/to/edges.csv --name eth --predict
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd

from common import RAW_PP, setup_logging, OUT

# Thresholds fixed before any prospective use. Changing them after seeing a new graph's labels
# would make the test circular; change them only with a dated note.
THRESHOLDS = {"min_cyclic_share": 0.5, "min_median_cycle_rank": 2, "max_drift_10pct": 0.5}


def store_from_edges(path: str, name: str):
    from src.graph_construction.graphs import _build_store
    e = pd.read_csv(path)
    src, dst = e.columns[:2]
    nodes = pd.unique(np.concatenate([e[src].to_numpy(), e[dst].to_numpy()]))
    idx = pd.Series(np.arange(len(nodes)), index=nodes)
    w = e[e.columns[2]].to_numpy(np.float32) if e.shape[1] > 2 else np.ones(len(e), np.float32)
    t = e[e.columns[3]].to_numpy(np.float32) if e.shape[1] > 3 else np.ones(len(e), np.float32)
    return _build_store(idx[e[src]].to_numpy(), idx[e[dst]].to_numpy(), w, t, nodes, name)


def signature(g):
    from src.features.topological import betti_signature
    from src.topology.persistence import persistence_of
    return betti_signature(persistence_of(g, "value_superlevel", max_dim=1, complex_dim=2),
                           max_dim=1)[0]


def measure(store, n_sample: int, radius: int, seed: int, max_nodes: int) -> dict:
    rng = np.random.default_rng(seed)
    targets = store.node_ids[rng.choice(store.n_nodes, size=min(n_sample, store.n_nodes),
                                        replace=False)]
    ranks, bridges, drifts = [], [], []
    for node in targets:
        g = store.ego_subgraph(node, radius, max_nodes=max_nodes)
        m = g.number_of_edges()
        ranks.append(m - g.number_of_nodes() + nx.number_connected_components(g))
        if m == 0:
            continue
        bridges.append(sum(1 for _ in nx.bridges(g)) / m)
        if m >= 10:
            base = signature(g)
            h = g.copy()
            drop = rng.choice(m, size=max(1, m // 10), replace=False)
            edges = list(h.edges())
            h.remove_edges_from([edges[i] for i in drop])
            denom = np.linalg.norm(base)
            if denom > 0:
                drifts.append(float(np.linalg.norm(signature(h) - base) / denom))
    ranks = np.array(ranks)
    return {"n_sampled": int(len(targets)), "radius": radius,
            "cyclic_share": float((ranks > 0).mean()),
            "median_cycle_rank": float(np.median(ranks)),
            "mean_bridge_fraction": float(np.mean(bridges)) if bridges else None,
            "drift_10pct_value_filtration": float(np.mean(drifts)) if drifts else None,
            "n_drift_measured": len(drifts)}


def predict(m: dict) -> dict:
    checks = {
        "has_cycles": m["cyclic_share"] >= THRESHOLDS["min_cyclic_share"],
        "enough_cycles": m["median_cycle_rank"] >= THRESHOLDS["min_median_cycle_rank"],
        "stable": (m["drift_10pct_value_filtration"] is not None and
                   m["drift_10pct_value_filtration"] <= THRESHOLDS["max_drift_10pct"]),
    }
    verdict = ("topology may carry signal; worth testing against a tuned baseline"
               if all(checks.values()) else
               "topology unlikely to add over conventional features; do not build the pipeline")
    return {"checks": checks, "thresholds": THRESHOLDS, "prediction": verdict}


def main(view, edges, name, n_sample, radius, seed, max_nodes, do_predict) -> None:
    setup_logging()
    if edges:
        store = store_from_edges(edges, name or "custom")
    else:
        from src.graph_construction.graphs import build_address_graph, build_tx_graph
        store = (build_tx_graph if view == "G_tx" else build_address_graph)(RAW_PP)
        name = name or view
    m = measure(store, n_sample, radius, seed, max_nodes)
    out = {"graph": name, "measured_at": datetime.now(timezone.utc).isoformat(),
           "labels_read": False, **m}
    if do_predict:
        out.update(predict(m))
    text = json.dumps(out, indent=2)
    path = OUT / f"screen_{name}.json"
    path.write_text(text)
    print(text)
    print("SHA-256 of", path.name, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=["G_tx", "G_addr"], default="G_tx")
    ap.add_argument("--edges", help="CSV: src,dst[,value[,time]]")
    ap.add_argument("--name")
    ap.add_argument("--n-sample", type=int, default=2000)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=25_000)
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    main(a.view, a.edges, a.name, a.n_sample, a.radius, a.seed, a.max_nodes, a.predict)
