"""R7. Hyp3 wallet retrieval with the two caveats removed.

The 12.1x retrieval lift on the address view carries two caveats in the manuscript:
(1) graph neighbours of a query were not excluded from its pool, so part of the lift may be
    neighbourhood overlap (wallets that share transactions look alike trivially);
(2) neighbourhoods included edges from later time steps, so the result is retrospective.

This script repeats the illicit-query retrieval of ``experiments/hybrid/run_wallet_similarity.py``
(same pool construction and seed, same standardisation on training-window wallets) under:
* ``exclude_hops`` in {0, 1, 2}: pool wallets within that many hops of the query in G_addr are
  removed from the query's ranking (1 = shares a transaction; 2 = ego-subgraphs share the query);
* ``time_cut``: each wallet's ego-subgraph is extracted from G_addr restricted to transactions up
  to that wallet's first time step, so no later edge is used.

Usage:
    .venv/Scripts/python.exe experiments/revision/hyp3_controls.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (CACHE, RAW_PP, canonical_config, extract_features, logger, setup_logging,
                    write_json, OUT)


def features_for(pool, train_sample, time_cut: bool, n_jobs: int):
    from common import build_address_graph
    from tx_on_address_view import address_store_upto
    blocks = {}
    for name, frame in (("pool", pool), ("train", train_sample)):
        if not time_cut:
            store = build_address_graph(RAW_PP)
            m = extract_features(store, frame["address"].to_numpy(), canonical_config("G_addr"),
                                 n_jobs=n_jobs, cache_dir=CACHE)
            blocks[name] = (m.topological, m.classical, list(m.topological_names))
            continue
        T = np.zeros((len(frame), 0)); S = None; names = None
        parts = {}
        for step, idx in frame.groupby("Time step").indices.items():
            store = address_store_upto(int(step))
            ids = frame["address"].to_numpy()[idx]
            present = np.isin(ids, store.node_ids)
            m = extract_features(store, ids[present], canonical_config(store.name),
                                 n_jobs=n_jobs, cache_dir=CACHE)
            parts[step] = (idx[present], m)
            names = list(m.topological_names)
        dt = next(iter(parts.values()))[1]
        T = np.zeros((len(frame), dt.topological.shape[1])); S = np.zeros((len(frame), dt.classical.shape[1]))
        for idx, m in parts.values():
            T[idx], S[idx] = m.topological, m.classical
        blocks[name] = (T, S, names)
    return blocks


def neighbour_mask(store, pool_ids, query_rows, hops: int) -> np.ndarray:
    """(q, n) boolean: pool item lies within ``hops`` of the query in G_addr."""
    mask = np.zeros((len(query_rows), len(pool_ids)), dtype=bool)
    if hops == 0:
        return mask
    pos = {a: i for i, a in enumerate(pool_ids)}
    for qi, row in enumerate(query_rows):
        try:
            seed = store.index_of(pool_ids[row])
        except KeyError:
            continue
        near = store.node_ids[store.k_hop(seed, hops, max_nodes=200_000)]
        for a in near:
            j = pos.get(a)
            if j is not None and j != row:
                mask[qi, j] = True
    return mask


def main(n_pool: int, seed: int, hops_list, n_jobs: int, quick: bool) -> None:
    from experiments.hybrid.run_wallet_similarity import load_wallets, stratified_sample
    from src.evaluation import splits as sp
    from src.similarity.topological import evaluate_retrieval, pairwise_euclidean
    from common import build_address_graph

    setup_logging()
    wallets = load_wallets()
    labelled = wallets[wallets.label != -1].reset_index(drop=True)
    split = sp.temporal_split(labelled["Time step"].to_numpy(), labelled["label"].to_numpy())
    train_frame = labelled[split.train].reset_index(drop=True)
    eval_frame = labelled[split.test].reset_index(drop=True)
    pool = stratified_sample(eval_frame, min(n_pool, len(eval_frame)), seed=seed)
    train_sample = stratified_sample(train_frame, min(5000, len(train_frame)), seed=seed + 2)
    labels = pool["label"].to_numpy()
    queries = np.flatnonzero(labels == 1)
    full_store = build_address_graph(RAW_PP)

    provided = [c for c in pool.columns if c not in ("address", "class", "label")
                and pd.api.types.is_numeric_dtype(pool[c])]
    results = []
    for time_cut in ((False,) if quick else (False, True)):
        blocks = features_for(pool, train_sample, time_cut, n_jobs)
        Tp, Sp, names = blocks["pool"]; Tt, St, _ = blocks["train"]
        betti = [i for i, n in enumerate(names) if n.startswith("bc_")]

        def std(a, b):
            c, s = np.nanmean(b, 0), np.nanstd(b, 0); s[s == 0] = 1
            return np.nan_to_num((a - c) / s)
        spaces = {"T_betti": std(Tp[:, betti], Tt[:, betti]), "T_full": std(Tp, Tt),
                  "S": std(Sp, St),
                  "W": std(pool[provided].to_numpy(float), train_sample[provided].to_numpy(float))}
        rng = np.random.default_rng(seed)
        for hops in hops_list:
            nb = neighbour_mask(full_store, pool["address"].to_numpy(), queries, hops)
            excluded = float(nb.sum(1).mean())
            for name, block in list(spaces.items()) + [("shuffled", None)]:
                if block is None:
                    d = pairwise_euclidean(spaces["T_full"], spaces["T_full"][queries])
                    d = d[:, rng.permutation(d.shape[1])]
                else:
                    d = pairwise_euclidean(block, block[queries])
                d[nb] = np.inf
                r = evaluate_retrieval(d, labels[queries], labels, space=name,
                                       exclude_self=queries)
                results.append({"time_cut": time_cut, "exclude_hops": hops, "space": name,
                                "mean_excluded_per_query": excluded,
                                "p_at_10": r.precision_at[10], "p_at_50": r.precision_at[50],
                                "lift_at_10": r.precision_at[10] / r.base_rate,
                                "map": r.map_score, "n_queries": r.n_queries})
                logger.info("%s", results[-1])
    df = pd.DataFrame(results)
    df.to_csv(OUT / "hyp3_controls.csv", index=False)
    write_json("hyp3_controls.json", {"n_pool": int(len(pool)), "n_illicit_queries": int(queries.size),
                                      "rows": df.round(4).to_dict("records")})
    print(df.pivot_table(index=["time_cut", "exclude_hops"], columns="space",
                         values="lift_at_10").round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pool", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hops", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.n_pool, a.seed, a.hops, a.n_jobs, a.quick)
