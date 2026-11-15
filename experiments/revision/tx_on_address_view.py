"""R6. Transaction classification with descriptors computed on the ADDRESS view.

The paper's main confound: the address view was evaluated on wallet retrieval, the
transaction view on transaction classification, so "denser view, more signal" is mixed with
"different task". Limitation 2 names this as the experiment that could reverse the
transaction-level conclusion. Here the task is held fixed (classify transactions, frozen
temporal split, same learners and budget) and only the view changes:

* T_tx   -- the canonical signature of the transaction's own G_tx neighbourhood;
* T_addr -- signatures of the radius-2 G_addr neighbourhoods of the transaction's input and
            output addresses, aggregated per transaction (mean and max over addresses, plus the
            address count), and likewise S_addr.

Leakage control: an address neighbourhood can contain edges from later time steps. With
``--time-cut`` (default on) each address subgraph is extracted from a graph restricted to edges
up to the END OF THE TRANSACTION'S OWN STEP, so nothing later than the prediction point is used.
This is done by building one filtered G_addr per time step, so the cost is 49 graph builds.

Cost: classical statistics on address neighbourhoods cost about 0.74 s each at radius 2
(Table S1), so S_addr is off by default; T_addr alone is about 0.1 s per address.

Usage:
    .venv/Scripts/python.exe experiments/revision/tx_on_address_view.py
    .venv/Scripts/python.exe experiments/revision/tx_on_address_view.py --with-classical
    .venv/Scripts/python.exe experiments/revision/tx_on_address_view.py --no-time-cut   # faster, retrospective
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (CACHE, RAW_PP, canonical_config, extract_features, load_tx, logger,
                    search_and_eval, setup_logging, write_json, OUT)


def address_store_upto(step: int | None):
    """G_addr built only from transactions up to ``step`` (inclusive); None = all."""
    from src.graph_construction.graphs import _build_store
    actors = RAW_PP / "Actors Dataset"
    if not hasattr(address_store_upto, "_joined"):
        addr_tx = pd.read_csv(actors / "AddrTx_edgelist.csv")
        tx_addr = pd.read_csv(actors / "TxAddr_edgelist.csv")
        txs = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv",
                          usecols=["txId", "Time step", "total_BTC"])
        address_store_upto._joined = (addr_tx.merge(txs, on="txId")
                                      .merge(tx_addr, on="txId"))
        address_store_upto._addr_tx, address_store_upto._tx_addr = addr_tx, tx_addr
    j = address_store_upto._joined
    if step is not None:
        j = j[j["Time step"] <= step]
    addresses = pd.unique(np.concatenate([j["input_address"].to_numpy(),
                                          j["output_address"].to_numpy()]))
    index = pd.Series(np.arange(len(addresses), dtype=np.int64), index=addresses)
    first = (pd.concat([pd.DataFrame({"a": j["input_address"], "t": j["Time step"]}),
                        pd.DataFrame({"a": j["output_address"], "t": j["Time step"]})])
             .groupby("a")["t"].min())
    name = "G_addr" if step is None else f"G_addr_le{step}"
    return _build_store(index[j["input_address"]].to_numpy(), index[j["output_address"]].to_numpy(),
                        j["total_BTC"].to_numpy(np.float32), j["Time step"].to_numpy(np.float32),
                        addresses, name,
                        node_timestep=first.reindex(addresses).fillna(-1).to_numpy(np.int32))


def tx_addresses(tx_ids: np.ndarray) -> pd.DataFrame:
    address_store_upto(None) if not hasattr(address_store_upto, "_addr_tx") else None
    a = address_store_upto._addr_tx.rename(columns={"input_address": "address"})
    b = address_store_upto._tx_addr.rename(columns={"output_address": "address"})
    m = pd.concat([a[["txId", "address"]], b[["txId", "address"]]]).drop_duplicates()
    return m[m.txId.isin(set(tx_ids.tolist()))]


def aggregate(tx_ids, links: pd.DataFrame, ids: np.ndarray, block: np.ndarray, prefix: str):
    df = pd.DataFrame(np.nan_to_num(block), index=pd.Index(ids, name="address"))
    joined = links.join(df, on="address", how="inner")
    g = joined.drop(columns="address").groupby("txId")
    out = pd.concat([g.mean().add_prefix(f"{prefix}_mean_"), g.max().add_prefix(f"{prefix}_max_"),
                     g.size().rename(f"{prefix}_n_addr")], axis=1)
    return out.reindex(tx_ids).fillna(0.0).to_numpy(np.float32)


def main(quick: bool, time_cut: bool, model: str, budget: int, n_seeds: int, n_jobs: int,
         with_classical: bool, max_nodes: int):
    from src.evaluation import splits as sp
    setup_logging()
    data, _ = load_tx(n_jobs)
    tx_ids = data.frame["txId"].to_numpy()
    links = tx_addresses(tx_ids)
    logger.info("%d tx-address links; %.1f%% of labelled transactions have an address",
                len(links), 100 * np.isin(tx_ids, links.txId.unique()).mean())
    step_of = dict(zip(tx_ids, data.steps))
    links = links.assign(step=links.txId.map(step_of))

    T_parts, S_parts = [], []
    groups = [(None, links)] if not time_cut else list(links.groupby("step"))
    if quick:  # smoke test: one untruncated graph
        groups = [(None, links)]
    for step, part in groups:
        store = address_store_upto(None if step is None else int(step))
        addrs = pd.unique(part.address)
        addrs = addrs[np.isin(addrs, store.node_ids)]
        cfg = canonical_config(view=store.name, include_classical=with_classical,
                               max_nodes=max_nodes)
        m = extract_features(store, addrs, cfg, n_jobs=n_jobs, cache_dir=CACHE)
        sub_tx = part.txId.unique()
        T_parts.append(pd.DataFrame(aggregate(sub_tx, part, addrs, m.topological, "Ta"), index=sub_tx))
        if with_classical:
            S_parts.append(pd.DataFrame(aggregate(sub_tx, part, addrs, m.classical, "Sa"),
                                        index=sub_tx))
        logger.info("step %s: %d addresses featurised", step, len(addrs))
    T_addr = pd.concat(T_parts).reindex(tx_ids).fillna(0).to_numpy(np.float32)
    fams = {"T_tx": data.families["T"], "T_addr": T_addr, "S_tx": data.families["S"],
            "P": data.families["P"], "P+T_addr": np.hstack([data.families["P"], T_addr]),
            "P+T_tx": data.families["P+T"]}
    if with_classical:
        S_addr = pd.concat(S_parts).reindex(tx_ids).fillna(0).to_numpy(np.float32)
        fams["S_addr"] = S_addr
        fams["S_addr+T_addr"] = np.hstack([S_addr, T_addr])
    y = data.labels
    split = sp.temporal_split(data.steps, y)
    rows = []
    for name, X in fams.items():
        for r in search_and_eval(model, X, y, split.train, split.val, split.test,
                                 range(2 if quick else n_seeds), 3 if quick else budget):
            rows.append({"family": name, **r})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tx_on_address_view.csv", index=False)
    means = df.groupby("family")["pr_auc"].agg(["mean", "std"]).round(4)
    write_json("tx_on_address_view.json", {"time_cut": time_cut, "model": model, "with_classical": with_classical,
                                           "max_nodes": max_nodes,
                                           "test_pr_auc": means.to_dict("index")})
    print(means.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-time-cut", dest="time_cut", action="store_false")
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--with-classical", action="store_true",
                    help="also compute S on address neighbourhoods (about 8x slower)")
    ap.add_argument("--max-nodes", type=int, default=25_000)
    a = ap.parse_args()
    main(a.quick, a.time_cut, a.model, a.budget, a.n_seeds, a.n_jobs, a.with_classical,
         a.max_nodes)
