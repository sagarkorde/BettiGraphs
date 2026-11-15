"""R4. Graph neural network baseline, with and without the topological signature.

Answers the reviewer objection "only tabular learners were tested": does a message-passing
model, which can learn neighbourhood structure itself, gain anything from T?

Model: two-layer GraphSAGE (mean aggregator) written in plain PyTorch on a sparse adjacency,
so torch-geometric is not required. Full-batch on G_tx; every time step is its own connected
component, so no message crosses from the test window into training. Loss is masked to the
labelled training steps; early stopping on validation PR-AUC; test scored once per seed.

Node inputs: P, P+T, P+S, T (standardised on training rows). Unlabelled nodes take part in
message passing, so S and T are extracted for EVERY node of G_tx (cached after the first run;
about 200k ego-subgraphs at radius 2).

Usage:
    pip install torch --index-url https://download.pytorch.org/whl/cpu    # if not installed
    .venv/Scripts/python.exe experiments/revision/gnn_baseline.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import (CACHE, canonical_config, extract_features, logger, setup_logging,
                    write_json, OUT)

FEATURE_SETS = ("P", "T", "P+T", "P+S")


def build_inputs(n_jobs: int):
    from experiments.baseline.run_transaction_experiment import load_transaction_table
    from common import RAW_PP, build_tx_graph
    frame = load_transaction_table()
    store = build_tx_graph(RAW_PP)
    # align frame rows to graph node order
    order = pd.Series(np.arange(len(frame)), index=frame["txId"].to_numpy())
    frame = frame.iloc[order.reindex(store.node_ids).to_numpy()].reset_index(drop=True)
    m = extract_features(store, store.node_ids, canonical_config(), n_jobs=n_jobs,
                         cache_dir=CACHE)
    local = [c for c in frame.columns if c.startswith(("Local_feature_", "Aggregate_feature_"))]
    P = frame[["Time step"] + local].to_numpy(np.float32)
    return frame, store, {"P": P, "T": m.topological.astype(np.float32),
                          "S": m.classical.astype(np.float32)}


def run(features: np.ndarray, store, labels, steps, seed: int, epochs: int, hidden: int,
        device: str) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = (steps <= 34) & (labels != -1)
    va = (steps >= 35) & (steps <= 41) & (labels != -1)
    te = (steps >= 42) & (labels != -1)

    X = np.nan_to_num(features)
    mu, sd = X[steps <= 34].mean(0), X[steps <= 34].std(0)
    sd[sd == 0] = 1
    X = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)

    n = store.n_nodes
    rows = np.repeat(np.arange(n), np.diff(store.indptr))
    deg = np.maximum(np.diff(store.indptr), 1).astype(np.float32)
    A = torch.sparse_coo_tensor(
        torch.tensor(np.vstack([rows, store.indices]), dtype=torch.long),
        torch.tensor(1.0 / deg[rows], dtype=torch.float32), (n, n), device=device).coalesce()

    class SAGE(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.s1, self.n1 = nn.Linear(d_in, hidden), nn.Linear(d_in, hidden)
            self.s2, self.n2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, 1)
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            h = F.relu(self.s1(x) + self.n1(torch.sparse.mm(A, x)))
            h = self.drop(h)
            h = F.relu(self.s2(h) + self.n2(torch.sparse.mm(A, h)))
            return self.out(self.drop(h)).squeeze(-1)

    model = SAGE(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    y = torch.tensor(np.clip(labels, 0, 1), dtype=torch.float32, device=device)
    tr_t = torch.tensor(tr, device=device)
    pos_w = torch.tensor((y[tr_t] == 0).sum().item() / max((y[tr_t] == 1).sum().item(), 1),
                         device=device)
    best, best_state, patience = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X)[tr_t], y[tr_t], pos_weight=pos_w)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(X)).cpu().numpy()
        v = average_precision_score(labels[va], s[va])
        if v > best:
            best, patience = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 30:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(X)).cpu().numpy()
    return {"val_pr_auc": float(best), "pr_auc": float(average_precision_score(labels[te], s[te])),
            "epochs": epoch + 1}


def main(quick: bool, n_seeds: int, epochs: int, hidden: int, n_jobs: int) -> None:
    import torch
    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame, store, base = build_inputs(n_jobs)
    labels = frame["label"].to_numpy().astype(int)
    steps = frame["Time step"].to_numpy()
    rows = []
    for name in (("P", "P+T") if quick else FEATURE_SETS):
        X = np.hstack([base[p] for p in name.split("+")])
        for seed in range(2 if quick else n_seeds):
            r = run(X, store, labels, steps, seed, 20 if quick else epochs, hidden, device)
            rows.append({"features": name, "seed": seed, **r})
            logger.info("GraphSAGE %s seed %d: %s", name, seed, r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gnn_baseline.csv", index=False)
    means = df.groupby("features")["pr_auc"].agg(["mean", "std"]).round(4)
    piv = df.pivot(index="seed", columns="features", values="pr_auc")
    diff = (piv["P+T"] - piv["P"]) if {"P", "P+T"} <= set(piv) else pd.Series(dtype=float)
    write_json("gnn_baseline.json", {
        "model": "GraphSAGE-mean, 2 layers", "hidden": hidden, "device": device,
        "test_pr_auc": means.to_dict("index"),
        "P+T_minus_P": {"mean": float(diff.mean()) if diff.size else None,
                        "per_seed": diff.round(4).tolist()}})
    print(means.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    main(a.quick, a.n_seeds, a.epochs, a.hidden, a.n_jobs)
