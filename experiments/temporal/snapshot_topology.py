"""Gate 8, RQ4a / H4: snapshot-level topological dynamics.

The Elliptic transaction graph has zero cross-time-step edges, so it is a disjoint union of 49
snapshot graphs. That makes a *per-snapshot* topological summary well defined and cheap: compute
β₀, β₁ and persistence statistics of each whole snapshot under a value filtration, giving a series
of length 49, and ask whether it tracks the illicit base rate.

**This is a descriptive analysis and is reported as one.** n = 49 supports a correlation and a
figure; it does not support a classification claim, and the pre-registration says so. Two further
cautions are built into the test rather than left to the discussion:

* The 49 snapshots are a time series, so ordinary permutation of the labels would overstate
  significance by ignoring autocorrelation. A **block permutation** null is used instead.
* Snapshot size varies by a factor of seven (1,089 to 7,880 nodes), and both β₀ and β₁ scale with
  size. Raw counts would therefore correlate with anything that correlates with size. Size-normalised
  quantities are the primary ones reported, with raw counts shown alongside.

Usage::

    .venv\\Scripts\\python.exe experiments\\temporal\\snapshot_topology.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.statistics import partial_spearman, spearman_with_ci  # noqa: E402
from src.features.topological import persistence_entropy, persistence_statistics  # noqa: E402
from src.topology.filtrations import edge_filtration  # noqa: E402
from src.topology.persistence import flag_persistence  # noqa: E402
from src.utils.seeds import set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
OUT = REPO_ROOT / "results" / "processed"
TESTS = REPO_ROOT / "results" / "statistical_tests"

logger = logging.getLogger("snapshot_topology")
CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}


def block_permutation_p(
    x: np.ndarray, y: np.ndarray, observed: float, *, block: int = 5,
    n_permutations: int = 10_000, seed: int = 0
) -> float:
    """Two-sided p-value for a Spearman correlation under a block-permutation null.

    Contiguous blocks of length ``block`` are shuffled as units, which preserves short-range
    autocorrelation in the series. Permuting individual time steps would destroy it and make the
    null far too easy to reject.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    n = x.size
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    extreme = 0
    for _ in range(n_permutations):
        order = rng.permutation(len(blocks))
        permuted = np.concatenate([blocks[i] for i in order])[:n]
        candidate = stats.spearmanr(x, y[permuted]).statistic
        if abs(candidate) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (n_permutations + 1))


def main(filtration: str, complex_dim: int, n_permutations: int) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    set_all_seeds(0)
    OUT.mkdir(parents=True, exist_ok=True)
    TESTS.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv",
                           usecols=["txId", "Time step", "total_BTC"])
    classes = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_classes.csv")
    edges = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_edgelist.csv")
    frame = features.merge(classes, on="txId", how="left")
    frame["label"] = frame["class"].map(CLASS_TO_LABEL).fillna(-1).astype(np.int8)

    step_of = frame.set_index("txId")["Time step"]
    btc_of = frame.set_index("txId")["total_BTC"].fillna(0.0)
    src_col, dst_col = edges.columns[0], edges.columns[1]
    edges["step"] = edges[src_col].map(step_of)

    rows: list[dict] = []
    for step in sorted(frame["Time step"].unique()):
        nodes = frame.loc[frame["Time step"] == step, "txId"].to_numpy()
        step_edges = edges.loc[edges["step"] == step]

        graph = nx.Graph()
        graph.add_nodes_from(nodes.tolist())
        for u, v in zip(step_edges[src_col].to_numpy(), step_edges[dst_col].to_numpy()):
            graph.add_edge(int(u), int(v), weight=float(btc_of.get(u, 0.0)), timestamp=float(step))

        started = time.perf_counter()
        values = edge_filtration(graph, filtration, normalise="rank")
        result = flag_persistence(graph, values, max_dim=1, complex_dim=complex_dim)
        elapsed = time.perf_counter() - started

        labels = frame.loc[frame["Time step"] == step, "label"].to_numpy()
        n_illicit = int((labels == 1).sum())
        n_licit = int((labels == 0).sum())
        order = graph.number_of_nodes()
        size = graph.number_of_edges()
        components = nx.number_connected_components(graph)

        stats0 = persistence_statistics(result, 0)
        stats1 = persistence_statistics(result, 1)
        row = {
            "time_step": int(step),
            "n_nodes": order, "n_edges": size,
            "n_illicit": n_illicit, "n_licit": n_licit,
            "base_rate_illicit": n_illicit / max(n_illicit + n_licit, 1),
            "beta0": components,
            "cycle_rank": size - order + components,
            # Size-normalised primaries: beta scales with order, so raw counts would correlate
            # with anything that correlates with snapshot size.
            "beta0_per_node": components / order,
            "cycle_rank_per_node": (size - order + components) / order,
            "mean_degree": 2.0 * size / order,
            "entropy_h0": persistence_entropy(result, 0),
            "entropy_h1": persistence_entropy(result, 1),
            "h0_life_mean": stats0["h0_life_mean"],
            "h1_life_mean": stats1["h1_life_mean"],
            "h1_n_finite": stats1["h1_n_finite"],
            "h1_n_finite_per_node": stats1["h1_n_finite"] / order,
            "n_simplices": result.n_simplices,
            "seconds": elapsed,
        }
        rows.append(row)
        logger.info(
            "step %2d: n=%5d e=%5d beta0=%4d cycles=%5d entropy_h1=%.4f base_rate=%.4f (%.2fs)",
            step, order, size, components, row["cycle_rank"], row["entropy_h1"],
            row["base_rate_illicit"], elapsed,
        )

    series = pd.DataFrame(rows)
    series.to_csv(OUT / "snapshot_topology.csv", index=False)

    # --- H4: do topological summaries track the illicit base rate? ---
    descriptors = ["beta0_per_node", "cycle_rank_per_node", "entropy_h0", "entropy_h1",
                   "h1_life_mean", "h1_n_finite_per_node", "mean_degree"]
    target = series["base_rate_illicit"].to_numpy()
    # Elementary counts that several "topological" descriptors are algebraically determined by.
    controls = series[["n_nodes", "mean_degree"]].to_numpy()

    tests: dict[str, dict] = {}
    for name in descriptors:
        values = series[name].to_numpy()
        result = spearman_with_ci(values, target, n_resamples=5000)
        result["block_permutation_p"] = block_permutation_p(
            values, target, result["rho"], n_permutations=n_permutations
        )
        result["rho_with_n_nodes"] = spearman_with_ci(
            values, series["n_nodes"].to_numpy(), n_resamples=1000
        )["rho"]
        result["rho_with_mean_degree"] = spearman_with_ci(
            values, series["mean_degree"].to_numpy(), n_resamples=1000
        )["rho"]

        # The decisive test. A descriptor that loses its association once snapshot order and mean
        # degree are removed was reporting size or density, not topology.
        if name == "mean_degree":
            result["partial"] = {"rho": float("nan"), "ci_low": float("nan"),
                                 "ci_high": float("nan"), "n": int(values.size),
                                 "note": "is itself a control"}
            survives = False
        else:
            partial = partial_spearman(values, target, controls, n_resamples=2000)
            result["partial"] = partial
            survives = bool(
                np.isfinite(partial["ci_low"])
                and not (partial["ci_low"] <= 0 <= partial["ci_high"])
            )

        result["raw_association"] = bool(
            np.isfinite(result["ci_low"]) and not (result["ci_low"] <= 0 <= result["ci_high"])
            and result["block_permutation_p"] < 0.05
        )
        # H4 requires the association to survive the elementary-count controls.
        result["supports_h4"] = bool(result["raw_association"] and survives)
        tests[name] = result

        partial_text = (f"partial rho={result['partial']['rho']:+.3f} "
                        f"[{result['partial']['ci_low']:+.3f},{result['partial']['ci_high']:+.3f}]"
                        if np.isfinite(result["partial"]["rho"]) else "partial n/a")
        logger.info(
            "%-24s rho=%+.3f [%+.3f,%+.3f] perm_p=%.4f | %s | %s",
            name, result["rho"], result["ci_low"], result["ci_high"],
            result["block_permutation_p"], partial_text,
            "SUPPORTS H4" if result["supports_h4"] else
            ("confounded" if result["raw_association"] else "no association"),
        )

    size_check = {name: tests[name]["rho_with_n_nodes"] for name in descriptors}

    # Record the algebraic identities explicitly, so the confound is documented as a fact about
    # the descriptors rather than as an empirical surprise.
    connected = int((series["beta0"] == 1).sum())
    identities = {
        "snapshots_with_beta0_equal_1": connected,
        "of_total": int(len(series)),
        "note": (
            "For a connected graph beta_0 per node is exactly 1/|V|, a strictly decreasing "
            "function of order, and cycle rank per node is |E|/|V| - 1, i.e. mean degree halved "
            "and shifted. Those two descriptors therefore carry no information beyond |V| and |E|, "
            "and any association they show with the illicit base rate is a size or density effect "
            "wearing a topological name. Only lifetime- and entropy-based descriptors, which "
            "depend on the filtration rather than on the counts, can carry anything further."
        ),
    }

    payload = {
        "filtration": filtration,
        "complex_dim": complex_dim,
        "n_time_steps": int(len(series)),
        "note": (
            "Descriptive analysis. n = 49 supports a correlation and a figure, not a "
            "classification claim. A block-permutation null is used because the 49 snapshots form "
            "a time series and step-wise permutation would ignore autocorrelation. Size-normalised "
            "descriptors are primary because beta scales with snapshot order, which varies "
            "sevenfold across the series."
        ),
        "h4_tests": tests,
        "descriptor_correlation_with_snapshot_size": size_check,
        "algebraic_identities": identities,
        "total_seconds": float(series["seconds"].sum()),
    }
    (TESTS / "h4_snapshot_topology.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n=== H4: snapshot topology vs illicit base rate (n=49) ===")
    for name, result in tests.items():
        print(f"  {name:24s} rho={result['rho']:+.3f} "
              f"CI[{result['ci_low']:+.3f},{result['ci_high']:+.3f}] "
              f"perm_p={result['block_permutation_p']:.4f} "
              f"{'SUPPORTS' if result['supports_h4'] else '-'}")
    print("\n=== confounder check: descriptor vs snapshot size ===")
    for name, rho in size_check.items():
        print(f"  {name:24s} rho_with_size={rho:+.3f}")
    print(f"\ntotal persistent-homology time for all 49 snapshots: "
          f"{series['seconds'].sum():.1f} s")
    logger.info("wrote %s", OUT / "snapshot_topology.csv")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--complex-dim", type=int, default=2, dest="complex_dim")
    parser.add_argument("--n-permutations", type=int, default=10_000, dest="n_permutations")
    raise SystemExit(main(**vars(parser.parse_args())))
