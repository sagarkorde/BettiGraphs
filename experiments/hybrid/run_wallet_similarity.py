"""Gate 8, RQ3 / H3: does topological distance between wallets measure behavioural relatedness?

H3 asks whether distance between wallet ego-subgraph persistence diagrams ranks same-class wallets
above distance in **two non-topological control spaces**. Beating one control is not enough; the
pre-registration requires both, because a topological space that beats hand-built graph statistics
but loses to the dataset's own wallet features has not earned a place in a forensic pipeline.

Spaces compared
---------------
=================  ==================================================================
``T_image``        Euclidean distance between persistence images
``T_betti``        Euclidean distance between Betti curves
``T_full``         Euclidean distance between the full standardised signature
``S`` (control 1)  Euclidean distance in the classical graph-structural space
``W`` (control 2)  Euclidean distance in the dataset's own 56 wallet features
``random``         a shuffled distance matrix, as a floor
=================  ==================================================================

Two design points that matter for validity
------------------------------------------
**Nothing is fitted on the evaluation data.** Retrieval by distance involves no training, and the
standardisation applied before distances is fitted on *training-window* wallets only. So the usual
leakage question does not arise, and that is asserted rather than assumed.

**A random-distance floor is included.** Retrieval metrics under class imbalance can look
impressive purely from the base rate. The floor makes that visible instead of leaving the reader to
work out whether P@10 of 0.3 is good.

What this experiment does not claim
-----------------------------------
A small topological distance means two wallets have structurally similar transaction
neighbourhoods. It is **not** evidence of common ownership, and no attribution claim is made. The
labels are third-party behavioural classes, so "relevant" here means "same behavioural class in
that labelling", nothing more.

Usage::

    .venv\\Scripts\\python.exe experiments\\hybrid\\run_wallet_similarity.py
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import splits as sp  # noqa: E402
from src.features.pipeline import ExtractionConfig, extract_features  # noqa: E402
from src.graph_construction.graphs import build_address_graph  # noqa: E402
from src.similarity.topological import (  # noqa: E402
    bottleneck_distance, clustering_agreement, evaluate_retrieval, pairwise_euclidean,
    proxy_fidelity, wasserstein_distance,
)
from src.topology.persistence import persistence_of  # noqa: E402
from src.utils.seeds import set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
CACHE = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "results" / "processed"
TESTS = REPO_ROOT / "results" / "statistical_tests"
TABLES = REPO_ROOT / "tables"

logger = logging.getLogger("wallet_similarity")
WALLET_CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}


def load_wallets() -> pd.DataFrame:
    """Wallet features and labels, one row per (address, time step), reduced to first appearance."""
    features = pd.read_csv(RAW_PP / "Actors Dataset" / "wallets_features.csv")
    classes = pd.read_csv(RAW_PP / "Actors Dataset" / "wallets_classes.csv")
    frame = features.merge(classes, on="address", how="inner")
    frame["label"] = frame["class"].map(WALLET_CLASS_TO_LABEL).fillna(-1).astype(np.int8)
    # One row per address: its earliest observation, so an address cannot appear twice in a pool.
    frame = frame.sort_values("Time step").drop_duplicates("address", keep="first")
    logger.info("wallets: %d unique addresses, %d labelled",
                len(frame), int((frame.label != -1).sum()))
    return frame.reset_index(drop=True)


def stratified_sample(
    frame: pd.DataFrame, n: int, *, seed: int, label_column: str = "label"
) -> pd.DataFrame:
    """Sample ``n`` rows preserving the label ratio, so the base rate is unchanged."""
    rng = np.random.default_rng(seed)
    parts = []
    for label, group in frame.groupby(label_column):
        take = max(1, int(round(n * len(group) / len(frame))))
        take = min(take, len(group))
        parts.append(group.iloc[rng.choice(len(group), size=take, replace=False)])
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def exact_diagram_distances(
    store, addresses: np.ndarray, config: ExtractionConfig, *, dim: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Wasserstein-1 and bottleneck distance matrices for a small address set.

    Each is an optimal-transport problem per pair, so this is only feasible on a subsample. It
    exists to validate the vectorised proxies used at scale.
    """
    results = []
    for address in addresses:
        subgraph = store.ego_subgraph(address, config.radius, max_nodes=config.max_nodes)
        if subgraph.number_of_edges() == 0:
            results.append(None)
            continue
        results.append(persistence_of(
            subgraph, config.filtration, max_dim=config.max_dim,
            complex_dim=config.complex_dim, normalise=config.normalise,
        ))

    n = len(addresses)
    wasserstein = np.zeros((n, n))
    bottleneck = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if results[i] is None or results[j] is None:
                wasserstein[i, j] = wasserstein[j, i] = np.nan
                bottleneck[i, j] = bottleneck[j, i] = np.nan
                continue
            w = wasserstein_distance(results[i], results[j], dim, order=1.0)
            b = bottleneck_distance(results[i], results[j], dim)
            wasserstein[i, j] = wasserstein[j, i] = w
            bottleneck[i, j] = bottleneck[j, i] = b
    return wasserstein, bottleneck


def main(n_queries: int, n_pool: int, radius: int, filtration: str, exact_n: int,
         n_jobs: int, seed: int) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_all_seeds(seed)
    for directory in (OUT, TESTS, TABLES):
        directory.mkdir(parents=True, exist_ok=True)

    wallets = load_wallets()
    labelled = wallets[wallets.label != -1].reset_index(drop=True)

    # Wallet-level temporal split by first observed time step, per PREREGISTRATION.md §3.
    split = sp.temporal_split(labelled["Time step"].to_numpy(), labelled["label"].to_numpy())
    train_frame = labelled[split.train].reset_index(drop=True)
    eval_frame = labelled[split.test].reset_index(drop=True)
    logger.info("evaluation window holds %d labelled wallets (%d illicit)",
                len(eval_frame), int((eval_frame.label == 1).sum()))

    pool = stratified_sample(eval_frame, min(n_pool, len(eval_frame)), seed=seed)
    queries = stratified_sample(pool, min(n_queries, len(pool)), seed=seed + 1)
    # Queries are a subset of the pool, so each query's own entry is masked out at scoring time.
    pool_index = {address: i for i, address in enumerate(pool["address"].to_numpy())}
    query_positions = np.array([pool_index[a] for a in queries["address"].to_numpy()])

    store = build_address_graph(RAW_PP)
    config = ExtractionConfig(
        view="G_addr", radius=radius, filtration=filtration, complex_dim=2, max_dim=1,
        max_nodes=25_000, include_classical=True,
    )

    started = time.perf_counter()
    matrices = extract_features(store, pool["address"].to_numpy(), config,
                               n_jobs=n_jobs, cache_dir=CACHE)
    logger.info("extracted %d wallet signatures in %.1f s; %s",
                matrices.n_entities, time.perf_counter() - started,
                json.dumps(matrices.summary()))

    names = matrices.topological_names
    image_index = [i for i, name in enumerate(names) if name.startswith("pi_")]
    betti_index = [i for i, name in enumerate(names) if name.startswith("bc_")]

    # Standardisation is fitted on TRAINING-window wallets, never on the evaluation pool.
    train_config = config
    train_sample = stratified_sample(train_frame, min(5000, len(train_frame)), seed=seed + 2)
    train_matrices = extract_features(store, train_sample["address"].to_numpy(), train_config,
                                     n_jobs=n_jobs, cache_dir=CACHE)

    def standardise(pool_block: np.ndarray, train_block: np.ndarray) -> np.ndarray:
        centre = np.nanmean(train_block, axis=0)
        spread = np.nanstd(train_block, axis=0)
        spread[spread == 0] = 1.0
        return np.nan_to_num((pool_block - centre) / spread)

    provided_columns = [
        c for c in pool.columns
        if c not in ("address", "class", "label") and pd.api.types.is_numeric_dtype(pool[c])
    ]

    spaces: dict[str, np.ndarray] = {
        "T_image": standardise(matrices.topological[:, image_index],
                               train_matrices.topological[:, image_index]),
        "T_betti": standardise(matrices.topological[:, betti_index],
                               train_matrices.topological[:, betti_index]),
        "T_full": standardise(matrices.topological, train_matrices.topological),
        "S": standardise(matrices.classical, train_matrices.classical),
        "W": standardise(pool[provided_columns].to_numpy(dtype=np.float64),
                         train_sample[provided_columns].to_numpy(dtype=np.float64)),
    }

    pool_labels = pool["label"].to_numpy()
    query_labels = queries["label"].to_numpy()

    results: list[dict] = []
    illicit_results: list[dict] = []
    distance_cache: dict[str, np.ndarray] = {}

    # Every illicit wallet in the pool serves as a query for the illicit-only view, not just the
    # ones that happened to fall in the stratified query sample. At a 4.6% base rate the stratified
    # sample yields under a hundred illicit queries, which is too few to separate the spaces; using
    # the whole illicit population gives an order of magnitude more at no extra cost, since the
    # distances are computed against the same pool either way.
    illicit_pool_rows = np.flatnonzero(pool_labels == 1)
    logger.info("illicit-query view uses all %d illicit wallets in the pool as queries",
                illicit_pool_rows.size)

    for space_name, block in spaces.items():
        distances = pairwise_euclidean(block, block[query_positions])
        distance_cache[space_name] = distances
        retrieval = evaluate_retrieval(
            distances, query_labels, pool_labels, space=space_name,
            exclude_self=query_positions,
        )
        results.append(retrieval.as_dict())
        logger.info("%-8s MAP=%.4f P@10=%.4f P@50=%.4f NDCG@10=%.4f",
                    space_name, retrieval.map_score, retrieval.precision_at[10],
                    retrieval.precision_at[50], retrieval.ndcg_at[10])

        # Illicit-only queries. The pooled figure above is dominated by the majority class: the
        # pool is roughly 95% licit, so a licit query retrieves same-class neighbours almost for
        # free and every space scores near 0.89 whatever it encodes. Restricting queries to illicit
        # wallets asks the question an investigator actually has -- given a known bad wallet, are
        # its topological neighbours also bad? -- against a base rate of about 5%.
        if illicit_pool_rows.size >= 20:
            illicit_distances = pairwise_euclidean(block, block[illicit_pool_rows])
            illicit = evaluate_retrieval(
                illicit_distances, pool_labels[illicit_pool_rows], pool_labels,
                space=space_name, exclude_self=illicit_pool_rows,
            )
            illicit_results.append(illicit.as_dict())
            logger.info("%-8s [illicit queries, n=%d] MAP=%.4f P@10=%.4f lift@10=%.2f",
                        space_name, illicit.n_queries, illicit.map_score,
                        illicit.precision_at[10],
                        illicit.precision_at[10] / max(illicit.base_rate, 1e-9))

    # --- random floor: shuffle one distance matrix's columns ---
    rng = np.random.default_rng(seed)
    shuffled = distance_cache["T_full"][:, rng.permutation(distance_cache["T_full"].shape[1])]
    floor = evaluate_retrieval(shuffled, query_labels, pool_labels, space="random",
                              exclude_self=query_positions)
    results.append(floor.as_dict())
    logger.info("%-8s MAP=%.4f P@10=%.4f (floor)", "random", floor.map_score,
                floor.precision_at[10])
    if illicit_pool_rows.size >= 20:
        illicit_base = pairwise_euclidean(spaces["T_full"], spaces["T_full"][illicit_pool_rows])
        illicit_shuffled = illicit_base[:, rng.permutation(illicit_base.shape[1])]
        illicit_floor = evaluate_retrieval(
            illicit_shuffled, pool_labels[illicit_pool_rows], pool_labels,
            space="random", exclude_self=illicit_pool_rows,
        )
        illicit_results.append(illicit_floor.as_dict())
        logger.info("%-8s [illicit queries] MAP=%.4f P@10=%.4f (floor)", "random",
                    illicit_floor.map_score, illicit_floor.precision_at[10])

    # --- proxy fidelity against exact diagram distances, on a subsample ---
    subsample = stratified_sample(pool, min(exact_n, len(pool)), seed=seed + 3)
    sub_positions = np.array([pool_index[a] for a in subsample["address"].to_numpy()])
    logger.info("computing exact Wasserstein and bottleneck distances on %d wallets",
                len(subsample))
    exact_w, exact_b = exact_diagram_distances(store, subsample["address"].to_numpy(), config)
    upper = np.triu_indices(len(subsample), k=1)
    fidelity = {}
    for space_name in ("T_image", "T_betti", "T_full"):
        proxy = pairwise_euclidean(spaces[space_name][sub_positions])
        fidelity[space_name] = {
            "vs_wasserstein1": proxy_fidelity(exact_w[upper], proxy[upper]),
            "vs_bottleneck": proxy_fidelity(exact_b[upper], proxy[upper]),
        }
        logger.info("%s fidelity: Spearman vs W1 = %.3f, vs bottleneck = %.3f",
                    space_name, fidelity[space_name]["vs_wasserstein1"]["spearman"],
                    fidelity[space_name]["vs_bottleneck"]["spearman"])

    # --- exact-distance retrieval on the subsample, for a like-for-like check ---
    exact_results = []
    sub_labels = subsample["label"].to_numpy()
    for label, matrix in (("exact_wasserstein1", exact_w), ("exact_bottleneck", exact_b)):
        filled = np.where(np.isfinite(matrix), matrix, np.nanmax(matrix[np.isfinite(matrix)]) * 1.1)
        try:
            retrieval = evaluate_retrieval(
                filled, sub_labels, sub_labels, space=label, ks=(10,),
                exclude_self=np.arange(len(subsample)),
            )
            exact_results.append(retrieval.as_dict())
            logger.info("%-20s MAP=%.4f P@10=%.4f (n=%d subsample)",
                        label, retrieval.map_score, retrieval.precision_at[10], len(subsample))
        except ValueError as exc:
            logger.warning("%s retrieval skipped: %s", label, exc)

    # --- clustering agreement on the subsample, where a dense matrix is affordable ---
    clustering: dict[str, dict] = {}
    for space_name in ("T_full", "S", "W"):
        matrix = pairwise_euclidean(spaces[space_name][sub_positions])
        clustering[space_name] = clustering_agreement(matrix, sub_labels, n_clusters=2, seed=seed)
        logger.info("%-8s clustering ARI=%.4f NMI=%.4f silhouette=%.4f", space_name,
                    clustering[space_name]["ari"], clustering[space_name]["nmi"],
                    clustering[space_name]["silhouette"])

    frame_out = pd.DataFrame(results)
    frame_out.to_csv(OUT / "wallet_retrieval.csv", index=False)
    illicit_frame = pd.DataFrame(illicit_results)
    if not illicit_frame.empty:
        illicit_frame.to_csv(OUT / "wallet_retrieval_illicit_queries.csv", index=False)

    topo_best = max((r for r in results if r["space"].startswith("T_")),
                    key=lambda r: r["map"])
    control_s = next(r for r in results if r["space"] == "S")
    control_w = next(r for r in results if r["space"] == "W")
    h3_supported = bool(topo_best["map"] > control_s["map"] and topo_best["map"] > control_w["map"])

    h3_illicit = None
    if illicit_results:
        topo_illicit = max((r for r in illicit_results if r["space"].startswith("T_")),
                           key=lambda r: r["map"])
        s_illicit = next(r for r in illicit_results if r["space"] == "S")
        w_illicit = next(r for r in illicit_results if r["space"] == "W")
        h3_illicit = {
            "best_topological_space": topo_illicit["space"],
            "best_topological_map": topo_illicit["map"],
            "control_S_map": s_illicit["map"],
            "control_W_map": w_illicit["map"],
            "supported": bool(topo_illicit["map"] > s_illicit["map"]
                              and topo_illicit["map"] > w_illicit["map"]),
        }

    payload = {
        "config": {"n_queries": len(queries), "n_pool": len(pool), "radius": radius,
                   "filtration": filtration, "exact_subsample": len(subsample), "seed": seed},
        "pool_label_distribution": {str(k): int(v) for k, v in
                                    pd.Series(pool_labels).value_counts().items()},
        "retrieval": results,
        "retrieval_illicit_queries": illicit_results,
        "illicit_query_note": (
            "The pooled retrieval figures are dominated by the majority class: the pool is about "
            "95% licit, so a licit query retrieves same-class neighbours almost for free and every "
            "space scores near the random floor whatever it encodes. The illicit-query breakdown "
            "asks the question an investigator actually has, against a base rate of about 5%."
        ),
        "exact_distance_retrieval": exact_results,
        "proxy_fidelity": fidelity,
        "clustering": clustering,
        "h3": {
            "best_topological_space": topo_best["space"],
            "best_topological_map": topo_best["map"],
            "control_S_map": control_s["map"],
            "control_W_map": control_w["map"],
            "supported": h3_supported,
            "illicit_query_view": h3_illicit,
            "criterion": "H3 requires the best topological space to beat BOTH controls; beating "
                         "hand-built graph statistics while losing to the dataset's own wallet "
                         "features would not earn a place in a forensic pipeline",
        },
        "caveat": (
            "A small topological distance means two wallets have structurally similar transaction "
            "neighbourhoods. It is not evidence of common ownership and no attribution claim is "
            "made. Labels are third-party behavioural classes, so 'relevant' means same class in "
            "that labelling."
        ),
    }
    (TESTS / "h3_wallet_similarity.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n=== RQ3: wallet retrieval by distance space ===")
    print(frame_out[["space", "map", "precision_at_10", "precision_at_50",
                     "recall_at_50", "ndcg_at_10"]].round(4).to_string(index=False))
    print(f"\nH3 supported: {h3_supported}  "
          f"(best topological {topo_best['space']} MAP={topo_best['map']:.4f} vs "
          f"S={control_s['map']:.4f}, W={control_w['map']:.4f})")
    logger.info("wrote %s", TESTS / "h3_wallet_similarity.json")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-queries", type=int, default=3000, dest="n_queries")
    parser.add_argument("--n-pool", type=int, default=30000, dest="n_pool")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--exact-n", type=int, default=300, dest="exact_n")
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    parser.add_argument("--seed", type=int, default=0)
    raise SystemExit(main(**vars(parser.parse_args())))
