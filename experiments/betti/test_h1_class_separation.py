"""Gate 8, RQ1 / H1: do topological descriptors separate illicit from licit entities?

H1 as pre-registered is rejected unless some descriptor achieves Holm-corrected *p* < 0.05 **and**
|Cliff's δ| ≥ 0.15.

A raw test is not enough, and the H4 result is why. There, four descriptors cleared the raw bar
and every one of them turned out to be an elementary graph count wearing a topological name. The
same trap is present here in a different form: illicit ego-subgraphs are **smaller** than licit ones
(median order 44 against 71 at radius 4), and β₁ grows with order, so a raw β₁ difference may be a
size difference restated.

Every descriptor is therefore tested twice:

1. **Raw** -- Mann-Whitney U with Cliff's δ, across all training-window entities.
2. **Size-matched** -- the same test within strata of ego-subgraph order and edge count, with the
   per-stratum effect sizes pooled by stratum weight. A descriptor that keeps its effect here is
   saying something the counts do not.

Only the size-matched result is allowed to support H1. The raw result is reported alongside so the
size of the confound is visible rather than merely asserted.

Everything runs on the **training window only**. The test split is untouched.

Usage::

    .venv\\Scripts\\python.exe experiments\\betti\\test_h1_class_separation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import splits as sp  # noqa: E402
from src.evaluation.statistics import cliffs_delta, holm_bonferroni, mann_whitney_with_effect  # noqa: E402
from src.features.pipeline import ExtractionConfig, extract_features  # noqa: E402
from src.graph_construction.graphs import build_tx_graph  # noqa: E402
from src.utils.seeds import set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
CACHE = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "results" / "statistical_tests"
TABLES = REPO_ROOT / "tables"

logger = logging.getLogger("h1")
CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}

#: Pre-registered thresholds (PREREGISTRATION.md §2, H1).
ALPHA = 0.05
MIN_EFFECT = 0.15


def size_matched_effect(
    values: np.ndarray,
    labels: np.ndarray,
    n_nodes: np.ndarray,
    n_edges: np.ndarray,
    *,
    n_bins: int = 6,
    min_per_cell: int = 30,
) -> dict[str, float]:
    """Cliff's δ pooled across strata of ego-subgraph order and edge count.

    Strata are quantile bins, so they adapt to the heavily right-skewed size distribution instead
    of being swamped by the tail. Cells with too few of either class are dropped and their loss is
    reported, because pooling a δ estimated from four points would add noise, not information.

    Returns the weighted mean δ, the weighted-variance-based standard error, the number of usable
    strata and the share of entities they cover.
    """
    def quantile_bin(array: np.ndarray) -> np.ndarray:
        edges = np.unique(np.quantile(array, np.linspace(0, 1, n_bins + 1)))
        return np.clip(np.digitize(array, edges[1:-1]), 0, max(len(edges) - 2, 0))

    node_bin = quantile_bin(n_nodes)
    edge_bin = quantile_bin(n_edges)
    strata = node_bin * (n_bins + 1) + edge_bin

    deltas: list[float] = []
    weights: list[float] = []
    covered = 0
    for stratum in np.unique(strata):
        mask = strata == stratum
        a = values[mask & (labels == 1)]
        b = values[mask & (labels == 0)]
        if a.size < min_per_cell or b.size < min_per_cell:
            continue
        deltas.append(cliffs_delta(a, b))
        weights.append(float(a.size + b.size))
        covered += int(a.size + b.size)

    if not deltas:
        return {"delta": float("nan"), "se": float("nan"), "n_strata": 0, "coverage": 0.0}

    deltas_array = np.array(deltas)
    weights_array = np.array(weights)
    weights_array /= weights_array.sum()
    pooled = float(np.sum(weights_array * deltas_array))
    variance = float(np.sum(weights_array * (deltas_array - pooled) ** 2))
    return {
        "delta": pooled,
        "se": float(np.sqrt(variance / max(len(deltas) - 1, 1))),
        "n_strata": len(deltas),
        "coverage": covered / len(values),
        "per_stratum_min": float(deltas_array.min()),
        "per_stratum_max": float(deltas_array.max()),
        "sign_consistent": bool(np.all(deltas_array > 0) or np.all(deltas_array < 0)),
    }


def main(radius: int, filtration: str, n_jobs: int, max_descriptors: int) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_all_seeds(0)
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv",
                           usecols=["txId", "Time step"])
    classes = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_classes.csv")
    frame = features.merge(classes, on="txId", how="inner")
    frame["label"] = frame["class"].map(CLASS_TO_LABEL).astype(np.int8)
    labelled = frame[frame.label != -1].reset_index(drop=True)

    store = build_tx_graph(RAW_PP)
    config = ExtractionConfig(view="G_tx", radius=radius, filtration=filtration,
                              complex_dim=2, max_dim=1, include_classical=True)
    matrices = extract_features(store, labelled["txId"].to_numpy(), config,
                               n_jobs=n_jobs, cache_dir=CACHE)

    split = sp.temporal_split(labelled["Time step"].to_numpy(), labelled["label"].to_numpy())
    train = split.train
    labels = labelled["label"].to_numpy()[train]
    n_nodes = matrices.n_nodes[train].astype(float)
    n_edges = matrices.n_edges[train].astype(float)
    logger.info("H1 on the training window: %d entities (%d illicit), median order %.0f",
                train.sum(), int((labels == 1).sum()), float(np.median(n_nodes)))

    # Topological descriptors, plus the classical ones for comparison in the same table.
    blocks = [("T", matrices.topological, matrices.topological_names)]
    if matrices.classical is not None:
        blocks.append(("S", matrices.classical, matrices.classical_names))

    records: list[dict] = []
    for family, block, names in blocks:
        subset = block[train]
        # Constant columns carry no information and would only inflate the correction family.
        variance = subset.std(axis=0)
        keep = np.flatnonzero(variance > 0)
        if max_descriptors and len(keep) > max_descriptors:
            # Keep the highest-variance descriptors; declared, and applied identically to both
            # families so neither is advantaged.
            keep = keep[np.argsort(-variance[keep])][:max_descriptors]
        logger.info("%s: testing %d of %d descriptors (%d were constant)",
                    family, len(keep), block.shape[1], int((variance == 0).sum()))

        for column in keep:
            values = subset[:, column].astype(float)
            p_value, delta = mann_whitney_with_effect(values[labels == 1], values[labels == 0])
            matched = size_matched_effect(values, labels, n_nodes, n_edges)
            records.append({
                "family": family,
                "descriptor": names[column],
                "p_raw": p_value,
                "delta_raw": delta,
                "delta_size_matched": matched["delta"],
                "delta_size_matched_se": matched["se"],
                "n_strata": matched["n_strata"],
                "stratum_coverage": matched["coverage"],
                "sign_consistent_across_strata": matched.get("sign_consistent", False),
                "mean_illicit": float(np.nanmean(values[labels == 1])),
                "mean_licit": float(np.nanmean(values[labels == 0])),
            })

    results = pd.DataFrame(records)
    finite = results["p_raw"].notna()
    rejected = np.zeros(len(results), dtype=bool)
    adjusted = np.full(len(results), np.nan)
    rejected[finite.to_numpy()], adjusted[finite.to_numpy()] = holm_bonferroni(
        results.loc[finite, "p_raw"].to_numpy(), alpha=ALPHA
    )
    results["p_holm"] = adjusted
    results["passes_raw"] = rejected & (results["delta_raw"].abs() >= MIN_EFFECT)
    # Only the size-matched effect may support H1, and its interval must exclude the threshold.
    results["passes_size_matched"] = (
        rejected
        & (results["delta_size_matched"].abs() >= MIN_EFFECT)
        & (results["delta_size_matched"].abs() - 1.96 * results["delta_size_matched_se"] > 0)
        & results["sign_consistent_across_strata"]
    )
    results = results.sort_values("delta_size_matched", key=lambda s: s.abs(), ascending=False)
    results.to_csv(OUT / "h1_descriptor_separation.csv", index=False)

    topological = results[results.family == "T"]
    supported = bool(topological["passes_size_matched"].any())
    n_raw = int(topological["passes_raw"].sum())
    n_matched = int(topological["passes_size_matched"].sum())

    payload = {
        "radius": radius, "filtration": filtration,
        "n_train_entities": int(train.sum()),
        "n_illicit": int((labels == 1).sum()),
        "median_order_illicit": float(np.median(n_nodes[labels == 1])),
        "median_order_licit": float(np.median(n_nodes[labels == 0])),
        "criterion": f"Holm-corrected p < {ALPHA} and |Cliff's delta| >= {MIN_EFFECT}",
        "n_topological_passing_raw": n_raw,
        "n_topological_passing_size_matched": n_matched,
        "h1_supported": supported,
        "note": (
            "Only the size-matched result may support H1. Illicit ego-subgraphs are smaller than "
            "licit ones, and beta_1 grows with order, so a raw difference can be a size difference "
            "restated -- which is exactly how the snapshot-level H4 result failed."
        ),
        "top_descriptors": topological.head(15)[
            ["descriptor", "p_holm", "delta_raw", "delta_size_matched",
             "delta_size_matched_se", "n_strata", "passes_size_matched"]
        ].to_dict(orient="records"),
    }
    (OUT / "h1_class_separation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\n=== H1: descriptor separation at radius {radius} ({filtration}) ===")
    print(f"training entities: {int(train.sum()):,} ({int((labels == 1).sum()):,} illicit)")
    print(f"median ego order: illicit {np.median(n_nodes[labels == 1]):.0f}, "
          f"licit {np.median(n_nodes[labels == 0]):.0f}  <- the confound")
    print(f"\ntopological descriptors passing raw criterion       : {n_raw}")
    print(f"topological descriptors passing size-matched criterion: {n_matched}")
    print(f"\nH1 supported: {supported}\n")
    print(topological.head(12)[
        ["descriptor", "p_holm", "delta_raw", "delta_size_matched", "n_strata",
         "passes_size_matched"]
    ].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    if "S" in results.family.values:
        classical = results[results.family == "S"]
        print(f"\nfor comparison, classical descriptors passing size-matched: "
              f"{int(classical['passes_size_matched'].sum())} of {len(classical)}")
        print(classical.head(6)[
            ["descriptor", "delta_raw", "delta_size_matched", "passes_size_matched"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    logger.info("wrote %s", OUT / "h1_class_separation.json")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    parser.add_argument("--max-descriptors", type=int, default=0, dest="max_descriptors",
                        help="0 tests every non-constant descriptor")
    raise SystemExit(main(**vars(parser.parse_args())))
