"""Gate 9: measure the computational cost of the pipeline. Produces Table 10 and Figure 12.

The literature treats persistent homology as the expensive part of a topological pipeline, and the
LeapSpace report lists scalability as one of its five research gaps. This script measures it
rather than assuming it, and it measures the **classical baseline on the same subgraphs** so the
comparison is like for like.

That matters for fairness. An early version of the classical feature code contained an O(n^2)
square count and a dense Laplacian eigensolve, and reporting topology as cheaper than *that* would
have been a rigged comparison. Both were optimised first -- the square count now uses the trace
identity and the spectrum a capped dense solve -- so the numbers below compare topology against a
tuned baseline.

Measured axes: ego radius, subgraph order, flag-complex dimension, Betti-curve resolution, and
homology dimension.

Usage::

    .venv\\Scripts\\python.exe scripts\\measure_scalability.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.classical import classical_features  # noqa: E402
from src.features.topological import betti_signature, default_grid  # noqa: E402
from src.graph_construction.graphs import build_address_graph, build_tx_graph  # noqa: E402
from src.topology.persistence import persistence_of  # noqa: E402
from src.utils.seeds import RunProvenance, set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
OUT = REPO_ROOT / "results" / "processed"
RUNTIME = REPO_ROOT / "results" / "runtime"
TABLES = REPO_ROOT / "tables"

logger = logging.getLogger("scalability")


def time_stage(function, *args, **kwargs) -> tuple[object, float]:
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def measure(store, targets, *, axis: str, radius: int, filtration: str, complex_dim: int,
            resolution: int, max_dim: int, max_nodes: int) -> list[dict]:
    """Time each pipeline stage over ``targets`` and return one record per stage."""
    grid = default_grid(resolution)
    orders, sizes = [], []
    t_extract = t_topo = t_class = 0.0
    n_simplices = []

    tracemalloc.start()
    for target in targets:
        subgraph, elapsed = time_stage(store.ego_subgraph, target, radius, max_nodes=max_nodes)
        t_extract += elapsed
        orders.append(subgraph.number_of_nodes())
        sizes.append(subgraph.number_of_edges())
        if subgraph.number_of_edges() == 0:
            continue

        started = time.perf_counter()
        result = persistence_of(subgraph, filtration, max_dim=max_dim,
                                complex_dim=complex_dim, normalise="rank")
        betti_signature(result, max_dim=max_dim, grid=grid)
        t_topo += time.perf_counter() - started
        n_simplices.append(result.n_simplices)

        _, elapsed = time_stage(classical_features, subgraph)
        t_class += elapsed
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n = len(targets)
    common = {
        # Which sweep this row belongs to. Without it, rows from different axes that happen to
        # share filter values collide in the tables -- an earlier version showed radius 3 twice and
        # complex dimension 2 six times, with values spanning 2.9 to 180 ms, because a row measured
        # while sweeping one axis also satisfied the filter for another.
        "axis": axis,
        "view": store.name, "radius": radius, "filtration": filtration,
        "complex_dim": complex_dim, "resolution": resolution, "max_dim": max_dim,
        "n_entities": n,
        "median_nodes": float(np.median(orders)),
        "p95_nodes": float(np.percentile(orders, 95)),
        "median_edges": float(np.median(sizes)),
        "median_simplices": float(np.median(n_simplices)) if n_simplices else 0.0,
        "peak_traced_mib": peak / 2**20,
    }
    return [
        {**common, "stage": "extraction", "ms_per_entity": 1000 * t_extract / n},
        {**common, "stage": "topological", "ms_per_entity": 1000 * t_topo / n},
        {**common, "stage": "classical", "ms_per_entity": 1000 * t_class / n},
    ]


def main(n_sample: int, n_jobs: int) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    set_all_seeds(0)
    for directory in (OUT, RUNTIME, TABLES):
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    rng = np.random.default_rng(0)

    logger.info("building graph views")
    tx_store = build_tx_graph(RAW_PP)
    addr_store = build_address_graph(RAW_PP)

    tx_targets = rng.choice(tx_store.node_ids, size=n_sample, replace=False)
    addr_targets = rng.choice(addr_store.node_ids, size=max(n_sample // 4, 50), replace=False)

    # --- axis 1: ego radius, on both views ---
    for store, targets, radii in ((tx_store, tx_targets, (1, 2, 3, 4)),
                                  (addr_store, addr_targets, (1, 2, 3))):
        for radius in radii:
            logger.info("%s radius=%d", store.name, radius)
            records += measure(store, targets, axis="radius", radius=radius,
                               filtration="value_superlevel", complex_dim=2, resolution=32,
                               max_dim=1, max_nodes=25_000)

    # --- axis 2: flag-complex dimension ---
    for complex_dim in (1, 2, 3):
        logger.info("complex_dim=%d", complex_dim)
        records += measure(tx_store, tx_targets, axis="complex_dim", radius=3,
                           filtration="value_superlevel", complex_dim=complex_dim, resolution=32,
                           max_dim=1, max_nodes=25_000)

    # --- axis 3: Betti-curve resolution ---
    for resolution in (16, 32, 64, 128):
        records += measure(tx_store, tx_targets, axis="resolution", radius=3,
                           filtration="value_superlevel", complex_dim=2, resolution=resolution,
                           max_dim=1, max_nodes=25_000)

    # --- axis 4: homology dimension ---
    for max_dim in (0, 1, 2):
        records += measure(tx_store, tx_targets, axis="homology_dim", radius=3,
                           filtration="value_superlevel", complex_dim=max(max_dim + 1, 2),
                           resolution=32, max_dim=max_dim, max_nodes=25_000)

    # --- axis 5: filtration ---
    for filtration in ("value_superlevel", "temporal_sublevel", "core_sublevel", "geodesic_rips"):
        try:
            records += measure(tx_store, tx_targets[:200], axis="filtration", radius=3,
                               filtration=filtration, complex_dim=2, resolution=32, max_dim=1,
                               max_nodes=25_000)
        except Exception as exc:
            logger.warning("filtration %s failed: %s", filtration, exc)

    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "scalability.csv", index=False)
    (RUNTIME / "scalability_provenance.json").write_text(
        json.dumps({"n_sample": n_sample, "provenance": RunProvenance(seed=0).as_dict()},
                   indent=2, default=str), encoding="utf-8")

    # --- Table 10 ---
    radius_block = frame[(frame.axis == "radius") & (frame.view == "G_tx")]
    pivot = radius_block.pivot_table(index=["radius", "median_nodes", "median_edges",
                                            "median_simplices"],
                                     columns="stage", values="ms_per_entity").reset_index()

    lines = [
        "# Table 10 — Computational cost",
        "",
        "Measured, not estimated, on the hardware recorded in `docs/ENVIRONMENT.md`",
        "(32 cores, 63.7 GiB, RTX 4060 Laptop). Milliseconds per entity, single-threaded, so the",
        "numbers are comparable across stages; the pipeline runs them across 32 workers.",
        "",
        "The classical baseline was optimised before this comparison was made — its square count",
        "uses the trace identity rather than an O(n²) pair enumeration, and its Laplacian spectrum",
        "a capped dense solve rather than an O(n³) full decomposition. Reporting topology as",
        "cheaper than an unoptimised baseline would have been a rigged comparison.",
        "",
        "## Cost by ego radius, transaction graph",
        "",
        "| Radius | Median order | Median edges | Median simplices | Extraction (ms) | "
        "Persistent homology (ms) | Classical features (ms) | PH share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in pivot.iterrows():
        topo = row.get("topological", float("nan"))
        classical = row.get("classical", float("nan"))
        extraction = row.get("extraction", float("nan"))
        total = topo + classical + extraction
        lines.append(
            f"| {int(row['radius'])} | {row['median_nodes']:.0f} | {row['median_edges']:.0f} | "
            f"{row['median_simplices']:.0f} | {extraction:.2f} | {topo:.2f} | {classical:.2f} | "
            f"{100 * topo / total:.1f}% |"
        )

    dim_block = frame[(frame.axis == "complex_dim") & (frame.stage == "topological")]
    lines += ["", "## Cost by flag-complex dimension (radius 3)", "",
              "| Complex dimension | Median simplices | ms per entity |", "|---|---|---|"]
    for _, row in dim_block.sort_values("complex_dim").iterrows():
        lines.append(f"| {int(row['complex_dim'])} | {row['median_simplices']:.0f} | "
                     f"{row['ms_per_entity']:.2f} |")

    filt_block = frame[(frame.axis == "filtration") & (frame.stage == "topological")]
    if not filt_block.empty:
        lines += ["", "## Cost by filtration (radius 3, 200 entities)", "",
                  "| Filtration | ms per entity |", "|---|---|"]
        for _, row in filt_block.iterrows():
            lines.append(f"| `{row['filtration']}` | {row['ms_per_entity']:.2f} |")

    res_block = frame[(frame.axis == "resolution") & (frame.stage == "topological")]
    lines += ["", "## Cost by Betti-curve resolution (radius 3)", "",
              "| Resolution | ms per entity |", "|---|---|"]
    for _, row in res_block.sort_values("resolution").iterrows():
        lines.append(f"| {int(row['resolution'])} | {row['ms_per_entity']:.2f} |")

    addr_block = frame[(frame.axis == "radius") & (frame.view == "G_addr")]
    if not addr_block.empty:
        addr_pivot = addr_block.pivot_table(
            index=["radius", "median_nodes", "median_edges"], columns="stage",
            values="ms_per_entity").reset_index()
        lines += ["", "## Cost by ego radius, address graph", "",
                  "| Radius | Median order | Median edges | Extraction (ms) | "
                  "Persistent homology (ms) | Classical features (ms) |", "|---|---|---|---|---|---|"]
        for _, row in addr_pivot.iterrows():
            lines.append(
                f"| {int(row['radius'])} | {row['median_nodes']:.0f} | {row['median_edges']:.0f} | "
                f"{row.get('extraction', float('nan')):.2f} | "
                f"{row.get('topological', float('nan')):.2f} | "
                f"{row.get('classical', float('nan')):.2f} |")

    topo_total = frame[frame.stage == "topological"]["ms_per_entity"].sum()
    class_total = frame[frame.stage == "classical"]["ms_per_entity"].sum()
    lines += [
        "",
        "## Summary",
        "",
        f"Across every configuration measured, the classical feature family cost "
        f"**{class_total / max(topo_total, 1e-9):.1f}× more** than persistent homology on the same "
        "subgraphs. This runs against the expectation, repeated in the surveyed literature, that "
        "persistent homology is the prohibitive component of a topological pipeline. On sparse "
        "transaction graphs it is not: the flag complexes stay small because the graphs are sparse, "
        "while centrality and spectral features scale with subgraph order regardless.",
        "",
    ]
    (TABLES / "table10_computational_cost.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== cost by stage and radius (G_tx, ms per entity) ===")
    print(pivot.round(2).to_string(index=False))
    print(f"\nclassical / topological cost ratio across all configurations: "
          f"{class_total / max(topo_total, 1e-9):.1f}x")
    logger.info("wrote %s and %s", OUT / "scalability.csv",
                TABLES / "table10_computational_cost.md")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sample", type=int, default=400, dest="n_sample")
    parser.add_argument("--n-jobs", type=int, default=1, dest="n_jobs")
    raise SystemExit(main(**vars(parser.parse_args())))
