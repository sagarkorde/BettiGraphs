"""Generate the manuscript figures from ``results/``.

Every figure is built from a run record. Nothing is hand-drawn from remembered numbers, and a
figure whose input file is absent is **skipped with a message** rather than produced from a
placeholder -- a plausible-looking figure with invented data is far worse than a missing one.

Figures produced, when their inputs exist:

===  =======================================================================
  4  example persistence diagrams, illicit against licit
  5  Betti curve comparison by class
  6  topological profile comparison, illicit against licit
  8  ROC and precision-recall curves by feature family
  9  confusion matrices for the best model per family
 10  feature-family comparison and within-T ablation
 11  snapshot-level topological series over the 49 time steps
 12  scalability: cost against ego radius, subgraph size and homology dimension
===  =======================================================================

Figures 1-3 and 7 are schematics of the framework and are drawn by
``scripts/make_schematics.py``.

Usage::

    .venv\\Scripts\\python.exe scripts\\make_figures.py
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import plotting as pl  # noqa: E402
from src.evaluation import splits as sp  # noqa: E402

RAW = REPO_ROOT / "results" / "raw"
PROCESSED = REPO_ROOT / "results" / "processed"
TESTS = REPO_ROOT / "results" / "statistical_tests"
FIGURES = REPO_ROOT / "figures"
RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
CACHE = REPO_ROOT / "data" / "processed"

logger = logging.getLogger("figures")
CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}


def _latest(pattern: str) -> Path | None:
    matches = sorted(RAW.glob(pattern))
    return matches[-1] if matches else None


def figure_11_snapshot_series() -> None:
    """Snapshot-level topology across the 49 time steps, against the illicit base rate."""
    import matplotlib.pyplot as plt

    path = PROCESSED / "snapshot_topology.csv"
    if not path.exists():
        logger.warning("skipping figure 11: %s is absent", path.name)
        return
    series = pd.read_csv(path)

    figure, axes = plt.subplots(3, 1, figsize=(6.6, 6.4), sharex=True)

    axis = axes[0]
    axis.plot(series.time_step, series.cycle_rank, color=pl.PALETTE["topological"],
              marker="o", label=r"cycle rank $=\beta_1$ of the 1-skeleton")
    axis.set_ylabel("cycle rank")
    twin = axis.twinx()
    twin.plot(series.time_step, series.mean_degree, color=pl.PALETTE["neutral"],
              linestyle="--", label="mean degree")
    twin.set_ylabel("mean degree", color=pl.PALETTE["neutral"])
    twin.grid(False)
    handles = axis.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = axis.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    axis.legend(handles, labels, loc="upper right")
    axis.set_title("Snapshot topology tracks density, not illicit activity")

    axis = axes[1]
    axis.plot(series.time_step, series.h1_life_mean, color=pl.PALETTE["hybrid"], marker="s",
              label=r"mean $H_1$ lifetime")
    axis.plot(series.time_step, series.entropy_h1, color=pl.PALETTE["accent"], marker="^",
              linestyle="-.", label=r"$H_1$ persistence entropy")
    axis.set_ylabel("persistence summary")
    axis.legend(loc="upper right")

    axis = axes[2]
    axis.plot(series.time_step, series.base_rate_illicit, color=pl.PALETTE["illicit"],
              marker="o", label="illicit base rate")
    axis.axvline(sp.REGIME_SHIFT_STEP, color="#444444", linestyle=":", linewidth=1.0)
    axis.annotate(f"regime change\nat step {sp.REGIME_SHIFT_STEP}",
                  xy=(sp.REGIME_SHIFT_STEP, axis.get_ylim()[1] * 0.82),
                  xytext=(sp.REGIME_SHIFT_STEP - 13, axis.get_ylim()[1] * 0.78),
                  fontsize=7.5, color="#444444",
                  arrowprops=dict(arrowstyle="->", color="#444444", lw=0.7))
    axis.set_ylabel("illicit share of labelled")
    axis.set_xlabel("time step")
    axis.legend(loc="upper right")

    tests_path = TESTS / "h4_snapshot_topology.json"
    if tests_path.exists():
        tests = json.loads(tests_path.read_text(encoding="utf-8"))["h4_tests"]
        raw = tests["cycle_rank_per_node"]["rho"]
        partial = tests["cycle_rank_per_node"]["partial"]["rho"]
        # Figure-level caption rather than an in-axes annotation: placed inside the top panel it
        # sat on top of the plotted series, and the panel is already dense.
        figure.text(
            0.5, -0.005,
            f"Cycle rank per node against the illicit base rate: raw ρ = {raw:+.3f}, "
            f"partial ρ = {partial:+.3f} after controlling for snapshot order and mean degree. "
            f"All 49 snapshots are connected, so cycle rank per node is mean degree rescaled.",
            ha="center", va="top", fontsize=7.2, color="#444444", wrap=True,
        )

    figure.align_ylabels(axes)
    pl.save(figure, FIGURES / "fig11_snapshot_topology_series")


def figure_12_scalability() -> None:
    """Measured cost of the pipeline against radius, subgraph size and homology dimension."""
    import matplotlib.pyplot as plt

    path = PROCESSED / "scalability.csv"
    if not path.exists():
        logger.warning("skipping figure 12: %s is absent (run scripts/measure_scalability.py)",
                       path.name)
        return
    frame = pd.read_csv(path)

    if "axis" not in frame.columns:
        logger.warning("skipping figure 12: scalability.csv predates the axis column; re-run "
                       "scripts/measure_scalability.py")
        return

    figure, axes = plt.subplots(1, 4, figsize=(12.6, 3.1))

    # Every panel selects on the measurement axis. Without that, a row measured while sweeping one
    # factor also satisfies the filter for another: an earlier version plotted the geodesic-Rips
    # row from the filtration sweep as though it were a radius-3 measurement, producing a 3000 ms
    # spike in the middle of a radius curve.
    axis = axes[0]
    radius_rows = frame[(frame.axis == "radius") & (frame.view == "G_tx")]
    for stage, colour, marker in (("classical", pl.PALETTE["structural"], "^"),
                                  ("topological", pl.PALETTE["topological"], "D"),
                                  ("extraction", pl.PALETTE["neutral"], "o")):
        subset = radius_rows[radius_rows.stage == stage].sort_values("radius")
        if subset.empty:
            continue
        axis.plot(subset.radius, subset.ms_per_entity, color=colour, marker=marker, label=stage)
    axis.set_yscale("log")
    axis.set_xlabel("ego radius")
    axis.set_ylabel("ms per entity")
    axis.set_title("Transaction graph, by stage")
    axis.legend()

    axis = axes[1]
    addr_rows = frame[(frame.axis == "radius") & (frame.view == "G_addr")]
    for stage, colour, marker in (("classical", pl.PALETTE["structural"], "^"),
                                  ("topological", pl.PALETTE["topological"], "D"),
                                  ("extraction", pl.PALETTE["neutral"], "o")):
        subset = addr_rows[addr_rows.stage == stage].sort_values("radius")
        if subset.empty:
            continue
        axis.plot(subset.radius, subset.ms_per_entity, color=colour, marker=marker, label=stage)
    axis.set_yscale("log")
    axis.set_xlabel("ego radius")
    axis.set_title("Address graph, by stage")
    axis.legend()

    axis = axes[2]
    size = radius_rows[radius_rows.stage == "topological"].sort_values("median_nodes")
    if not size.empty:
        axis.plot(size.median_nodes, size.ms_per_entity, color=pl.PALETTE["topological"],
                  marker="D", label="transaction graph")
    addr_size = addr_rows[addr_rows.stage == "topological"].sort_values("median_nodes")
    if not addr_size.empty:
        axis.plot(addr_size.median_nodes, addr_size.ms_per_entity, color=pl.PALETTE["hybrid"],
                  marker="s", linestyle="--", label="address graph")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("median subgraph order")
    axis.set_title("Persistent homology vs size")
    axis.legend()

    axis = axes[3]
    filtrations = (frame[(frame.axis == "filtration") & (frame.stage == "topological")]
                   .sort_values("ms_per_entity"))
    if not filtrations.empty:
        labels = [f.replace("_", "\n") for f in filtrations.filtration]
        colours = [pl.PALETTE["illicit"] if v > 100 else pl.PALETTE["topological"]
                   for v in filtrations.ms_per_entity]
        axis.bar(labels, filtrations.ms_per_entity, color=colours, alpha=0.88)
        axis.set_yscale("log")
        for position, value in enumerate(filtrations.ms_per_entity):
            axis.text(position, value * 1.25, f"{value:,.0f}", ha="center", fontsize=6.5)
    axis.set_ylabel("ms per entity")
    axis.set_title("Cost by filtration (radius 3)")
    axis.tick_params(axis="x", labelsize=6.5)

    figure.tight_layout()
    pl.save(figure, FIGURES / "fig12_scalability")


def figure_10_family_comparison(tag: str) -> None:
    """Feature-family comparison, with error bars over the ten seeds."""
    import matplotlib.pyplot as plt

    path = _latest(f"transaction_results_{tag}*.csv")
    if path is None:
        logger.warning("skipping figure 10: no transaction results yet")
        return
    results = pd.read_csv(path)

    # One model per family, chosen on validation, as in the analysis script.
    chosen = (results.groupby(["family", "model"])["validation_pr_auc_at_search"].mean()
              .reset_index().sort_values("validation_pr_auc_at_search", ascending=False)
              .groupby("family").head(1)[["family", "model"]])
    selected = results.merge(chosen, on=["family", "model"], how="inner")

    order = [f for f in ("P", "Pplus", "S", "T", "P+S", "P+T", "S+T", "P+S+T")
             if f in selected.family.unique()]
    grouped = selected.groupby("family")["pr_auc"].agg(["mean", "std"]).reindex(order)

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    pl.plot_family_bars(axes[0], order, grouped["mean"].to_numpy(),
                        grouped["std"].fillna(0).to_numpy())
    axes[0].set_title("Test PR-AUC by feature family")
    pl.annotate_n(axes[0], f"mean ± sd over {selected.seed.nunique()} seeds; "
                           f"test window = steps {sp.TEST_STEPS[0]}–{sp.TEST_STEPS[1]}",
                  loc="upper left")

    axis = axes[1]
    for family in order:
        subset = selected[selected.family == family]
        colour, style, marker = pl.FAMILY_STYLE.get(family, ("#666666", "-", "o"))
        axis.scatter(subset["roc_auc"], subset["pr_auc"], color=colour, marker=marker,
                     s=22, label=family, alpha=0.85, edgecolors="none")
    axis.set_xlabel("ROC-AUC")
    axis.set_ylabel("PR-AUC (illicit)")
    axis.set_title("PR-AUC against ROC-AUC, all seeds")
    axis.legend(ncol=2, fontsize=7)

    figure.tight_layout()
    pl.save(figure, FIGURES / "fig10_family_comparison")


def figure_9_confusion(tag: str) -> None:
    """Confusion matrices at the operating threshold, one per feature family.

    PR-AUC is threshold-free, which is the right primary metric on a 4.6% base rate but hides what
    an analyst would actually receive. These are the counts the runs recorded, averaged over seeds.

    They make one point that the ranking metrics obscure. The topological family does not fail by
    missing illicit transactions -- its recall is the highest of the eight. It fails by flagging
    almost half the licit ones to get there, which is why a metric that accounts for the false
    alarms puts it last. A reader who saw only recall would draw the opposite conclusion.
    """
    import matplotlib.pyplot as plt

    path = _latest(f"transaction_results_{tag}*.csv")
    if path is None:
        logger.warning("skipping figure 9: no transaction results yet")
        return
    results = pd.read_csv(path)
    counts = ["true_positives", "false_positives", "true_negatives", "false_negatives"]
    if not all(column in results.columns for column in counts):
        logger.warning("skipping figure 9: the run did not record confusion counts")
        return

    chosen = (results.groupby(["family", "model"])["validation_pr_auc_at_search"].mean()
              .reset_index().sort_values("validation_pr_auc_at_search", ascending=False)
              .groupby("family").head(1)[["family", "model"]])
    selected = results.merge(chosen, on=["family", "model"], how="inner")

    order = [f for f in ("P", "Pplus", "S", "T", "P+S", "P+T", "S+T", "P+S+T")
             if f in selected.family.unique()]
    figure, axes = plt.subplots(2, 4, figsize=(9.2, 4.6))
    for axis, family in zip(axes.ravel(), order):
        subset = selected[selected.family == family]
        tp, fp = subset.true_positives.mean(), subset.false_positives.mean()
        tn, fn = subset.true_negatives.mean(), subset.false_negatives.mean()
        matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
        # Normalised by true-class total, so the diagonal reads as recall per class and the
        # imbalance does not make every panel look identical.
        totals = matrix.sum(axis=1, keepdims=True)
        shown = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

        axis.imshow(shown, cmap="Blues", vmin=0.0, vmax=1.0)
        for row in range(2):
            for column in range(2):
                axis.text(column, row,
                          "{:.2f}\n{:,.0f}".format(shown[row, column], matrix[row, column]),
                          ha="center", va="center", fontsize=6.5,
                          color="white" if shown[row, column] > 0.5 else "#222222")
        model = subset.model.iloc[0]
        axis.set_title("{}  ({})".format(family, model), fontsize=8)
        axis.set_xticks([0, 1], ["pred licit", "pred illicit"], fontsize=6.5)
        axis.set_yticks([0, 1], ["licit", "illicit"], fontsize=6.5)
    for axis in axes.ravel()[len(order):]:
        axis.set_axis_off()

    figure.suptitle("Confusion matrices at the selected operating threshold, "
                    "mean over {} seeds (row-normalised, counts beneath)"
                    .format(selected.seed.nunique()), fontsize=9)
    figure.tight_layout()
    pl.save(figure, FIGURES / "fig9_confusion_matrices")


def figure_8_curves(tag: str) -> None:
    """Per-time-step PR-AUC by family across the regime change.

    Each point is annotated with that step's illicit count, because steps 45 and 46 hold only 5 and
    2 illicit transactions and a PR-AUC there is nearly meaningless alone.
    """
    import matplotlib.pyplot as plt

    path = _latest(f"transaction_per_step_{tag}*.csv")
    if path is None:
        logger.warning("skipping figure 8: no per-step results yet")
        return
    frame = pd.read_csv(path).dropna(subset=["pr_auc"])
    if frame.empty:
        logger.warning("skipping figure 8: per-step frame is empty")
        return

    figure, axes = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    axis = axes[0]
    for family in [f for f in ("P", "S", "T", "P+S+T") if f in frame.family.unique()]:
        subset = (frame[frame.family == family]
                  .groupby("time_step")["pr_auc"].mean().reset_index())
        colour, style, marker = pl.FAMILY_STYLE.get(family, ("#666666", "-", "o"))
        axis.plot(subset.time_step, subset.pr_auc, color=colour, linestyle=style,
                  marker=marker, label=family)
    axis.axvline(sp.REGIME_SHIFT_STEP, color="#444444", linestyle=":", linewidth=1.0)
    axis.set_ylabel("PR-AUC (illicit)")
    axis.set_title("Per-time-step performance across the regime change")
    axis.legend(ncol=4)

    axis = axes[1]
    counts = frame.groupby("time_step")["n_illicit"].first().reset_index()
    axis.bar(counts.time_step, counts.n_illicit, color=pl.PALETTE["illicit"], alpha=0.7)
    axis.axvline(sp.REGIME_SHIFT_STEP, color="#444444", linestyle=":", linewidth=1.0)
    axis.set_ylabel("illicit\ncount")
    axis.set_xlabel("time step")
    pl.annotate_n(axis, "steps with very few positives give unstable PR-AUC", loc="upper right")

    figure.tight_layout()
    pl.save(figure, FIGURES / "fig8_per_step_performance")


def figures_4_5_6_topology_by_class(radius: int, filtration: str, n_sample: int) -> None:
    """Persistence diagrams, Betti curves and topological profiles, illicit against licit."""
    import matplotlib.pyplot as plt

    from src.features.pipeline import ExtractionConfig, extract_features
    from src.features.topological import betti_curve, default_grid
    from src.graph_construction.graphs import build_tx_graph
    from src.topology.persistence import persistence_of

    features_path = RAW_PP / "Transactions Dataset" / "txs_features.csv"
    if not features_path.exists():
        logger.warning("skipping figures 4-6: dataset absent")
        return

    frame = pd.read_csv(features_path, usecols=["txId", "Time step"])
    classes = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_classes.csv")
    frame = frame.merge(classes, on="txId", how="inner")
    frame["label"] = frame["class"].map(CLASS_TO_LABEL).astype(np.int8)
    labelled = frame[frame.label != -1].reset_index(drop=True)
    # Training window only, so the figures do not display test-window structure.
    labelled = labelled[labelled["Time step"] <= sp.TRAIN_STEPS[1]].reset_index(drop=True)

    store = build_tx_graph(RAW_PP)
    config = ExtractionConfig(view="G_tx", radius=radius, filtration=filtration,
                              complex_dim=2, max_dim=1, include_classical=False)
    grid = default_grid(config.resolution)
    rng = np.random.default_rng(0)

    curves: dict[int, list[np.ndarray]] = {0: [], 1: []}
    class_curves: dict[str, dict[int, list[np.ndarray]]] = {
        "illicit": {0: [], 1: []}, "licit": {0: [], 1: []},
    }
    examples: dict[str, dict] = {}

    for class_name, label in (("illicit", 1), ("licit", 0)):
        pool = labelled[labelled.label == label]["txId"].to_numpy()
        chosen = rng.choice(pool, size=min(n_sample, len(pool)), replace=False)
        best_example = None
        for target in chosen:
            subgraph = store.ego_subgraph(target, radius, max_nodes=25_000)
            if subgraph.number_of_edges() == 0:
                continue
            result = persistence_of(subgraph, filtration, max_dim=1, complex_dim=2)
            for dim in (0, 1):
                class_curves[class_name][dim].append(betti_curve(result, dim, grid))
            n_cycles = result.diagrams.get(1, np.empty((0, 2))).shape[0]
            if best_example is None or n_cycles > best_example[1]:
                best_example = (result, n_cycles, subgraph.number_of_nodes())
        if best_example is not None:
            examples[class_name] = {
                "result": best_example[0], "n_cycles": best_example[1],
                "n_nodes": best_example[2],
            }

    # --- figure 4: example persistence diagrams ---
    if examples:
        figure, axes = plt.subplots(1, len(examples), figsize=(3.3 * len(examples), 3.3))
        axes = np.atleast_1d(axes)
        for axis, (class_name, payload) in zip(axes, examples.items()):
            pl.plot_persistence_diagram(
                axis, payload["result"].diagrams, max_dim=1,
                title=f"{class_name}: {payload['n_nodes']} nodes, "
                      f"{payload['n_cycles']} $H_1$ classes",
            )
        figure.tight_layout()
        pl.save(figure, FIGURES / "fig4_persistence_diagrams")

    # --- figure 5: Betti curve comparison ---
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for axis, dim in zip(axes, (0, 1)):
        illicit = np.array(class_curves["illicit"][dim]) if class_curves["illicit"][dim] else np.empty((0, grid.size))
        licit = np.array(class_curves["licit"][dim]) if class_curves["licit"][dim] else np.empty((0, grid.size))
        pl.plot_betti_curves(axis, grid, illicit, licit, dim=dim)
        axis.set_title(rf"$\beta_{dim}$ curve, median with interquartile band")
    pl.annotate_n(axes[0], f"radius {radius}, {filtration}, training window only", loc="upper left")
    figure.tight_layout()
    pl.save(figure, FIGURES / "fig5_betti_curve_comparison")

    # --- figure 6: profile comparison, normalised so shape is comparable ---
    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    for class_name, colour in (("illicit", pl.PALETTE["illicit"]),
                               ("licit", pl.PALETTE["licit"])):
        stack = class_curves[class_name][1]
        if not stack:
            continue
        values = np.array(stack)
        peak = np.nanmax(values, axis=1, keepdims=True)
        peak[peak == 0] = 1.0
        normalised = values / peak
        axis.plot(grid, np.nanmedian(normalised, axis=0), color=colour, label=class_name)
        axis.fill_between(grid, np.nanpercentile(normalised, 25, axis=0),
                          np.nanpercentile(normalised, 75, axis=0), color=colour,
                          alpha=0.18, linewidth=0)
    axis.set_xlabel("filtration value (rank-normalised)")
    axis.set_ylabel(r"$\beta_1$, scaled by each subgraph's own peak")
    axis.set_title("Shape of the cycle profile, independent of magnitude")
    axis.legend()
    pl.annotate_n(axis, "scaling removes size so only profile shape is compared", loc="upper left")
    figure.tight_layout()
    pl.save(figure, FIGURES / "fig6_topological_profiles")


def main(tag: str, radius: int, filtration: str, n_sample: int, only: list[str] | None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pl.apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    tasks = {
        "11": lambda: figure_11_snapshot_series(),
        "12": lambda: figure_12_scalability(),
        "10": lambda: figure_10_family_comparison(tag),
        "9": lambda: figure_9_confusion(tag),
        "8": lambda: figure_8_curves(tag),
        "456": lambda: figures_4_5_6_topology_by_class(radius, filtration, n_sample),
    }
    selected = only or list(tasks)
    for key in selected:
        if key not in tasks:
            logger.warning("unknown figure key %r", key)
            continue
        started = time.perf_counter()
        try:
            tasks[key]()
        except Exception as exc:
            logger.error("figure %s failed: %s", key, exc, exc_info=True)
            continue
        logger.info("figure group %s done in %.1f s", key, time.perf_counter() - started)

    produced = sorted(p.name for p in FIGURES.glob("*.png"))
    print("\nfigures in figures/:")
    for name in produced:
        print("  " + name)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--n-sample", type=int, default=800, dest="n_sample")
    parser.add_argument("--only", nargs="*", default=None,
                        help="figure keys to build: 456 8 10 11 12")
    raise SystemExit(main(**vars(parser.parse_args())))
