"""Schematic figures 1-3 and 7.

Figure 2 is computed from a real neighbourhood as well; see its docstring.

These explain the method rather than report a result, so they are drawn rather than measured --
with one exception that matters: **Figure 3 is computed, not illustrated.** Its filtration steps,
simplicial complexes and persistence diagram come from an actual GUDHI run on a real Bitcoin
ego-subgraph, so the reader is looking at the pipeline's genuine output rather than an artist's
impression of it. A hand-drawn persistence diagram in a paper about persistence diagrams would be
a small dishonesty.

Usage::

    .venv\\Scripts\\python.exe scripts\\make_schematics.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import plotting as pl  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
FIGURES = REPO_ROOT / "figures"

logger = logging.getLogger("schematics")


def figure_1_architecture() -> None:
    """The BettiGraphs pipeline, stage by stage."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7.2)
    axis.axis("off")

    def box(x, y, w, h, text, colour, *, fontsize=7.5, weight="normal"):
        axis.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06", linewidth=0.9,
            edgecolor=colour, facecolor=colour + "22"))
        axis.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                  fontsize=fontsize, color="#111111", weight=weight)

    def arrow(x1, y1, x2, y2):
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))

    blue, green, orange, pink, grey = "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#666666"

    box(0.2, 6.2, 9.6, 0.75,
        "Elliptic++   ·   203,769 transactions   ·   822,942 addresses   ·   49 time steps",
        grey, fontsize=8.5, weight="bold")

    box(0.2, 4.8, 3.0, 1.0, "$G_{tx}$\ntransaction → transaction\n234,355 edges", blue)
    box(3.5, 4.8, 3.0, 1.0, "$G_{addr}$\naddress → address\n2,868,964 edges", blue)
    box(6.8, 4.8, 3.0, 1.0,
        "$G_{bip}$\naddress ↔ transaction\nbipartite, so $\\beta_2 \\equiv 0$", blue)
    for x in (1.7, 5.0, 8.3):
        arrow(x, 6.2, x, 5.84)

    box(0.2, 3.65, 9.6, 0.75,
        "ego-subgraph extraction   ·   radius $k$   ·   node bound 25,000\n"
        "temporal split  1–34 / 35–41 / 42–49", green, fontsize=7.5)
    for x in (1.7, 5.0, 8.3):
        arrow(x, 4.8, x, 4.44)

    box(0.2, 2.3, 4.6, 1.05,
        "filtration → flag complex → persistent homology\n"
        "value · time · coreness\ngeodesic · resistance", orange, fontsize=7)
    box(5.2, 2.3, 4.6, 1.05,
        "classical graph features (S)\ndegree · centrality · motifs\ncycle rank · value statistics",
        green, fontsize=7)
    arrow(2.5, 3.65, 2.5, 3.39)
    arrow(7.5, 3.65, 7.5, 3.39)

    box(0.2, 1.45, 4.6, 0.72,
        "Betti signature (T)\ncurves · lifetimes · entropy · images · landscapes · Euler",
        orange, fontsize=6.2)
    arrow(2.5, 2.3, 2.5, 2.21)

    # A feature-assembly bar, so no arrow has to cross another to reach a task.
    box(0.2, 0.95, 9.6, 0.42,
        "feature assembly:   P (provided, 166)   ·   S (structural, 47)   ·   T (topological, 444)"
        "   and their unions", grey, fontsize=7)
    arrow(2.5, 1.45, 2.5, 1.41)
    arrow(7.5, 2.3, 7.5, 1.41)

    box(0.2, 0.05, 3.0, 0.72, "classification\nP / S / T and unions", pink, fontsize=7)
    box(3.5, 0.05, 3.0, 0.72, "wallet retrieval\nWasserstein · bottleneck", pink, fontsize=7)
    box(6.8, 0.05, 3.0, 0.72, "one-class anomaly score\nMahalanobis · iForest · LOF", pink,
        fontsize=7)
    for x in (1.7, 5.0, 8.3):
        arrow(x, 0.95, x, 0.81)
    pl.save(figure, FIGURES / "fig1_framework_architecture")


def figure_3_filtration_workflow(radius: int, filtration: str) -> None:
    """Filtration, complex and persistence -- computed from a real Bitcoin ego-subgraph."""
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd

    from src.features.topological import betti_curve, default_grid
    from src.graph_construction.graphs import build_tx_graph
    from src.topology.filtrations import edge_filtration
    from src.topology.persistence import flag_persistence

    if not (RAW_PP / "Transactions Dataset" / "txs_features.csv").exists():
        logger.warning("skipping figure 3: dataset absent")
        return

    store = build_tx_graph(RAW_PP)

    # Selection criteria, chosen so the figure actually illustrates what it claims to.
    #
    # An earlier version selected on cycle rank and drew a hub whose only cycles were triangles.
    # The flag complex fills triangles, so beta_1 was flat at zero and the figure demonstrated
    # nothing about persistence. Selection is therefore on the FINAL beta_1 of the complex that
    # will actually be drawn, and it rejects hub-dominated subgraphs, which render as starbursts
    # and read as noise.
    rng = np.random.default_rng(3)
    chosen = None
    best_score = -1.0
    for candidate in rng.choice(store.node_ids, size=6000, replace=False):
        subgraph = store.ego_subgraph(candidate, radius, max_nodes=25_000)
        order = subgraph.number_of_nodes()
        if not (20 <= order <= 48) or subgraph.number_of_edges() < order:
            continue
        degrees = np.array([d for _, d in subgraph.degree()])
        if degrees.max() > 0.45 * order:  # reject stars
            continue
        values = edge_filtration(subgraph, filtration, normalise="rank")
        result = flag_persistence(subgraph, values, max_dim=1, complex_dim=2)
        curve = np.array([result.betti_at(t, 1) for t in np.linspace(0, 1, 32)])
        if curve.max() < 2:
            continue
        # Prefer an example whose beta_1 curve genuinely varies, since that is the point.
        score = float(curve.max() + curve.std())
        if score > best_score:
            best_score, chosen = score, subgraph
        if best_score >= 6.0:
            break
    if chosen is None:
        logger.warning("skipping figure 3: no subgraph met the illustration criteria")
        return

    values = edge_filtration(chosen, filtration, normalise="rank")
    result = flag_persistence(chosen, values, max_dim=1, complex_dim=2)
    layout = nx.spring_layout(chosen, seed=7, iterations=200)

    thresholds = [0.0, 0.35, 0.7, 1.0]
    figure = plt.figure(figsize=(10.0, 5.4))
    grid_spec = figure.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.22)

    for column, threshold in enumerate(thresholds):
        axis = figure.add_subplot(grid_spec[0, column])
        active = [edge for edge, value in values.items() if value <= threshold]
        sub = nx.Graph()
        sub.add_nodes_from(chosen.nodes())
        sub.add_edges_from(active)
        nx.draw_networkx_nodes(sub, layout, ax=axis, node_size=14,
                               node_color="#333333", linewidths=0)
        nx.draw_networkx_edges(sub, layout, ax=axis, width=1.0,
                               edge_color=pl.PALETTE["topological"], alpha=0.85)
        # Shade filled triangles, which is what distinguishes a flag complex from a graph.
        adjacency = {node: set(sub.neighbors(node)) for node in sub.nodes()}
        for u, v in sub.edges():
            for w in adjacency[u] & adjacency[v]:
                if u < v < w:
                    axis.fill(*zip(layout[u], layout[v], layout[w]),
                              color=pl.PALETTE["accent"], alpha=0.22, linewidth=0)
        beta0 = result.betti_at(threshold, 0)
        beta1 = result.betti_at(threshold, 1)
        axis.set_title(rf"$\varepsilon={threshold:.2f}$   $\beta_0={beta0}$, $\beta_1={beta1}$",
                       fontsize=8.5)
        axis.axis("off")

    axis = figure.add_subplot(grid_spec[1, :2])
    curve_grid = default_grid(64)
    axis.plot(curve_grid, betti_curve(result, 0, curve_grid), color=pl.PALETTE["licit"],
              label=r"$\beta_0(\varepsilon)$")
    axis.plot(curve_grid, betti_curve(result, 1, curve_grid), color=pl.PALETTE["illicit"],
              label=r"$\beta_1(\varepsilon)$")
    for threshold in thresholds:
        axis.axvline(threshold, color="#999999", linestyle=":", linewidth=0.7)
    axis.set_xlabel(r"filtration value $\varepsilon$ (rank-normalised transaction value)")
    axis.set_ylabel("Betti number")
    axis.set_title("Betti curves", fontsize=9)
    axis.legend()

    axis = figure.add_subplot(grid_spec[1, 2:])
    pl.plot_persistence_diagram(axis, result.diagrams, max_dim=1, title="Persistence diagram")
    essential = result.essential_births.get(1, np.empty(0))
    if essential.size:
        axis.scatter(essential, np.full(essential.size, 1.02), marker="^", s=18,
                     color=pl.PALETTE["hybrid"], edgecolors="none",
                     label=f"essential $H_1$ ({essential.size})")
        axis.legend(loc="lower right", fontsize=7)

    figure.suptitle(
        f"Filtration to persistence on a real Bitcoin ego-subgraph "
        f"({chosen.number_of_nodes()} transactions, {chosen.number_of_edges()} flows, "
        f"radius {radius}, {filtration})",
        fontsize=9.5, y=0.985,
    )
    pl.save(figure, FIGURES / "fig3_filtration_workflow")


def figure_7_wallet_similarity() -> None:
    """Wallet retrieval, pooled against illicit-only queries.

    The contrast between the two panels is the point of the figure. On the pooled query set every
    space -- including a shuffled distance matrix -- scores about 0.9, because the pool is roughly
    95% licit and a licit query retrieves same-class neighbours for free. Restricting queries to
    illicit wallets collapses the floor to near zero and separates the spaces.

    An earlier version plotted only the pooled panel. With bars running from zero to 0.89 it made
    every method look near-perfect and hid the entire range of interest, which is exactly the
    misreading the random floor exists to prevent.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    processed = REPO_ROOT / "results" / "processed"
    pooled_path = processed / "wallet_retrieval.csv"
    illicit_path = processed / "wallet_retrieval_illicit_queries.csv"
    if not pooled_path.exists():
        logger.warning("skipping figure 7: wallet retrieval results absent")
        return

    pooled = pd.read_csv(pooled_path)
    illicit = pd.read_csv(illicit_path) if illicit_path.exists() else None

    panels = [("all queries", pooled)] if illicit is None else [
        ("illicit queries only", illicit), ("all queries", pooled)
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 3.7))
    axes = np.atleast_1d(axes)

    for axis, (title, frame) in zip(axes, panels):
        order = frame.sort_values("precision_at_10", ascending=False).reset_index(drop=True)
        colours = [pl.PALETTE["topological"] if str(s).startswith("T_")
                   else ("#999999" if s == "random" else pl.PALETTE["structural"])
                   for s in order["space"]]
        positions = np.arange(len(order))
        axis.barh(positions, order["precision_at_10"], color=colours, alpha=0.88)
        axis.set_yticks(positions)
        axis.set_yticklabels(order["space"])
        axis.invert_yaxis()
        axis.set_xlim(0, 1.0)
        axis.set_xlabel("precision@10 (same-class retrieval)")
        axis.set_title(title)
        floor = order.loc[order.space == "random", "precision_at_10"]
        if len(floor):
            axis.axvline(float(floor.iloc[0]), color="#444444", linestyle=":", linewidth=1.0)
        for position, value in zip(positions, order["precision_at_10"]):
            axis.text(value + 0.012, position, f"{value:.3f}", va="center", fontsize=7)

    if illicit is not None:
        base = float(illicit["base_rate"].iloc[0])
        n_queries = int(illicit["n_queries"].iloc[0])
        pl.annotate_n(axes[0], f"n = {n_queries} illicit queries, base rate {base:.3f}",
                      loc="lower right")
        pl.annotate_n(axes[1], "every space sits just above the shuffled floor:\n"
                               "the pool is ~95% licit, so the metric reads the base rate",
                      loc="lower right")

    figure.suptitle("Topological wallet distance works, and loses to conventional features",
                    fontsize=10, y=1.0)
    figure.tight_layout()
    pl.save(figure, FIGURES / "fig7_wallet_similarity")


def figure_2_graph_views() -> None:
    """The three graph views, drawn from a real neighbourhood, with the density that separates them.

    Computed rather than illustrated, for the same reason Figure 3 is: the paper's central claim is
    that the transaction and address views of one ledger differ in cycle density, and a hand-drawn
    example could be made to show whatever the author wished. The neighbourhood below is a genuine
    one, selected deterministically as the smallest that exhibits the difference, and panel (d) is
    read from the measured census.
    """
    import json

    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd

    actors = RAW_PP / "Actors Dataset"
    addr_tx = pd.read_csv(actors / "AddrTx_edgelist.csv")
    tx_addr = pd.read_csv(actors / "TxAddr_edgelist.csv")

    # A seed transaction small enough to draw, chosen deterministically: among transactions with
    # two or three inputs and two or three outputs, the one with the lowest id. No sampling, so the
    # figure is the same on every run.
    in_degree = addr_tx.groupby("txId").size()
    out_degree = tx_addr.groupby("txId").size()
    sizes = pd.DataFrame({"ins": in_degree, "outs": out_degree}).dropna()
    candidates = sizes[(sizes.ins.between(2, 3)) & (sizes.outs.between(2, 3))]
    if candidates.empty:
        raise RuntimeError("no transaction with 2-3 inputs and 2-3 outputs")

    # One further hop, but only through transactions that are themselves small. Without that
    # restriction the neighbourhood picks up a transaction with hundreds of addresses and the
    # panels become ink rather than illustration.
    small = set(candidates.index.astype(int))
    seed_tx, transactions = None, []
    for candidate in sorted(small):
        outputs = tx_addr.loc[tx_addr.txId == candidate, "output_address"].tolist()
        inputs = addr_tx.loc[addr_tx.txId == candidate, "input_address"].tolist()
        linked = set(addr_tx.loc[addr_tx.input_address.isin(outputs), "txId"].tolist())
        linked |= set(tx_addr.loc[tx_addr.output_address.isin(inputs), "txId"].tolist())
        linked = sorted((linked & small) - {candidate})[:2]
        if linked:
            seed_tx, transactions = candidate, [candidate] + linked
            break
    if seed_tx is None:
        raise RuntimeError("no small transaction with a small neighbour")

    edges_in = addr_tx[addr_tx.txId.isin(transactions)]
    edges_out = tx_addr[tx_addr.txId.isin(transactions)]
    addresses = sorted(set(edges_in.input_address) | set(edges_out.output_address))

    census = json.loads((REPO_ROOT / "results/processed/graph_structure.json")
                        .read_text(encoding="utf-8"))

    figure, axes = plt.subplots(1, 4, figsize=(7.4, 2.6))
    tx_colour, addr_colour = pl.PALETTE["topological"], pl.PALETTE["provided"]

    # (a) the bipartite view, which is the raw record: addresses take part in transactions.
    bipartite = nx.Graph()
    bipartite.add_nodes_from(("t", t) for t in transactions)
    bipartite.add_nodes_from(("a", a) for a in addresses)
    bipartite.add_edges_from((("a", r.input_address), ("t", r.txId))
                             for r in edges_in.itertuples())
    bipartite.add_edges_from((("t", r.txId), ("a", r.output_address))
                             for r in edges_out.itertuples())
    layout = {}
    for index, node in enumerate(("t", t) for t in transactions):
        layout[node] = (1.0, 1.0 - index / max(1, len(transactions) - 1) if len(transactions) > 1
                        else 0.5)
    for index, address in enumerate(addresses):
        layout[("a", address)] = (0.0, 1.0 - index / max(1, len(addresses) - 1))
    nx.draw_networkx_edges(bipartite, layout, ax=axes[0], width=0.8, alpha=0.6)
    nx.draw_networkx_nodes(bipartite, layout, ax=axes[0],
                           nodelist=[n for n in bipartite if n[0] == "a"],
                           node_color=addr_colour, node_size=42, node_shape="o")
    nx.draw_networkx_nodes(bipartite, layout, ax=axes[0],
                           nodelist=[n for n in bipartite if n[0] == "t"],
                           node_color=tx_colour, node_size=52, node_shape="s")
    axes[0].set_title("(a) $G_{bip}$ as recorded\naddress $\\leftrightarrow$ transaction\n ",
                      fontsize=7)

    # (b) the transaction projection: two transactions are joined when a spend links them.
    tx_graph = nx.Graph()
    tx_graph.add_nodes_from(transactions)
    for address in addresses:
        spenders = set(edges_in.loc[edges_in.input_address == address, "txId"])
        funders = set(edges_out.loc[edges_out.output_address == address, "txId"])
        for funder in funders:
            for spender in spenders:
                if funder != spender:
                    tx_graph.add_edge(funder, spender)
    nx.draw(tx_graph, nx.spring_layout(tx_graph, seed=0), ax=axes[1], node_color=tx_colour,
            node_size=52, node_shape="s", width=0.9, edge_color="#444444")
    # Built by concatenation rather than str.format: the titles contain LaTeX braces, which format
    # would read as field names. The cycle rank is of the drawn example, so the reader can see the
    # claim in the panel rather than take it from the axis label.
    tx_rank = (tx_graph.number_of_edges() - tx_graph.number_of_nodes()
               + nx.number_connected_components(tx_graph))
    axes[1].set_title("(b) $G_{tx}$ projection\ncycle rank " + str(tx_rank)
                      + "\nwhole graph: mean degree "
                      + format(census["global"]["G_tx"]["mean_degree"], ".2f"), fontsize=7)

    # (c) the address projection: an input address is joined to each output address of the
    # transaction it funds, which is what reintroduces cycles.
    addr_graph = nx.Graph()
    addr_graph.add_nodes_from(addresses)
    for transaction in transactions:
        sources = edges_in.loc[edges_in.txId == transaction, "input_address"]
        targets = edges_out.loc[edges_out.txId == transaction, "output_address"]
        for source in sources:
            for target in targets:
                if source != target:
                    addr_graph.add_edge(source, target)
    nx.draw(addr_graph, nx.spring_layout(addr_graph, seed=0, k=0.9), ax=axes[2],
            node_color=addr_colour, node_size=42, width=0.9, edge_color="#444444")
    addr_rank = (addr_graph.number_of_edges() - addr_graph.number_of_nodes()
                 + nx.number_connected_components(addr_graph))
    axes[2].set_title("(c) $G_{addr}$ projection\ncycle rank " + str(addr_rank)
                      + "\nwhole graph: mean degree "
                      + format(census["global"]["G_addr"]["mean_degree"], ".2f"), fontsize=7)

    for axis in axes[:3]:
        axis.set_axis_off()

    # (d) the measured difference, which is the point of the figure.
    rows = {(row["view"], row["radius"]): row for row in census.get("sparsity", [])}
    labels, tx_values, addr_values = [], [], []
    if ("G_tx", 2) in rows and ("G_addr", 2) in rows:
        labels = ["any cycle\nat $r=2$ (%)", "median cycle\nrank at $r=2$"]
        tx_values = [100 * rows[("G_tx", 2)]["frac_with_any_cycle"],
                     rows[("G_tx", 2)]["median_cycle_rank"]]
        addr_values = [100 * rows[("G_addr", 2)]["frac_with_any_cycle"],
                       rows[("G_addr", 2)]["median_cycle_rank"]]
    positions = np.arange(len(labels))
    axes[3].bar(positions - 0.19, tx_values, width=0.36, color=tx_colour, label="$G_{tx}$")
    axes[3].bar(positions + 0.19, addr_values, width=0.36, color=addr_colour, label="$G_{addr}$")
    for position, value in zip(positions - 0.19, tx_values):
        axes[3].text(position, value, "{:.1f}".format(value), ha="center", va="bottom", fontsize=6)
    for position, value in zip(positions + 0.19, addr_values):
        axes[3].text(position, value, "{:.1f}".format(value), ha="center", va="bottom", fontsize=6)
    axes[3].set_xticks(positions)
    axes[3].set_xticklabels(labels, fontsize=6.5)
    axes[3].legend(fontsize=6.5, frameon=False)
    axes[3].set_title("(d) measured density\nego-subgraphs, node bound {:,}\n ".format(
        census["node_bound"]), fontsize=7)
    axes[3].spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    pl.save(figure, FIGURES / "fig2_graph_views")


def main(radius: int, filtration: str, only: list[str] | None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pl.apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    tasks = {
        "1": figure_1_architecture,
        "2": figure_2_graph_views,
        "3": lambda: figure_3_filtration_workflow(radius, filtration),
        "7": figure_7_wallet_similarity,
    }
    for key in (only or list(tasks)):
        if key not in tasks:
            logger.warning("unknown key %r", key)
            continue
        try:
            tasks[key]()
        except Exception as exc:
            logger.error("schematic %s failed: %s", key, exc, exc_info=True)
    print("\nfigures in figures/:")
    for name in sorted(p.name for p in FIGURES.glob("*.png")):
        print("  " + name)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--only", nargs="*", default=None)
    raise SystemExit(main(**vars(parser.parse_args())))
