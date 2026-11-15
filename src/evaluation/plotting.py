"""Publication figure style and shared plot helpers.

Every figure in the manuscript is produced by a script reading ``results/``. No value is ever typed
into a figure by hand, which is why the helpers here take dataframes rather than literals.

Style choices are for print legibility and for readers who cannot distinguish red from green:
a colourblind-safe palette, distinct line styles and markers as well as distinct colours, and no
information carried by colour alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

#: Okabe-Ito, which stays distinguishable under the common forms of colour vision deficiency
#: and in greyscale print.
PALETTE = {
    "illicit": "#D55E00",
    "licit": "#0072B2",
    "unknown": "#999999",
    "provided": "#0072B2",
    "structural": "#009E73",
    "topological": "#D55E00",
    "hybrid": "#CC79A7",
    "accent": "#E69F00",
    "neutral": "#56B4E9",
}

FAMILY_STYLE = {
    "P": ("#0072B2", "-", "o"),
    "Pplus": ("#56B4E9", "--", "s"),
    "S": ("#009E73", "-.", "^"),
    "T": ("#D55E00", "-", "D"),
    "P+S": ("#E69F00", "--", "v"),
    "P+T": ("#CC79A7", "-", "P"),
    "S+T": ("#F0E442", "-.", "X"),
    "P+S+T": ("#000000", "-", "*"),
}

FIGURE_DPI = 300


def apply_style() -> None:
    """Apply the project's matplotlib style. Call once per script."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "errorbar.capsize": 2.5,
    })


def save(figure: plt.Figure, path: Path, *, also_pdf: bool = True) -> None:
    """Save as PNG and, by default, PDF as well, since journals ask for vector figures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"))
    if also_pdf:
        figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    logger.info("wrote %s", path.with_suffix(".png").name)


def annotate_n(axis: plt.Axes, text: str, *, loc: str = "lower right") -> None:
    """Put a sample-size note on an axis.

    Used liberally: a per-time-step metric without its positive count is not interpretable, and
    several Elliptic steps hold only a handful of illicit transactions.
    """
    positions = {
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
    }
    x, y, ha, va = positions[loc]
    axis.text(x, y, text, transform=axis.transAxes, ha=ha, va=va, fontsize=7,
              color="#555555")


def plot_betti_curves(
    axis: plt.Axes,
    grid: np.ndarray,
    illicit: np.ndarray,
    licit: np.ndarray,
    *,
    dim: int,
    band: str = "iqr",
) -> None:
    """Mean Betti curve per class with an interquartile band.

    The band is the interquartile range rather than a standard error: these distributions are
    strongly right-skewed and a symmetric error bar would imply a symmetry that is not there.
    """
    for values, label, colour in ((illicit, "illicit", PALETTE["illicit"]),
                                  (licit, "licit", PALETTE["licit"])):
        if values.size == 0:
            continue
        centre = np.nanmedian(values, axis=0)
        axis.plot(grid, centre, color=colour, label=f"{label} (n={values.shape[0]:,})")
        if band == "iqr":
            low = np.nanpercentile(values, 25, axis=0)
            high = np.nanpercentile(values, 75, axis=0)
            axis.fill_between(grid, low, high, color=colour, alpha=0.18, linewidth=0)
    axis.set_xlabel("filtration value (rank-normalised)")
    axis.set_ylabel(rf"$\beta_{dim}$")
    axis.legend()


def plot_persistence_diagram(
    axis: plt.Axes, diagrams: dict[int, np.ndarray], *, max_dim: int = 1,
    title: str | None = None,
) -> None:
    """Scatter a persistence diagram with the diagonal drawn."""
    markers = {0: "o", 1: "^", 2: "s"}
    colours = {0: PALETTE["licit"], 1: PALETTE["illicit"], 2: PALETTE["structural"]}
    limit = 1.0
    for dim in range(max_dim + 1):
        pairs = diagrams.get(dim)
        if pairs is None or pairs.size == 0:
            continue
        axis.scatter(pairs[:, 0], pairs[:, 1], s=14, alpha=0.7, marker=markers.get(dim, "o"),
                     color=colours.get(dim, "#333333"), edgecolors="none",
                     label=rf"$H_{dim}$ ({pairs.shape[0]})")
        limit = max(limit, float(np.nanmax(pairs)))
    axis.plot([0, limit], [0, limit], color="#777777", linewidth=0.8, linestyle=":")
    axis.set_xlabel("birth")
    axis.set_ylabel("death")
    axis.set_aspect("equal", adjustable="box")
    if title:
        axis.set_title(title)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(loc="lower right")


def plot_family_bars(
    axis: plt.Axes, families: list[str], means: np.ndarray, errors: np.ndarray,
    *, ylabel: str = "test PR-AUC (illicit)",
) -> None:
    """Bar chart of a metric by feature family, with error bars over seeds."""
    positions = np.arange(len(families))
    colours = [FAMILY_STYLE.get(f, ("#666666",))[0] for f in families]
    axis.bar(positions, means, yerr=errors, color=colours, alpha=0.85,
             error_kw={"elinewidth": 1.0, "ecolor": "#333333"})
    axis.set_xticks(positions)
    axis.set_xticklabels(families, rotation=0)
    axis.set_ylabel(ylabel)
    for position, mean, error in zip(positions, means, errors):
        axis.text(position, mean + error + 0.008, f"{mean:.3f}", ha="center", fontsize=7)
