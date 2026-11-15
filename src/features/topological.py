"""The Betti Signature: fixed-length topological descriptors of a persistence result.

A persistence diagram is a multiset of variable size, so it cannot be fed to a classifier
directly. This module turns one into a fixed-length, named vector. Names matter: the
interpretability analysis attributes model behaviour to individual descriptors, which is only
possible if every coordinate has a stable, meaningful name.

Descriptor families
-------------------
======================  ==============================================================
``betti_curve``         ``beta_p`` sampled on a fixed grid. The multiscale profile.
``stats``               Lifetime count, sum, mean, max, std, quantiles; plus the
                        number of essential classes.
``entropy``             Persistence entropy of the lifetime distribution.
``image``               Persistence image (Adams et al. 2017), flattened.
``landscape``           First ``k`` persistence landscape functions, flattened.
``euler``               Euler-characteristic curve, ``chi(t) = sum_p (-1)^p beta_p(t)``.
======================  ==============================================================

No leakage by construction
--------------------------
With ``rank`` normalisation (the project default) every filtration value lies in ``[0, 1]``, so
the Betti-curve grid and the persistence-image range are **fixed constants**, not quantities
fitted to data. That removes an entire class of leakage: no descriptor here depends on any
statistic pooled across the train/test boundary. If a caller switches to ``minmax`` or ``none``
normalisation, the grid must be fitted on the training split only and passed in explicitly --
:func:`betti_signature` requires the grid rather than inferring it, precisely so that this
cannot happen by accident.

Why a curve rather than a cycle count
-------------------------------------
For a one-dimensional complex ``beta_1 = |E| - |V| + beta_0``, so a single ``beta_1`` value *is*
the cycle rank and carries no information a classical feature set lacks. The content is in the
profile: the function ``t -> beta_1(K_t)`` separates a graph whose cycles are all short-lived
from one whose cycles persist across the filtration, even when their cycle ranks agree. The
classical cycle-rank scalar is deliberately included in the baseline feature family so that
this comparison is made honestly rather than assumed.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from src.topology.persistence import PersistenceResult

logger = logging.getLogger(__name__)

#: Default Betti-curve / Euler-curve resolution. Ablated over {16, 32, 64}.
DEFAULT_RESOLUTION = 32
#: Quantiles reported for the lifetime distribution.
LIFETIME_QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)
#: Fixed grid endpoints, valid because rank normalisation maps filtration values into [0, 1].
RANK_GRID_RANGE: tuple[float, float] = (0.0, 1.0)


def default_grid(resolution: int = DEFAULT_RESOLUTION) -> np.ndarray:
    """Fixed filtration grid on ``[0, 1]``, appropriate for rank-normalised filtrations."""
    return np.linspace(RANK_GRID_RANGE[0], RANK_GRID_RANGE[1], resolution)


def lifetimes(result: PersistenceResult, dim: int) -> np.ndarray:
    """Finite lifetimes ``death - birth`` in dimension ``dim``."""
    pairs = result.diagrams.get(dim)
    if pairs is None or pairs.size == 0:
        return np.empty(0, dtype=float)
    return pairs[:, 1] - pairs[:, 0]


def betti_curve(result: PersistenceResult, dim: int, grid: np.ndarray) -> np.ndarray:
    """``beta_dim`` evaluated at every point of ``grid``.

    Essential classes never die, but they are still **born** at a particular filtration value, and
    they are counted only from that point on. Counting them from the start of the filtration
    instead would be a silent error with real consequences: in a one-dimensional complex no cycle
    can ever die, so every H1 class is essential, and a flat offset would collapse the whole
    ``beta_1`` curve to a constant equal to the cycle rank -- throwing away exactly the multiscale
    structure that distinguishes a Betti curve from the scalar already in the classical baseline.
    """
    points = grid[:, None]

    births = result.essential_births.get(dim)
    if births is not None and births.size:
        base = (births[None, :] <= points).sum(axis=1).astype(float)
    else:
        # No recorded births (e.g. a hand-constructed result): fall back to the count, which is
        # correct whenever the essential classes are born at the start of the filtration.
        base = np.full(grid.shape, float(result.essential.get(dim, 0)), dtype=float)

    pairs = result.diagrams.get(dim)
    if pairs is None or pairs.size == 0:
        return base
    alive = (pairs[:, 0][None, :] <= points) & (pairs[:, 1][None, :] > points)
    return base + alive.sum(axis=1).astype(float)


def euler_curve(result: PersistenceResult, grid: np.ndarray, max_dim: int) -> np.ndarray:
    """Euler-characteristic curve ``chi(t) = sum_p (-1)^p beta_p(t)``.

    Far cheaper than the individual Betti curves and, per the geometric-topological literature,
    often nearly as discriminative -- so it serves as an efficiency comparator in the ablation.
    """
    chi = np.zeros(grid.shape, dtype=float)
    for dim in range(max_dim + 1):
        chi += ((-1.0) ** dim) * betti_curve(result, dim, grid)
    return chi


def persistence_entropy(result: PersistenceResult, dim: int, *, normalised: bool = True) -> float:
    """Shannon entropy of the normalised lifetime distribution.

    Low entropy means a few dominant topological features; high entropy means lifetime mass is
    spread evenly. Returns 0.0 when there are no finite classes, which is the correct
    degenerate value rather than NaN.
    """
    life = lifetimes(result, dim)
    total = float(life.sum())
    if life.size == 0 or total <= 0.0:
        return 0.0
    probabilities = life / total
    probabilities = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    if normalised and probabilities.size > 1:
        entropy /= float(np.log(probabilities.size))
    return entropy


def persistence_statistics(result: PersistenceResult, dim: int) -> dict[str, float]:
    """Summary statistics of the dimension-``dim`` persistence diagram."""
    life = lifetimes(result, dim)
    pairs = result.diagrams.get(dim, np.empty((0, 2)))
    stats: dict[str, float] = {
        f"h{dim}_n_finite": float(life.size),
        f"h{dim}_n_essential": float(result.essential.get(dim, 0)),
        f"h{dim}_life_sum": float(life.sum()) if life.size else 0.0,
        f"h{dim}_life_mean": float(life.mean()) if life.size else 0.0,
        f"h{dim}_life_std": float(life.std()) if life.size > 1 else 0.0,
        f"h{dim}_life_max": float(life.max()) if life.size else 0.0,
        f"h{dim}_life_l2": float(np.sqrt(np.sum(life**2))) if life.size else 0.0,
        f"h{dim}_birth_mean": float(pairs[:, 0].mean()) if pairs.size else 0.0,
        f"h{dim}_death_mean": float(pairs[:, 1].mean()) if pairs.size else 0.0,
        f"h{dim}_entropy": persistence_entropy(result, dim),
    }
    for q in LIFETIME_QUANTILES:
        key = f"h{dim}_life_q{int(q * 100):02d}"
        stats[key] = float(np.quantile(life, q)) if life.size else 0.0
    return stats


def persistence_image(
    result: PersistenceResult,
    dim: int,
    *,
    resolution: int = 8,
    sigma: float = 0.05,
    value_range: tuple[float, float] = RANK_GRID_RANGE,
) -> np.ndarray:
    """Persistence image in birth-persistence coordinates, flattened row-major.

    Implemented directly rather than via a fitted transformer so that the grid is a fixed
    constant and no train/test-dependent range creeps in. Each finite class contributes an
    isotropic Gaussian weighted linearly by its persistence, which is the standard stable
    weighting of Adams et al. (2017).

    Args:
        resolution: output is ``resolution x resolution``.
        sigma: Gaussian bandwidth in filtration units.
        value_range: extent of both axes.
    """
    size = resolution * resolution
    pairs = result.diagrams.get(dim)
    if pairs is None or pairs.size == 0:
        return np.zeros(size, dtype=float)

    births = pairs[:, 0]
    persistences = pairs[:, 1] - pairs[:, 0]
    lo, hi = value_range
    axis = np.linspace(lo, hi, resolution)
    bx, py = np.meshgrid(axis, axis, indexing="ij")

    image = np.zeros((resolution, resolution), dtype=float)
    two_sigma_sq = 2.0 * sigma * sigma
    for birth, persistence in zip(births, persistences):
        if persistence <= 0.0:
            continue
        kernel = np.exp(-(((bx - birth) ** 2) + ((py - persistence) ** 2)) / two_sigma_sq)
        image += persistence * kernel  # linear weighting by persistence -> stability
    image /= two_sigma_sq * np.pi  # normalise the Gaussian
    return image.reshape(-1)


def persistence_landscape(
    result: PersistenceResult,
    dim: int,
    *,
    num_landscapes: int = 3,
    resolution: int = DEFAULT_RESOLUTION,
    value_range: tuple[float, float] = RANK_GRID_RANGE,
) -> np.ndarray:
    """First ``num_landscapes`` persistence landscape functions, concatenated.

    The ``k``-th landscape is the ``k``-th largest value of the tent functions
    ``max(min(t - b, d - t), 0)`` over all classes, evaluated on a fixed grid.
    """
    grid = np.linspace(value_range[0], value_range[1], resolution)
    pairs = result.diagrams.get(dim)
    if pairs is None or pairs.size == 0:
        return np.zeros(num_landscapes * resolution, dtype=float)

    births = pairs[:, 0][None, :]
    deaths = pairs[:, 1][None, :]
    points = grid[:, None]
    tents = np.maximum(np.minimum(points - births, deaths - points), 0.0)  # (R, k)

    # Partial sort is enough: we only need the top `num_landscapes` per grid point.
    take = min(num_landscapes, tents.shape[1])
    ordered = -np.sort(-tents, axis=1)[:, :take]
    out = np.zeros((resolution, num_landscapes), dtype=float)
    out[:, :take] = ordered
    return out.T.reshape(-1)


def betti_signature(
    result: PersistenceResult,
    *,
    max_dim: int = 1,
    grid: np.ndarray | None = None,
    families: Sequence[str] = ("betti_curve", "stats", "image", "landscape", "euler"),
    image_resolution: int = 8,
    num_landscapes: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the full topological signature and its feature names.

    Args:
        result: output of :mod:`src.topology.persistence`.
        max_dim: highest homology dimension to include.
        grid: filtration grid. Required explicitly when the filtration was not rank-normalised,
            so that a train-fitted grid cannot be silently replaced by the default.
        families: which descriptor families to include. The ablation switches these.
        image_resolution: persistence-image side length.
        num_landscapes: number of landscape functions.

    Returns:
        ``(vector, names)`` with ``len(vector) == len(names)``.
    """
    if grid is None:
        if result.meta.get("normalise") not in (None, "rank"):
            raise ValueError(
                "a filtration grid must be supplied explicitly when normalisation is "
                f"{result.meta.get('normalise')!r}; the default grid is only valid for rank "
                "normalisation, and inferring one from data risks train/test leakage"
            )
        grid = default_grid()

    values: list[np.ndarray] = []
    names: list[str] = []

    if "betti_curve" in families:
        for dim in range(max_dim + 1):
            curve = betti_curve(result, dim, grid)
            values.append(curve)
            names.extend(f"bc_h{dim}_{i:03d}" for i in range(curve.size))

    if "stats" in families:
        for dim in range(max_dim + 1):
            stats = persistence_statistics(result, dim)
            values.append(np.asarray(list(stats.values()), dtype=float))
            names.extend(stats.keys())

    if "image" in families:
        for dim in range(max_dim + 1):
            image = persistence_image(result, dim, resolution=image_resolution)
            values.append(image)
            names.extend(f"pi_h{dim}_{i:03d}" for i in range(image.size))

    if "landscape" in families:
        for dim in range(max_dim + 1):
            land = persistence_landscape(result, dim, num_landscapes=num_landscapes,
                                         resolution=grid.size)
            values.append(land)
            names.extend(f"pl_h{dim}_{i:03d}" for i in range(land.size))

    if "euler" in families:
        chi = euler_curve(result, grid, max_dim)
        values.append(chi)
        names.extend(f"euler_{i:03d}" for i in range(chi.size))

    if not values:
        raise ValueError(f"no descriptor family selected from {families!r}")

    vector = np.concatenate(values).astype(float)
    assert vector.size == len(names), f"{vector.size} values but {len(names)} names"
    return vector, names
