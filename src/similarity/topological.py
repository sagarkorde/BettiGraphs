"""Topological distance between entities, and retrieval evaluation (RQ3 / H3).

Two tiers of distance, for a reason that is about feasibility and is reported rather than hidden:

**Exact diagram distances** -- Wasserstein-1, Wasserstein-2 and bottleneck -- operate on the
persistence diagrams themselves and are the theoretically correct notion. They are also an optimal
transport problem per pair, so an all-pairs computation over hundreds of thousands of wallets is
out of reach.

**Vectorised proxies** -- Euclidean distance between persistence images, Betti curves or full
signatures -- are ordinary distances on fixed-length vectors, so a full distance matrix over tens
of thousands of entities is a single BLAS call.

The project therefore runs large-scale retrieval on the proxies and validates them against the
exact distances on a small random subsample, reporting the rank correlation between the two. If
the proxy does not track the exact distance, that is a finding about the proxy and it gets
reported; it is not assumed away.

What a topological distance does and does not mean
--------------------------------------------------
A small distance between two wallets says their transaction neighbourhoods have similar
multiscale structure. It is **not** evidence of common ownership, and this project makes no
attribution claim. Label agreement in the retrieval metrics below means "the same behavioural
class in a third-party labelling", nothing more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.topology.persistence import PersistenceResult

logger = logging.getLogger(__name__)


#: Exact optimal transport between persistence diagrams is cubic in the number of points, and
#: radius-2 ego-subgraphs on the address graph carry a mean cycle rank of 606. Diagrams are
#: therefore truncated to their most persistent classes before an exact distance is computed.
#: The truncation is applied identically to every diagram and is reported in the manuscript; it
#: keeps the dominant topology, which is what the distance is meant to compare, and discards the
#: short-lived classes that are closest to noise under the stability theorem anyway.
EXACT_DISTANCE_TOP_K = 50


def top_k_diagram(pairs: np.ndarray, k: int = EXACT_DISTANCE_TOP_K) -> np.ndarray:
    """The ``k`` most persistent (birth, death) pairs, longest lifetime first."""
    if pairs is None or pairs.size == 0:
        return np.empty((0, 2), dtype=float)
    if pairs.shape[0] <= k:
        return pairs
    lifetimes = pairs[:, 1] - pairs[:, 0]
    keep = np.argpartition(-lifetimes, k - 1)[:k]
    return pairs[keep]


def wasserstein_distance(
    a: PersistenceResult, b: PersistenceResult, dim: int, *, order: float = 1.0,
    top_k: int | None = EXACT_DISTANCE_TOP_K,
) -> float:
    """Wasserstein distance of order ``order`` between two persistence diagrams.

    Uses GUDHI's optimal-transport implementation. Essential classes are excluded, since a
    diagram with a different number of infinite bars has no finite Wasserstein distance to one
    without; the counts are compared separately as features.

    Args:
        top_k: truncate both diagrams to their ``top_k`` most persistent classes first. Pass
            ``None`` for the untruncated distance, which is exact but cubic and only tractable on
            small diagrams.
    """
    from gudhi.wasserstein import wasserstein_distance as gudhi_wasserstein

    left = a.diagrams.get(dim, np.empty((0, 2)))
    right = b.diagrams.get(dim, np.empty((0, 2)))
    if top_k is not None:
        left, right = top_k_diagram(left, top_k), top_k_diagram(right, top_k)
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(gudhi_wasserstein(left, right, order=order, internal_p=2.0))


def bottleneck_distance(
    a: PersistenceResult, b: PersistenceResult, dim: int,
    *, top_k: int | None = EXACT_DISTANCE_TOP_K,
) -> float:
    """Bottleneck (Wasserstein-infinity) distance between two persistence diagrams."""
    import gudhi

    left = a.diagrams.get(dim, np.empty((0, 2)))
    right = b.diagrams.get(dim, np.empty((0, 2)))
    if top_k is not None:
        left, right = top_k_diagram(left, top_k), top_k_diagram(right, top_k)
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(gudhi.bottleneck_distance(left, right))


def pairwise_euclidean(matrix: np.ndarray, queries: np.ndarray | None = None) -> np.ndarray:
    """Euclidean distance matrix, computed via the Gram-matrix identity.

    Args:
        matrix: ``(n, d)`` pool of vectors.
        queries: ``(q, d)`` queries; defaults to ``matrix`` itself.

    Returns:
        ``(q, n)`` distances. Negative values from floating-point cancellation are clipped to zero
        before the square root, which otherwise yields NaN on the diagonal.
    """
    pool = np.ascontiguousarray(matrix, dtype=np.float64)
    query = pool if queries is None else np.ascontiguousarray(queries, dtype=np.float64)
    pool_sq = np.einsum("ij,ij->i", pool, pool)
    query_sq = np.einsum("ij,ij->i", query, query)
    gram = query @ pool.T
    squared = query_sq[:, None] + pool_sq[None, :] - 2.0 * gram
    # Clipping keeps the square root real; a self-distance still lands around 1e-7 rather than
    # exactly zero, because the cancellation happens before the clip. That is left alone on
    # purpose. Zeroing the diagonal would only be possible in the symmetric form, which would make
    # the two call forms disagree, and it is unnecessary: callers exclude a query from its own
    # ranking by index via ``exclude_self``, never by relying on the distance being zero.
    return np.sqrt(np.clip(squared, 0.0, None))


@dataclass
class RetrievalResult:
    """Retrieval metrics for one distance space."""

    space: str
    n_queries: int
    n_pool: int
    precision_at: dict[int, float]
    recall_at: dict[int, float]
    map_score: float
    ndcg_at: dict[int, float]
    base_rate: float

    def as_dict(self) -> dict:
        return {
            "space": self.space,
            "n_queries": self.n_queries,
            "n_pool": self.n_pool,
            "map": self.map_score,
            "base_rate": self.base_rate,
            **{f"precision_at_{k}": v for k, v in self.precision_at.items()},
            **{f"recall_at_{k}": v for k, v in self.recall_at.items()},
            **{f"ndcg_at_{k}": v for k, v in self.ndcg_at.items()},
        }


def _dcg(relevance: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance * discounts))


def evaluate_retrieval(
    distances: np.ndarray,
    query_labels: np.ndarray,
    pool_labels: np.ndarray,
    *,
    space: str,
    ks: tuple[int, ...] = (10, 50),
    exclude_self: np.ndarray | None = None,
) -> RetrievalResult:
    """Precision@K, Recall@K, MAP and NDCG@K where relevance is label agreement.

    Args:
        distances: ``(q, n)`` distances from each query to each pool item.
        query_labels: ``(q,)`` labels.
        pool_labels: ``(n,)`` labels.
        exclude_self: ``(q,)`` pool indices to mask out, so a query cannot retrieve itself.

    Returns:
        A :class:`RetrievalResult`. Queries with no relevant item in the pool are skipped, since
        recall and average precision are undefined for them and scoring them as zero would drag
        every space down by the same arbitrary amount.
    """
    distances = np.asarray(distances, dtype=float).copy()
    if exclude_self is not None:
        distances[np.arange(distances.shape[0]), exclude_self] = np.inf

    max_k = max(ks)
    order = np.argsort(distances, axis=1, kind="stable")

    precision: dict[int, list[float]] = {k: [] for k in ks}
    recall: dict[int, list[float]] = {k: [] for k in ks}
    ndcg: dict[int, list[float]] = {k: [] for k in ks}
    average_precisions: list[float] = []

    for row in range(distances.shape[0]):
        label = query_labels[row]
        relevant_total = int(np.sum(pool_labels == label))
        if exclude_self is not None and pool_labels[exclude_self[row]] == label:
            relevant_total -= 1
        if relevant_total <= 0:
            continue

        ranked = order[row]
        ranked = ranked[np.isfinite(distances[row, ranked])]
        relevance = (pool_labels[ranked] == label).astype(float)

        for k in ks:
            top = relevance[:k]
            precision[k].append(float(top.mean()) if top.size else 0.0)
            recall[k].append(float(top.sum() / relevant_total))
            ideal = np.ones(min(k, relevant_total))
            ideal_dcg = _dcg(ideal)
            ndcg[k].append(_dcg(top) / ideal_dcg if ideal_dcg > 0 else 0.0)

        # Average precision over the retrieved prefix, which is what a finite pool supports.
        prefix = relevance[:max(max_k, min(1000, relevance.size))]
        hits = np.cumsum(prefix)
        positions = np.arange(1, prefix.size + 1)
        precisions_at_hits = (hits / positions)[prefix > 0]
        average_precisions.append(
            float(precisions_at_hits.sum() / min(relevant_total, prefix.size))
            if precisions_at_hits.size else 0.0
        )

    n_scored = len(average_precisions)
    if n_scored == 0:
        raise ValueError("no query had a relevant item in the pool")

    return RetrievalResult(
        space=space,
        n_queries=n_scored,
        n_pool=int(distances.shape[1]),
        precision_at={k: float(np.mean(v)) for k, v in precision.items()},
        recall_at={k: float(np.mean(v)) for k, v in recall.items()},
        map_score=float(np.mean(average_precisions)),
        ndcg_at={k: float(np.mean(v)) for k, v in ndcg.items()},
        base_rate=float(np.mean([np.mean(pool_labels == label) for label in query_labels])),
    )


def clustering_agreement(
    distances: np.ndarray, labels: np.ndarray, *, n_clusters: int = 2, seed: int = 0
) -> dict[str, float]:
    """Cluster on a precomputed distance matrix and compare against labels.

    Uses spectral clustering on an RBF affinity derived from the distances, plus the silhouette
    score in the same metric. Reported as ARI and NMI.
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    finite = distances[np.isfinite(distances)]
    scale = float(np.median(finite)) if finite.size else 1.0
    if scale <= 0:
        scale = 1.0
    affinity = np.exp(-(distances / scale) ** 2)
    np.fill_diagonal(affinity, 1.0)

    try:
        assignment = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=seed,
            assign_labels="kmeans",
        ).fit_predict(affinity)
    except Exception as exc:
        logger.warning("spectral clustering failed (%s); reporting NaN", exc)
        return {"ari": float("nan"), "nmi": float("nan"), "silhouette": float("nan"),
                "degenerate": True, "reason": f"clusterer raised {type(exc).__name__}"}

    sizes = np.bincount(assignment, minlength=n_clusters)
    smallest = int(sizes.min())
    # A clusterer that puts almost everything in one cluster has failed, and its agreement scores
    # are an artefact of that rather than a statement about the data. This was not hypothetical:
    # two quite different distance spaces returned ARI and NMI identical to four decimal places,
    # which is the signature of both collapsing to the same near-trivial split. Reporting -0.06 as
    # if it measured cluster quality would have been wrong.
    degenerate = smallest < max(5, 0.02 * len(labels))

    try:
        silhouette = float(silhouette_score(distances, assignment, metric="precomputed"))
    except ValueError:
        silhouette = float("nan")

    result = {
        "ari": float(adjusted_rand_score(labels, assignment)),
        "nmi": float(normalized_mutual_info_score(labels, assignment)),
        "silhouette": silhouette,
        "cluster_sizes": sizes.tolist(),
        "degenerate": bool(degenerate),
    }
    if degenerate:
        result["reason"] = (
            f"smallest cluster holds {smallest} of {len(labels)} points; the agreement scores "
            "describe a near-trivial partition, not cluster structure"
        )
        logger.warning("clustering is degenerate: sizes %s", sizes.tolist())
    return result


def proxy_fidelity(
    exact: np.ndarray, proxy: np.ndarray
) -> dict[str, float]:
    """How well a vectorised distance tracks the exact diagram distance.

    Reported so that the large-scale retrieval results, which necessarily use the proxy, can be
    interpreted. Both Spearman and Pearson are given: the retrieval metrics depend only on the
    ranking, so Spearman is the one that matters, and Pearson is included for completeness.
    """
    from scipy import stats

    exact = np.asarray(exact, dtype=float).ravel()
    proxy = np.asarray(proxy, dtype=float).ravel()
    keep = np.isfinite(exact) & np.isfinite(proxy)
    exact, proxy = exact[keep], proxy[keep]
    if exact.size < 3:
        return {"spearman": float("nan"), "pearson": float("nan"), "n": int(exact.size)}
    return {
        "spearman": float(stats.spearmanr(exact, proxy).statistic),
        "pearson": float(stats.pearsonr(exact, proxy).statistic),
        "n": int(exact.size),
    }
