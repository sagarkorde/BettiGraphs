"""Parallel per-entity feature extraction, with a configuration-keyed cache.

For each target entity this extracts an ego-subgraph once and computes both the classical (S) and
topological (T) feature families **on that same subgraph**, which is what keeps the S-versus-T
comparison about representation rather than about neighbourhood.

Caching is keyed by a hash of the full extraction configuration, so changing the radius, the
filtration, the complex dimension or the node bound produces a different cache file rather than
silently reusing stale vectors. The configuration is stored inside the cache file too, so a
result can always be traced back to the settings that produced it.

Cost, measured on the transaction graph
---------------------------------------
At radius 4 the split is roughly 0.9 ms to extract the subgraph, **1.1 ms** for persistent
homology plus the whole Betti signature, and **350 ms** for the classical family. The classical
centrality features are about two orders of magnitude more expensive than the topology, which is
the reverse of the usual expectation and is reported as a finding rather than buried.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from math import ceil
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src.features.classical import classical_features
from src.features.topological import betti_signature, default_grid
from src.graph_construction.graphs import GraphStore
from src.topology.persistence import persistence_of

logger = logging.getLogger(__name__)


def _feature_code_version() -> str:
    """Hash of the source of every module that produces a feature value.

    Without this, editing a feature implementation leaves the cache key unchanged and a later run
    silently reuses vectors computed by the old code. That already happened once during
    development: an optimisation pass changed the closeness normalisation and added a column, and
    the next run loaded the stale 46-column cache without complaint. Hashing the source makes the
    cache self-invalidating, which is the only version of this that stays correct under a code
    change nobody remembered to think about.
    """
    digest = hashlib.sha256()
    for module in ("classical.py", "topological.py"):
        path = Path(__file__).parent / module
        digest.update(path.read_bytes() if path.exists() else b"")
    for relative in ("../topology/persistence.py", "../topology/filtrations.py",
                     "../graph_construction/graphs.py"):
        path = (Path(__file__).parent / relative).resolve()
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()[:12]


def _load_snapshot() -> dict:
    """Machine load at extraction time, so a contended timing can be spotted later.

    Timings in this module are wall-clock per call. On a busy machine they measure contention as
    much as work: one measurement taken beside another job reported 4.5x the cost of identical
    work. Recording the load makes that detectable instead of silently wrong.
    """
    snapshot: dict = {"cpu_count": os.cpu_count()}
    try:
        import psutil

        snapshot["cpu_percent_at_start"] = psutil.cpu_percent(interval=0.3)
        snapshot["python_processes"] = sum(
            1 for p in psutil.process_iter(["name"])
            if "python" in (p.info.get("name") or "").lower()
        )
        snapshot["timing_uncontended"] = bool(snapshot["cpu_percent_at_start"] < 60.0)
    except Exception:
        snapshot["timing_uncontended"] = None
    return snapshot


@dataclass(frozen=True)
class ExtractionConfig:
    """Everything that affects the extracted vectors. Hashed to key the cache.

    The hash covers both the settings here **and** the source of the feature-producing modules,
    so a code change invalidates the cache automatically.
    """

    view: str = "G_tx"
    radius: int = 2
    filtration: str = "value_superlevel"
    normalise: str = "rank"
    max_dim: int = 1
    complex_dim: int = 2
    max_nodes: int = 25_000
    resolution: int = 32
    image_resolution: int = 8
    num_landscapes: int = 3
    families: tuple[str, ...] = ("betti_curve", "stats", "image", "landscape", "euler")
    include_classical: bool = True

    def fingerprint(self) -> str:
        payload = json.dumps(
            {**asdict(self), "_code": _feature_code_version()}, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def cache_path(self, root: Path, targets: np.ndarray | None = None) -> Path:
        """Cache file for this configuration and, optionally, this target set.

        The target set belongs in the key. Without it, two runs that share a configuration but
        extract different entities -- an evaluation pool and a training reference sample, say --
        map to the same file and overwrite each other on every alternation, so each recomputes
        what the other just discarded. That thrash is silent and can cost hours.

        ``targets=None`` reproduces the original path, so caches written before this change are
        still found and reused.
        """
        stem = f"features_{self.view}_r{self.radius}_{self.filtration}_{self.fingerprint()}"
        if targets is not None:
            digest = hashlib.sha256(
                np.asarray(targets).astype(str).tobytes()
            ).hexdigest()[:8]
            stem = f"{stem}_t{digest}"
        return root / f"{stem}.npz"


@dataclass
class FeatureMatrices:
    """Extracted features plus the bookkeeping needed to report honestly."""

    entity_ids: np.ndarray
    topological: np.ndarray
    topological_names: list[str]
    classical: np.ndarray | None
    classical_names: list[str]
    n_nodes: np.ndarray
    n_edges: np.ndarray
    truncated: np.ndarray
    seconds_topological: float = 0.0
    seconds_classical: float = 0.0
    config: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    @property
    def n_entities(self) -> int:
        return len(self.entity_ids)

    def summary(self) -> dict:
        return {
            "n_entities": self.n_entities,
            "n_topological_features": self.topological.shape[1],
            "n_classical_features": 0 if self.classical is None else self.classical.shape[1],
            "frac_truncated": float(self.truncated.mean()) if self.truncated.size else 0.0,
            "frac_empty_subgraph": float((self.n_edges == 0).mean()) if self.n_edges.size else 0.0,
            "n_failures": len(self.failures),
            "frac_failed": (len(self.failures) / self.n_entities) if self.n_entities else 0.0,
            "median_nodes": float(np.median(self.n_nodes)) if self.n_nodes.size else 0.0,
            "median_edges": float(np.median(self.n_edges)) if self.n_edges.size else 0.0,
            "seconds_topological": self.seconds_topological,
            "seconds_classical": self.seconds_classical,
        }


def _extract_chunk(
    node_ids: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    weight: np.ndarray,
    timestep: np.ndarray,
    out_degree: np.ndarray,
    in_degree: np.ndarray,
    node_timestep: np.ndarray,
    view: str,
    targets: np.ndarray,
    config: ExtractionConfig,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, float, float,
           list[str], list[str], list[tuple[str, str]]]:
    """Worker: extract features for one chunk of targets.

    The graph is passed as its component arrays rather than as a ``GraphStore``, so joblib can
    memory-map the large ones instead of pickling a copy into every worker.
    """
    store = GraphStore(
        node_ids=node_ids, indptr=indptr, indices=indices, weight=weight, timestep=timestep,
        out_degree=out_degree, in_degree=in_degree, node_timestep=node_timestep, name=view,
    )
    grid = default_grid(config.resolution)

    topo_rows: list[np.ndarray] = []
    class_rows: list[np.ndarray] = []
    sizes_n: list[int] = []
    sizes_e: list[int] = []
    truncated: list[bool] = []
    failures: list[tuple[str, str]] = []
    t_topo = t_class = 0.0
    topo_names: list[str] = []
    class_names: list[str] = []

    for target in targets:
        subgraph = store.ego_subgraph(target, config.radius, max_nodes=config.max_nodes)
        sizes_n.append(subgraph.number_of_nodes())
        sizes_e.append(subgraph.number_of_edges())
        truncated.append(bool(subgraph.graph["truncated"]))

        start = time.perf_counter()
        computed = False
        if subgraph.number_of_edges() > 0:
            try:
                result = persistence_of(
                    subgraph, config.filtration, max_dim=config.max_dim,
                    complex_dim=config.complex_dim, normalise=config.normalise,
                )
                vector, names = betti_signature(
                    result, max_dim=config.max_dim, grid=grid, families=config.families,
                    image_resolution=config.image_resolution,
                    num_landscapes=config.num_landscapes,
                )
                computed = True
            except (MemoryError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                # One pathological entity must not abort a batch of thousands in an unattended
                # run. The failure is counted and reported so the rate is visible, rather than
                # disappearing into a zero row that looks like a legitimate empty neighbourhood.
                # Effective-resistance filtrations refuse oversized subgraphs by design, which is
                # the common case here.
                failures.append((str(target), type(exc).__name__))
        if not computed:
            # An isolated entity has no topology to measure. Emit the zero vector of the right
            # length rather than dropping the row, so the entity set stays aligned with the
            # labels, and record it in the empty-subgraph rate.
            if not topo_names:
                from src.topology.persistence import PersistenceResult

                empty = PersistenceResult(
                    {d: np.empty((0, 2)) for d in range(config.max_dim + 1)},
                    {d: 0 for d in range(config.max_dim + 1)}, 0.0, 0, 0,
                    meta={"normalise": config.normalise},
                )
                vector, names = betti_signature(
                    empty, max_dim=config.max_dim, grid=grid, families=config.families,
                    image_resolution=config.image_resolution,
                    num_landscapes=config.num_landscapes,
                )
            else:
                vector, names = np.zeros(len(topo_names)), topo_names
        t_topo += time.perf_counter() - start
        topo_names = names
        topo_rows.append(vector)

        if config.include_classical:
            start = time.perf_counter()
            vector_s, names_s = classical_features(subgraph)
            t_class += time.perf_counter() - start
            class_names = names_s
            class_rows.append(vector_s)

    return (
        np.vstack(topo_rows) if topo_rows else np.empty((0, 0)),
        np.vstack(class_rows) if class_rows else None,
        np.asarray(sizes_n, dtype=np.int32),
        np.asarray(sizes_e, dtype=np.int32),
        np.asarray(truncated, dtype=bool),
        t_topo, t_class, topo_names, class_names, failures,
    )


def extract_features(
    store: GraphStore,
    targets: np.ndarray,
    config: ExtractionConfig,
    *,
    n_jobs: int = -1,
    chunk_size: int = 512,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> FeatureMatrices:
    """Extract S and T features for ``targets``, in parallel, with caching.

    Args:
        store: the graph view.
        targets: entity identifiers, in the order the returned rows will follow.
        config: extraction settings; also the cache key.
        n_jobs: joblib worker count, ``-1`` for all cores.
        chunk_size: targets per task. Chunking amortises worker startup and keeps the graph
            arrays from being re-sent per entity.
        cache_dir: where to read and write the cache.
        use_cache: set False to force recomputation.
    """
    targets = np.asarray(targets)
    cache_path = config.cache_path(cache_dir, targets) if cache_dir else None

    # Candidate files, newest scheme first, then the legacy target-less name so that caches
    # written before the target set entered the key are still reused rather than recomputed.
    candidates: list[Path] = []
    if cache_dir:
        candidates = [cache_path, config.cache_path(cache_dir)]

    for candidate in candidates if use_cache else []:
        if not candidate.exists():
            continue
        logger.info("loading cached features from %s", candidate.name)
        blob = np.load(candidate, allow_pickle=True)
        cached_ids = blob["entity_ids"]
        if len(cached_ids) == len(targets) and np.array_equal(
            cached_ids.astype(str), targets.astype(str)
        ):
            return FeatureMatrices(
                entity_ids=cached_ids,
                topological=blob["topological"],
                topological_names=list(blob["topological_names"]),
                classical=blob["classical"] if "classical" in blob.files else None,
                classical_names=list(blob["classical_names"]) if "classical_names" in blob.files else [],
                n_nodes=blob["n_nodes"], n_edges=blob["n_edges"], truncated=blob["truncated"],
                seconds_topological=float(blob["seconds_topological"]),
                seconds_classical=float(blob["seconds_classical"]),
                config=json.loads(str(blob["config"])),
                failures=(json.loads(str(blob["failures"]))
                          if "failures" in blob.files else []),
            )
        logger.info("cache at %s holds a different target set; trying the next candidate",
                    candidate.name)

    # Chunk so that every worker gets several pieces. A fixed chunk size leaves most of the
    # machine idle on smaller target sets -- 5,000 entities at 512 per chunk is 10 chunks across
    # 32 workers, so two thirds of the cores do nothing -- and it also load-balances badly, since
    # ego-subgraph cost is heavily right-tailed and one chunk can hold several of the worst cases.
    worker_count = os.cpu_count() or 1 if n_jobs in (-1, None) else max(n_jobs, 1)
    chunk_size = max(1, min(chunk_size, ceil(len(targets) / (4 * worker_count))))
    chunks = [targets[i: i + chunk_size] for i in range(0, len(targets), chunk_size)]
    logger.info(
        "extracting features for %d entities in %d chunks (view=%s radius=%d filtration=%s)",
        len(targets), len(chunks), config.view, config.radius, config.filtration,
    )

    started = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, verbose=5, batch_size=1)(
        delayed(_extract_chunk)(
            store.node_ids, store.indptr, store.indices, store.weight, store.timestep,
            store.out_degree, store.in_degree, store.node_timestep, store.name, chunk, config,
        )
        for chunk in chunks
    )
    wall = time.perf_counter() - started

    topo = np.vstack([r[0] for r in results])
    classical = np.vstack([r[1] for r in results]) if config.include_classical else None
    matrices = FeatureMatrices(
        entity_ids=targets,
        topological=topo,
        topological_names=results[0][7],
        classical=classical,
        classical_names=results[0][8],
        n_nodes=np.concatenate([r[2] for r in results]),
        n_edges=np.concatenate([r[3] for r in results]),
        truncated=np.concatenate([r[4] for r in results]),
        seconds_topological=float(sum(r[5] for r in results)),
        seconds_classical=float(sum(r[6] for r in results)),
        config={
            **asdict(config), "wall_seconds": wall, "n_jobs": n_jobs,
            # Stage timings are wall-clock per call, so they are only valid on an otherwise idle
            # machine. Running this alongside another job inflated `seconds_classical` 4.5-fold on
            # identical work, which would have put a badly wrong number in the cost table. The
            # load at extraction time is recorded so a contaminated measurement is identifiable
            # after the fact rather than trusted.
            **_load_snapshot(),
        },
        failures=[f for r in results for f in r[9]],
    )
    if matrices.failures:
        kinds: dict[str, int] = {}
        for _, kind in matrices.failures:
            kinds[kind] = kinds.get(kind, 0) + 1
        logger.warning(
            "%d of %d entities failed persistence and carry zero vectors: %s",
            len(matrices.failures), len(targets), kinds,
        )
    logger.info("extraction done in %.1f s wall; %s", wall, matrices.summary())

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entity_ids": matrices.entity_ids,
            "topological": matrices.topological,
            "topological_names": np.array(matrices.topological_names),
            "n_nodes": matrices.n_nodes, "n_edges": matrices.n_edges,
            "truncated": matrices.truncated,
            "seconds_topological": matrices.seconds_topological,
            "seconds_classical": matrices.seconds_classical,
            "config": json.dumps(matrices.config, default=str),
            # Failures must survive the cache. Without them a cached run reports a refusal rate of
            # zero for a filtration that in fact declined 903 of 46,564 entities, and the selection
            # table then understates a real limitation of that filtration.
            "failures": json.dumps(matrices.failures, default=str),
        }
        if classical is not None:
            payload["classical"] = classical
            payload["classical_names"] = np.array(matrices.classical_names)
        np.savez_compressed(cache_path, **payload)
        logger.info("cached features to %s", cache_path.name)

    return matrices
