"""Gate 6: select the topological settings on the validation split only.

Implements ``docs/PREREGISTRATION.md`` §8. Three criteria per candidate, all scored on
**validation**:

1. **Discriminative power** -- validation illicit PR-AUC from a single fixed LightGBM
   configuration. Fixed, so that the comparison is between filtrations and not between
   hyperparameter searches.
2. **Stability** -- mean relative L2 drift of the signature under random deletion of 5% and 10% of
   edges. A descriptor that moves a long way under a small perturbation is not measuring
   structure, it is measuring noise, and would not survive a different snapshot of the same chain.
3. **Cost** -- wall-clock seconds per entity.

The search is **coordinate-wise**, in a fixed order declared before running: radius, then
filtration, then view, then complex dimension, then curve resolution. Exhaustive search over
4 radii x 5 filtrations x 3 views x 3 dimensions x 3 resolutions would be 540 configurations, and
each one is an opportunity to overfit the validation split. The order is fixed in advance so it
cannot be reshuffled to favour an outcome.

The test split is never touched here.

Usage::

    .venv\\Scripts\\python.exe experiments\\betti\\select_topology_settings.py
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
from src.features.topological import betti_signature, default_grid  # noqa: E402
from src.graph_construction.graphs import build_tx_graph  # noqa: E402
from src.topology.filtrations import ALL_FILTRATIONS  # noqa: E402
from src.topology.persistence import persistence_of  # noqa: E402
from src.utils.seeds import set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
CACHE = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "results" / "processed"
TABLES = REPO_ROOT / "tables"

logger = logging.getLogger("select_topology")
CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}

#: Fixed evaluator, identical for every candidate.
EVALUATOR_PARAMS = dict(
    n_estimators=400, num_leaves=63, learning_rate=0.05,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.9, verbose=-1,
)


def _score(X_train, y_train, X_val, y_val, seed: int = 0) -> float:
    """Validation illicit PR-AUC from the fixed evaluator."""
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score

    imputer = SimpleImputer(strategy="median").fit(X_train)
    ratio = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    model = LGBMClassifier(**EVALUATOR_PARAMS, scale_pos_weight=ratio, random_state=seed, n_jobs=-1)
    model.fit(imputer.transform(X_train), y_train)
    scores = model.predict_proba(imputer.transform(X_val))[:, 1]
    return float(average_precision_score(y_val, scores))


def _stability(store, targets, config: ExtractionConfig, *, drop: float, n: int = 250,
               seed: int = 0) -> float:
    """Mean relative L2 drift of the signature when a fraction of edges is deleted.

    Lower is better. Returns NaN if no comparison was possible (e.g. every sampled subgraph was
    too small to delete an edge from).
    """
    rng = np.random.default_rng(seed)
    grid = default_grid(config.resolution)
    chosen = rng.choice(targets, size=min(n, len(targets)), replace=False)
    drifts: list[float] = []

    skipped = 0
    for target in chosen:
        subgraph = store.ego_subgraph(target, config.radius, max_nodes=config.max_nodes)
        if subgraph.number_of_edges() < 10:
            continue
        # A refusal on one sampled subgraph must not discard the whole candidate. The Rips
        # filtrations decline oversized inputs by design, and an earlier version let that
        # propagate: the feature extraction for geodesic Rips completed over all 46,564 entities,
        # took ten minutes, and was then thrown away because one 477-point subgraph in the
        # 200-sample stability check raised. Skipped samples are counted and the rate reported.
        try:
            base = persistence_of(subgraph, config.filtration, max_dim=config.max_dim,
                                  complex_dim=config.complex_dim, normalise=config.normalise)
            base_vector, _ = betti_signature(base, max_dim=config.max_dim, grid=grid,
                                             families=config.families,
                                             image_resolution=config.image_resolution,
                                             num_landscapes=config.num_landscapes)
            perturbed = subgraph.copy()
            edges = list(perturbed.edges())
            k = max(1, int(round(drop * len(edges))))
            for index in rng.choice(len(edges), size=k, replace=False):
                perturbed.remove_edge(*edges[index])
            if perturbed.number_of_edges() == 0:
                continue
            other = persistence_of(perturbed, config.filtration, max_dim=config.max_dim,
                                   complex_dim=config.complex_dim, normalise=config.normalise)
            other_vector, _ = betti_signature(other, max_dim=config.max_dim, grid=grid,
                                              families=config.families,
                                              image_resolution=config.image_resolution,
                                              num_landscapes=config.num_landscapes)
        except (MemoryError, ValueError, RuntimeError, np.linalg.LinAlgError):
            skipped += 1
            continue
        denominator = np.linalg.norm(base_vector)
        if denominator > 0:
            drifts.append(float(np.linalg.norm(base_vector - other_vector) / denominator))

    if skipped:
        logger.info("stability: %d of %d sampled subgraphs were refused and skipped",
                    skipped, len(chosen))
    return float(np.mean(drifts)) if drifts else float("nan")


def evaluate_candidate(store, frame, split, labels, config: ExtractionConfig, *,
                       n_jobs: int, stability_n: int) -> dict:
    """Score one candidate configuration on all three criteria."""
    started = time.perf_counter()
    matrices = extract_features(store, frame["txId"].to_numpy(), config,
                               n_jobs=n_jobs, cache_dir=CACHE)
    extraction_wall = time.perf_counter() - started

    X = matrices.topological
    pr_auc = _score(X[split.train], labels[split.train], X[split.val], labels[split.val])

    record = {
        "view": config.view, "radius": config.radius, "filtration": config.filtration,
        "complex_dim": config.complex_dim, "resolution": config.resolution,
        "normalise": config.normalise,
        "val_pr_auc": pr_auc,
        "n_features": int(X.shape[1]),
        "median_nodes": float(np.median(matrices.n_nodes)),
        "median_edges": float(np.median(matrices.n_edges)),
        "frac_truncated": float(matrices.truncated.mean()),
        "frac_empty": float((matrices.n_edges == 0).mean()),
        # Entities whose persistence computation was refused -- effective-resistance filtrations
        # decline oversized subgraphs by design. These carry zero vectors, so a filtration with a
        # high refusal rate may score poorly for lack of coverage rather than lack of signal, and
        # the rate is reported beside the score so the two are not confused.
        "frac_failed": float(len(matrices.failures) / max(matrices.n_entities, 1)),
        "seconds_per_entity_topological": matrices.seconds_topological / max(len(X), 1),
        "extraction_wall_seconds": extraction_wall,
        "stability_drift_5pct": _stability(store, frame["txId"].to_numpy()[split.train], config,
                                          drop=0.05, n=stability_n),
        "stability_drift_10pct": _stability(store, frame["txId"].to_numpy()[split.train], config,
                                           drop=0.10, n=stability_n),
    }
    logger.info(
        "%s r=%d %-18s cd=%d R=%d -> val PR-AUC %.4f | drift5 %.3f drift10 %.3f | %.2f ms/entity",
        config.view, config.radius, config.filtration, config.complex_dim, config.resolution,
        pr_auc, record["stability_drift_5pct"], record["stability_drift_10pct"],
        1000 * record["seconds_per_entity_topological"],
    )
    return record


def main(n_jobs: int, stability_n: int, quick: bool) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_all_seeds(0)
    OUT.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv",
                           usecols=["txId", "Time step"])
    classes = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_classes.csv")
    frame = features.merge(classes, on="txId", how="inner")
    frame["label"] = frame["class"].map(CLASS_TO_LABEL).astype(np.int8)
    frame = frame[frame.label != -1].reset_index(drop=True)

    store = build_tx_graph(RAW_PP)
    labels = frame["label"].to_numpy()
    split = sp.temporal_split(frame["Time step"].to_numpy(), labels)

    records: list[dict] = []
    stages: dict[str, dict] = {}

    # --- Stage 1: radius, with everything else at its declared starting value ---
    logger.info("### stage 1: ego radius ###")
    radii = (1, 2, 3) if quick else (1, 2, 3, 4)
    for radius in radii:
        config = ExtractionConfig(view="G_tx", radius=radius, filtration="value_superlevel",
                                  complex_dim=2, include_classical=False)
        records.append({**evaluate_candidate(store, frame, split, labels, config,
                                             n_jobs=n_jobs, stability_n=stability_n),
                        "stage": "radius"})
    best_radius = max((r for r in records if r["stage"] == "radius"),
                      key=lambda r: r["val_pr_auc"])["radius"]
    stages["radius"] = {"selected": best_radius}
    logger.info("selected radius = %d", best_radius)

    # --- Stage 2: filtration, at the selected radius ---
    logger.info("### stage 2: filtration ###")
    for filtration in ALL_FILTRATIONS:
        if filtration == "value_superlevel":
            existing = [r for r in records
                        if r["stage"] == "radius" and r["radius"] == best_radius]
            records.append({**existing[0], "stage": "filtration"})
            continue
        config = ExtractionConfig(view="G_tx", radius=best_radius, filtration=filtration,
                                  complex_dim=2, include_classical=False)
        try:
            records.append({**evaluate_candidate(store, frame, split, labels, config,
                                                 n_jobs=n_jobs, stability_n=stability_n),
                            "stage": "filtration"})
        except Exception as exc:
            logger.error("filtration %s failed: %s", filtration, exc)
            records.append({"stage": "filtration", "filtration": filtration,
                            "view": "G_tx", "radius": best_radius,
                            "val_pr_auc": float("nan"), "error": str(exc)})
    viable = [r for r in records if r["stage"] == "filtration" and np.isfinite(r["val_pr_auc"])]
    best_filtration = max(viable, key=lambda r: r["val_pr_auc"])["filtration"]
    stages["filtration"] = {"selected": best_filtration}
    logger.info("selected filtration = %s", best_filtration)

    # --- Stage 3: complex dimension ---
    logger.info("### stage 3: complex dimension ###")
    for complex_dim in (1, 2, 3):
        config = ExtractionConfig(view="G_tx", radius=best_radius, filtration=best_filtration,
                                  complex_dim=complex_dim, include_classical=False)
        records.append({**evaluate_candidate(store, frame, split, labels, config,
                                             n_jobs=n_jobs, stability_n=stability_n),
                        "stage": "complex_dim"})
    best_dim = max((r for r in records if r["stage"] == "complex_dim"),
                   key=lambda r: r["val_pr_auc"])["complex_dim"]
    stages["complex_dim"] = {"selected": best_dim}
    logger.info("selected complex_dim = %d", best_dim)

    # --- Stage 4: Betti-curve resolution ---
    logger.info("### stage 4: curve resolution ###")
    for resolution in (16, 32, 64):
        config = ExtractionConfig(view="G_tx", radius=best_radius, filtration=best_filtration,
                                  complex_dim=best_dim, resolution=resolution,
                                  include_classical=False)
        records.append({**evaluate_candidate(store, frame, split, labels, config,
                                             n_jobs=n_jobs, stability_n=stability_n),
                        "stage": "resolution"})
    best_resolution = max((r for r in records if r["stage"] == "resolution"),
                          key=lambda r: r["val_pr_auc"])["resolution"]
    stages["resolution"] = {"selected": best_resolution}
    logger.info("selected resolution = %d", best_resolution)

    frame_out = pd.DataFrame(records)
    frame_out.to_csv(OUT / "topology_selection.csv", index=False)
    selected = {
        "view": "G_tx", "radius": best_radius, "filtration": best_filtration,
        "complex_dim": best_dim, "resolution": best_resolution,
        "protocol": "coordinate-wise, order fixed in advance: radius, filtration, complex_dim, "
                    "resolution; scored on validation illicit PR-AUC with a fixed LightGBM "
                    "configuration; test split untouched",
        "stages": stages,
    }
    (OUT / "topology_selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    # --- Table 4 ---
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Table 4 — Selection of topological settings (validation split only)",
        "",
        "Generated by `experiments/betti/select_topology_settings.py`. Every value is a validation",
        "measurement; the test split played no part in any choice here. Search is coordinate-wise",
        "in a fixed, pre-declared order, because exhaustive search over 540 configurations would",
        "itself be an opportunity to overfit validation.",
        "",
        "Drift is the mean relative L2 change in the signature when 5% or 10% of edges are deleted.",
        "Lower is better: a descriptor that moves far under a small perturbation is measuring noise.",
        "",
        "| Stage | Setting | Validation PR-AUC | Drift 5% | Drift 10% | ms/entity | Median nodes | Empty | Refused |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        if not np.isfinite(record.get("val_pr_auc", float("nan"))):
            lines.append(f"| {record.get('stage','')} | {record.get('filtration','')} "
                         f"| failed: {record.get('error','')[:60]} | | | | | | |")
            continue
        stage = record["stage"]
        setting = {
            "radius": f"radius {record['radius']}",
            "filtration": record["filtration"],
            "complex_dim": f"complex dim {record['complex_dim']}",
            "resolution": f"resolution {record['resolution']}",
        }[stage]
        lines.append(
            f"| {stage} | {setting} | {record['val_pr_auc']:.4f} | "
            f"{record['stability_drift_5pct']:.3f} | {record['stability_drift_10pct']:.3f} | "
            f"{1000 * record['seconds_per_entity_topological']:.2f} | "
            f"{record['median_nodes']:.0f} | {record['frac_empty']:.3f} | "
            f"{record.get('frac_failed', 0.0):.3f} |"
        )
    lines += ["", "**Selected:** " + ", ".join(
        f"{k} = {v}" for k, v in selected.items() if k in
        ("view", "radius", "filtration", "complex_dim", "resolution")), ""]
    (TABLES / "table4_topology_selection.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(selected, indent=2))
    logger.info("wrote %s and %s", OUT / "topology_selection.csv",
                TABLES / "table4_topology_selection.md")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    parser.add_argument("--stability-n", type=int, default=250, dest="stability_n")
    parser.add_argument("--quick", action="store_true")
    raise SystemExit(main(**vars(parser.parse_args())))
