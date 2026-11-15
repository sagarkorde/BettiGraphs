"""Gate 8, RQ4b / H5: entity-level topological trajectories over time.

H5 asks whether the *trajectory* of an address's topological signature across time steps carries
information beyond a single static snapshot of it.

The cohort is not a random sample, and that bounds what this can claim
-----------------------------------------------------------------------
Only **61,487 of 822,942 addresses (7.5%)** appear in more than one time step, and only 3,589
appear in five or more. Long-lived addresses are structurally different from one-shot ones -- they
are more likely to be exchanges, services or reused wallets -- and that difference is plausibly
correlated with label. Every result here is therefore reported **on the restricted cohort, with its
own label distribution stated**, and is never generalised to all wallets.

A within-cohort static baseline is the comparison that matters
--------------------------------------------------------------
Comparing trajectory features on this cohort against static features on *all* wallets would
confound the representation with the population. The baseline is therefore the static signature of
the **same** addresses at their final observed time step, so the only thing that differs is whether
temporal structure is used.

Trajectory features
-------------------
For each address with observations at time steps t_1 < ... < t_k, and each scalar descriptor d:
mean, standard deviation, minimum, maximum, first value, last value, last minus first, the
ordinary-least-squares slope against time, and the largest single-step change. Plus the observation
count and the span, which are themselves informative and are also the obvious confound -- so the
evaluation controls for them.

Usage::

    .venv\\Scripts\\python.exe experiments\\temporal\\entity_trajectories.py
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
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.statistics import paired_comparison  # noqa: E402
from src.features.topological import betti_signature, default_grid  # noqa: E402
from src.graph_construction.graphs import build_address_graph  # noqa: E402
from src.topology.persistence import persistence_of  # noqa: E402
from src.utils.seeds import SEEDS, set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
OUT = REPO_ROOT / "results" / "processed"
TESTS = REPO_ROOT / "results" / "statistical_tests"

logger = logging.getLogger("trajectories")
WALLET_CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}

#: Scalar descriptors tracked over time. Kept small deliberately: a trajectory turns each into
#: nine features, so a large base set would produce a feature space far wider than the cohort.
TRACKED = ("beta0", "beta1_total", "h0_entropy", "h1_entropy", "h0_life_mean", "h1_life_mean",
           "n_nodes", "n_edges")


def summarise_subgraph(store, address, time_step: int, radius: int, filtration: str,
                       max_nodes: int) -> dict[str, float] | None:
    """Scalar topological descriptors of one address's ego-subgraph at one time step.

    The subgraph is restricted to edges at or before ``time_step``, so a trajectory point never
    contains information from the future.
    """
    import networkx as nx

    from src.features.topological import persistence_entropy, persistence_statistics

    subgraph = store.ego_subgraph(address, radius, max_nodes=max_nodes)
    if subgraph.number_of_edges() == 0:
        return None
    stale = [(u, v) for u, v, d in subgraph.edges(data=True)
             if d.get("timestamp", -1) > time_step]
    subgraph.remove_edges_from(stale)
    if subgraph.number_of_edges() == 0:
        return None

    result = persistence_of(subgraph, filtration, max_dim=1, complex_dim=2)
    stats0 = persistence_statistics(result, 0)
    stats1 = persistence_statistics(result, 1)
    return {
        "beta0": float(nx.number_connected_components(subgraph)),
        "beta1_total": float(stats1["h1_n_finite"] + stats1["h1_n_essential"]),
        "h0_entropy": persistence_entropy(result, 0),
        "h1_entropy": persistence_entropy(result, 1),
        "h0_life_mean": stats0["h0_life_mean"],
        "h1_life_mean": stats1["h1_life_mean"],
        "n_nodes": float(subgraph.number_of_nodes()),
        "n_edges": float(subgraph.number_of_edges()),
    }


def trajectory_features(series: pd.DataFrame) -> dict[str, float]:
    """Nine summaries per tracked descriptor, plus observation count and span."""
    series = series.sort_values("time_step")
    times = series["time_step"].to_numpy(dtype=float)
    out: dict[str, float] = {
        "n_observations": float(len(series)),
        "time_span": float(times.max() - times.min()) if len(times) > 1 else 0.0,
    }
    for name in TRACKED:
        values = series[name].to_numpy(dtype=float)
        out[f"{name}_mean"] = float(np.nanmean(values))
        out[f"{name}_std"] = float(np.nanstd(values)) if values.size > 1 else 0.0
        out[f"{name}_min"] = float(np.nanmin(values))
        out[f"{name}_max"] = float(np.nanmax(values))
        out[f"{name}_first"] = float(values[0])
        out[f"{name}_last"] = float(values[-1])
        out[f"{name}_delta"] = float(values[-1] - values[0])
        out[f"{name}_max_step"] = float(np.nanmax(np.abs(np.diff(values)))) if values.size > 1 else 0.0
        if values.size > 1 and np.ptp(times) > 0:
            out[f"{name}_slope"] = float(np.polyfit(times, values, 1)[0])
        else:
            out[f"{name}_slope"] = 0.0
    return out


def main(min_observations: int, max_addresses: int, radius: int, filtration: str,
         max_nodes: int, seed: int) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_all_seeds(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    TESTS.mkdir(parents=True, exist_ok=True)

    wallets = pd.read_csv(RAW_PP / "Actors Dataset" / "wallets_features.csv",
                          usecols=["address", "Time step"])
    classes = pd.read_csv(RAW_PP / "Actors Dataset" / "wallets_classes.csv")
    frame = wallets.merge(classes, on="address", how="inner")
    frame["label"] = frame["class"].map(WALLET_CLASS_TO_LABEL).fillna(-1).astype(np.int8)
    frame = frame[frame.label != -1]

    counts = frame.groupby("address")["Time step"].nunique()
    cohort = counts[counts >= min_observations].index
    logger.info("cohort: %d addresses with >= %d observations, of %d labelled (%.1f%%)",
                len(cohort), min_observations, counts.size, 100.0 * len(cohort) / counts.size)

    rng = np.random.default_rng(seed)
    if len(cohort) > max_addresses:
        cohort = rng.choice(cohort.to_numpy(), size=max_addresses, replace=False)
    cohort_frame = frame[frame.address.isin(set(cohort))]

    labels_by_address = cohort_frame.groupby("address")["label"].first()
    first_step = cohort_frame.groupby("address")["Time step"].min()
    logger.info("cohort label distribution: %s",
                labels_by_address.value_counts().to_dict())

    store = build_address_graph(RAW_PP)

    rows: list[dict] = []
    started = time.perf_counter()
    for index, (address, group) in enumerate(cohort_frame.groupby("address")):
        points = []
        for step in sorted(group["Time step"].unique()):
            summary = summarise_subgraph(store, address, int(step), radius, filtration, max_nodes)
            if summary is not None:
                points.append({"time_step": int(step), **summary})
        if len(points) < min_observations:
            continue
        series = pd.DataFrame(points)
        record = {"address": address, "label": int(labels_by_address[address]),
                  "first_step": int(first_step[address])}
        record.update(trajectory_features(series))
        # Static baseline: the same address's final observation, same descriptors.
        final = series.sort_values("time_step").iloc[-1]
        for name in TRACKED:
            record[f"static_{name}"] = float(final[name])
        rows.append(record)
        if (index + 1) % 250 == 0:
            logger.info("  %d/%d addresses (%.1f s)", index + 1, len(cohort),
                        time.perf_counter() - started)

    data = pd.DataFrame(rows)
    if data.empty:
        logger.error("no usable trajectories; nothing to test")
        return 1
    data.to_csv(OUT / "entity_trajectories.csv", index=False)
    logger.info("built %d trajectories in %.1f s", len(data), time.perf_counter() - started)

    # --- H5: trajectory features against a within-cohort static baseline ---
    split_mask_train = data["first_step"] <= sp.TRAIN_STEPS[1]
    split_mask_test = data["first_step"] >= sp.TEST_STEPS[0]
    # Adequacy is judged on POSITIVES, not rows. An earlier version required 20 test rows and
    # accepted a split holding 37 rows with exactly ONE illicit address; average precision on a
    # single positive is not a measurement, and the run reported a confident verdict from it. The
    # multi-observation cohort skews early -- median first appearance at step 9 -- so a boundary at
    # step 42 leaves almost nothing behind it.
    MIN_TEST_POSITIVES = 20
    n_test_positive = int((data.loc[split_mask_test, "label"] == 1).sum())
    temporal_is_adequate = bool(
        split_mask_train.sum() >= 50
        and n_test_positive >= MIN_TEST_POSITIVES
        and data.loc[split_mask_test, "label"].nunique() >= 2
    )

    if not temporal_is_adequate:
        logger.warning(
            "temporal split is not testable on this cohort: %d test rows holding %d illicit "
            "addresses, against a %d-positive minimum. Falling back to a stratified random split, "
            "which is NOT a temporal evaluation and is reported as exploratory.",
            int(split_mask_test.sum()), n_test_positive, MIN_TEST_POSITIVES,
        )
        from sklearn.model_selection import train_test_split

        train_index, test_index = train_test_split(
            np.arange(len(data)), test_size=0.3, random_state=seed,
            stratify=data["label"].to_numpy(),
        )
        split_type = "stratified_random_on_cohort_EXPLORATORY"
    else:
        train_index = np.flatnonzero(split_mask_train.to_numpy())
        test_index = np.flatnonzero(split_mask_test.to_numpy())
        split_type = "temporal_by_first_step"

    static_columns = [c for c in data.columns if c.startswith("static_")]
    trajectory_columns = [
        c for c in data.columns
        if c not in ("address", "label", "first_step") and not c.startswith("static_")
    ]
    # Observation count and span are the obvious confound: a longer-lived address is different
    # for reasons that have nothing to do with topology. They go in BOTH feature sets so that the
    # comparison isolates the trajectory shape.
    confounds = ["n_observations", "time_span"]
    feature_sets = {
        "static": static_columns + confounds,
        "trajectory": trajectory_columns,
        "static+trajectory": static_columns + trajectory_columns,
    }

    y = data["label"].to_numpy()
    results: list[dict] = []
    scores_by_set: dict[str, list[float]] = {}
    for name, columns in feature_sets.items():
        X = data[columns].to_numpy(dtype=np.float64)
        per_seed = []
        for run_seed in SEEDS:
            set_all_seeds(run_seed)
            from lightgbm import LGBMClassifier
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median").fit(X[train_index])
            ratio = float((y[train_index] == 0).sum()) / max(float((y[train_index] == 1).sum()), 1.0)
            model = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                   scale_pos_weight=ratio, random_state=run_seed,
                                   n_jobs=-1, verbose=-1)
            model.fit(imputer.transform(X[train_index]), y[train_index])
            probability = model.predict_proba(imputer.transform(X[test_index]))[:, 1]
            metrics = compute_metrics(y[test_index], probability)
            per_seed.append(metrics.pr_auc)
            results.append({"feature_set": name, "seed": run_seed, "n_features": len(columns),
                            **metrics.as_dict()})
        scores_by_set[name] = per_seed
        logger.info("%-18s PR-AUC %.4f +- %.4f (%d features)",
                    name, float(np.mean(per_seed)), float(np.std(per_seed)), len(columns))

    comparison = paired_comparison(
        np.array(scores_by_set["trajectory"]), np.array(scores_by_set["static"]),
        name_a="trajectory", name_b="static",
    )
    union = paired_comparison(
        np.array(scores_by_set["static+trajectory"]), np.array(scores_by_set["static"]),
        name_a="static+trajectory", name_b="static",
    )

    payload = {
        "cohort": {
            "min_observations": min_observations,
            "n_addresses": int(len(data)),
            "label_distribution": {str(k): int(v) for k, v in
                                   pd.Series(y).value_counts().items()},
            "share_of_all_labelled_wallets": float(len(cohort) / counts.size),
            "bias_note": (
                "Addresses observed in more than one time step are 7.5% of all addresses and are "
                "structurally unlike one-shot addresses. No result here generalises beyond this "
                "cohort."
            ),
        },
        "split": {
            "type": split_type,
            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),
            "temporal_split_adequate": bool(temporal_is_adequate),
            "temporal_test_positives": n_test_positive,
            "min_test_positives_required": MIN_TEST_POSITIVES,
            "note": (
                "A stratified random split on the cohort is not a temporal evaluation. Where it "
                "was used, the temporal split held too few positives to measure anything, and the "
                "result is exploratory rather than a test of H5."
            ),
        },
        "h5_trajectory_vs_static": comparison.as_dict(),
        "union_vs_static": union.as_dict(),
        # H5 can only be answered under the pre-registered temporal split. Where that split is
        # not testable, the verdict is None -- neither supported nor rejected -- rather than a
        # rejection inferred from an exploratory random split.
        "h5_supported": (bool(comparison.significant and comparison.mean_difference > 0)
                         if temporal_is_adequate else None),
        "h5_verdict": ("supported" if (temporal_is_adequate and comparison.significant
                                       and comparison.mean_difference > 0)
                       else "rejected" if temporal_is_adequate
                       else "not testable on this cohort under a temporal split"),
        "per_run": results,
    }
    (TESTS / "h5_entity_trajectories.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\n=== H5: entity-level trajectories, cohort of {len(data)} addresses ===")
    print(f"split: {split_type} ({len(train_index)} train / {len(test_index)} test)")
    for name, scores in scores_by_set.items():
        print(f"  {name:18s} PR-AUC {np.mean(scores):.4f} +- {np.std(scores):.4f}")
    print(f"\n  {comparison.verdict()}")
    print(f"  {union.verdict()}")
    print(f"\nH5 verdict: {payload['h5_verdict']}")
    if not temporal_is_adequate:
        print(f"  The temporal split held {n_test_positive} illicit of "
              f"{int(split_mask_test.sum())} test addresses, against a "
              f"{MIN_TEST_POSITIVES}-positive minimum. The figures above come from an exploratory "
              f"random split and do not test H5.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-observations", type=int, default=3, dest="min_observations")
    parser.add_argument("--max-addresses", type=int, default=4000, dest="max_addresses")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--max-nodes", type=int, default=5000, dest="max_nodes")
    parser.add_argument("--seed", type=int, default=0)
    raise SystemExit(main(**vars(parser.parse_args())))
