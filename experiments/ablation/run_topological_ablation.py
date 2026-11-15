"""Gate 7: within-T ablation. Which topological descriptors actually contribute? Table 8.

The pre-registration commits to an ablation over the descriptor families that make up the
topological signature. This runs it by **slicing the cached feature matrix** rather than
re-extracting, so every variant sees numerically identical inputs and the comparison is about the
descriptors alone.

Variants, from ``PREREGISTRATION.md`` §6.6:

===================  ==========================================================
``beta0_only``       β₀ Betti curve
``beta1_only``       β₁ Betti curve
``betti_curves``     both curves
``stats_only``       lifetime statistics and persistence entropy
``image_only``       persistence images
``landscape_only``   persistence landscapes
``euler_only``       Euler-characteristic curve
``entropy_only``     persistence entropy alone, the cheapest possible summary
``all``              the full signature
===================  ==========================================================

Two baselines are included in the same table and this is the point of the exercise:

* ``cycle_rank_only`` -- the single classical scalar that β₁ of a 1-skeleton equals exactly.
* ``size_only`` -- ego-subgraph order and edge count, nothing else.

If a topological variant cannot beat two numbers describing how big the subgraph is, it is not
contributing topology. Earlier results in this project make that a live possibility rather than a
rhetorical flourish, so it is tested rather than assumed.

Usage::

    .venv\\Scripts\\python.exe experiments\\ablation\\run_topological_ablation.py
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
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.statistics import correct_family, paired_comparison  # noqa: E402
from src.features.pipeline import ExtractionConfig, extract_features  # noqa: E402
from src.graph_construction.graphs import build_tx_graph  # noqa: E402
from src.utils.seeds import SEEDS, set_all_seeds  # noqa: E402

RAW_PP = REPO_ROOT / "data" / "raw" / "elliptic_plus_plus"
CACHE = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "results" / "processed"
TESTS = REPO_ROOT / "results" / "statistical_tests"
TABLES = REPO_ROOT / "tables"

logger = logging.getLogger("ablation")
CLASS_TO_LABEL = {1: 1, 2: 0, 3: -1}


def variant_columns(names: list[str]) -> dict[str, list[int]]:
    """Map each ablation variant to the column indices it uses."""
    index = {name: i for i, name in enumerate(names)}

    def matching(predicate) -> list[int]:
        return [i for name, i in index.items() if predicate(name)]

    betti0 = matching(lambda n: n.startswith("bc_h0_"))
    betti1 = matching(lambda n: n.startswith("bc_h1_"))
    stats = matching(lambda n: n.startswith("h0_") or n.startswith("h1_"))
    entropy = matching(lambda n: n.endswith("_entropy"))
    image = matching(lambda n: n.startswith("pi_"))
    landscape = matching(lambda n: n.startswith("pl_"))
    euler = matching(lambda n: n.startswith("euler_"))

    return {
        "beta0_only": betti0,
        "beta1_only": betti1,
        "betti_curves": sorted(betti0 + betti1),
        "stats_only": stats,
        "entropy_only": entropy,
        "image_only": image,
        "landscape_only": landscape,
        "euler_only": euler,
        "all": sorted(set(betti0 + betti1 + stats + image + landscape + euler)),
    }


def evaluate(X: np.ndarray, y: np.ndarray, split, seeds) -> list[float]:
    """PR-AUC on the test split for each seed, with a fixed model configuration."""
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer

    X_fit = np.vstack([X[split.train], X[split.val]])
    y_fit = np.concatenate([y[split.train], y[split.val]])
    imputer = SimpleImputer(strategy="median").fit(X_fit)
    ratio = float((y_fit == 0).sum()) / max(float((y_fit == 1).sum()), 1.0)

    scores = []
    for seed in seeds:
        set_all_seeds(seed)
        model = LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05,
                               subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                               scale_pos_weight=ratio, random_state=seed, n_jobs=-1, verbose=-1)
        model.fit(imputer.transform(X_fit), y_fit)
        probability = model.predict_proba(imputer.transform(X[split.test]))[:, 1]
        scores.append(compute_metrics(y[split.test], probability).pr_auc)
    return scores


def main(radius: int, filtration: str, complex_dim: int, n_jobs: int) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_all_seeds(0)
    for directory in (OUT, TESTS, TABLES):
        directory.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_features.csv",
                           usecols=["txId", "Time step"])
    classes = pd.read_csv(RAW_PP / "Transactions Dataset" / "txs_classes.csv")
    frame = features.merge(classes, on="txId", how="inner")
    frame["label"] = frame["class"].map(CLASS_TO_LABEL).astype(np.int8)
    labelled = frame[frame.label != -1].reset_index(drop=True)

    store = build_tx_graph(RAW_PP)
    config = ExtractionConfig(view="G_tx", radius=radius, filtration=filtration,
                              complex_dim=complex_dim, max_dim=1, include_classical=True)
    matrices = extract_features(store, labelled["txId"].to_numpy(), config,
                               n_jobs=n_jobs, cache_dir=CACHE)

    y = labelled["label"].to_numpy()
    split = sp.temporal_split(labelled["Time step"].to_numpy(), y)

    variants = variant_columns(matrices.topological_names)
    classical_index = {name: i for i, name in enumerate(matrices.classical_names)}

    blocks: dict[str, np.ndarray] = {}
    for name, columns in variants.items():
        if not columns:
            logger.warning("variant %s selected no columns; skipping", name)
            continue
        blocks[name] = matrices.topological[:, columns]

    # The two baselines that make the ablation meaningful.
    blocks["cycle_rank_only"] = matrices.classical[:, [classical_index["ego_cycle_rank"]]]
    blocks["size_only"] = np.column_stack([
        matrices.n_nodes.astype(float), matrices.n_edges.astype(float)
    ])

    records: list[dict] = []
    scores_by_variant: dict[str, list[float]] = {}
    for name, block in blocks.items():
        scores = evaluate(block, y, split, SEEDS)
        scores_by_variant[name] = scores
        records.append({
            "variant": name,
            "n_features": int(block.shape[1]),
            "pr_auc_mean": float(np.mean(scores)),
            "pr_auc_std": float(np.std(scores)),
        })
        logger.info("%-16s %3d features  PR-AUC %.4f +- %.4f",
                    name, block.shape[1], float(np.mean(scores)), float(np.std(scores)))

    results = pd.DataFrame(records).sort_values("pr_auc_mean", ascending=False)
    results.to_csv(OUT / "topological_ablation.csv", index=False)

    # Every topological variant against both baselines, Holm-corrected as one family.
    comparisons = []
    for name in blocks:
        if name in ("cycle_rank_only", "size_only"):
            continue
        for baseline in ("cycle_rank_only", "size_only"):
            comparisons.append(paired_comparison(
                np.array(scores_by_variant[name]), np.array(scores_by_variant[baseline]),
                name_a=name, name_b=baseline,
            ))
    correct_family(comparisons)

    beats_size = [c for c in comparisons
                  if c.name_b == "size_only" and c.significant and c.mean_difference > 0]
    beats_cycle = [c for c in comparisons
                   if c.name_b == "cycle_rank_only" and c.significant and c.mean_difference > 0]

    lines = [
        "# Table 8 — Ablation of the topological signature",
        "",
        "Each variant slices the **same cached feature matrix**, so every row sees numerically",
        "identical inputs and the comparison is about descriptors alone. Fixed LightGBM",
        f"configuration, {comparisons[0].n_pairs if comparisons else 0} pre-registered seeds, "
        "held-out test window (steps 42–49).",
        "",
        "Two baselines are included deliberately: `cycle_rank_only` is the single classical scalar",
        "that β₁ of a 1-skeleton equals exactly, and `size_only` is just the subgraph's order and",
        "edge count. A topological variant that cannot beat two numbers describing how big the",
        "subgraph is has not contributed topology.",
        "",
        "| Variant | Features | Test PR-AUC | vs `cycle_rank_only` | vs `size_only` |",
        "|---|---|---|---|---|",
    ]
    by_pair = {(c.name_a, c.name_b): c for c in comparisons}
    for _, row in results.iterrows():
        name = row["variant"]
        cells = []
        for baseline in ("cycle_rank_only", "size_only"):
            comparison = by_pair.get((name, baseline))
            if comparison is None:
                cells.append("—")
            else:
                marker = "**+**" if (comparison.significant and comparison.mean_difference > 0) \
                    else ("**−**" if (comparison.significant and comparison.mean_difference < 0)
                          else "ns")
                cells.append(f"{comparison.mean_difference:+.4f} {marker}")
        lines.append(
            f"| `{name}` | {int(row['n_features'])} | "
            f"{row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f} | {cells[0]} | {cells[1]} |"
        )

    lines += [
        "",
        "`+` means significantly higher after Holm correction with a confidence interval",
        "excluding zero; `ns` means not significant.",
        "",
        f"**{len(beats_size)}** topological variants significantly beat subgraph size alone, and",
        f"**{len(beats_cycle)}** beat the cycle-rank scalar.",
        "",
    ]
    (TABLES / "table8_ablation.md").write_text("\n".join(lines), encoding="utf-8")

    (TESTS / "topological_ablation.json").write_text(json.dumps({
        "radius": radius, "filtration": filtration, "complex_dim": complex_dim,
        "variants": records,
        "n_beating_size_baseline": len(beats_size),
        "n_beating_cycle_rank_baseline": len(beats_cycle),
        "comparisons": [c.as_dict() for c in comparisons],
    }, indent=2, default=str), encoding="utf-8")

    print("\n=== Table 8: topological ablation ===")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nvariants beating size-only baseline : {len(beats_size)}")
    print(f"variants beating cycle-rank baseline: {len(beats_cycle)}")
    for comparison in comparisons:
        if comparison.significant:
            print("  " + comparison.verdict())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--filtration", default="value_superlevel")
    parser.add_argument("--complex-dim", type=int, default=2, dest="complex_dim")
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    raise SystemExit(main(**vars(parser.parse_args())))
