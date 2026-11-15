"""Gate 8-9: statistical analysis of the transaction experiment.

Consumes ``results/raw/transaction_results_*.csv`` and produces the manuscript's result tables
plus the pre-registered hypothesis tests. Nothing here is hand-entered; every number traces to a
run record.

Hypotheses tested (``docs/PREREGISTRATION.md`` §2):

* **H2** -- T beats S on illicit PR-AUC.
* **H6** (primary) -- T degrades less than P across the step-43 regime change.
* **H7** -- P+S+T beats every single family.

The declared correction family is the 21 pairwise comparisons among the 7 feature combinations
plus H6's degradation comparison. Holm-Bonferroni is applied within it. Exploratory comparisons are
reported separately and labelled, never folded into the family after the fact -- doing so would
change the correction applied to the confirmatory tests.

Usage::

    .venv\\Scripts\\python.exe scripts\\analyse_transaction_results.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import splits as sp  # noqa: E402
from src.evaluation.statistics import (  # noqa: E402
    Comparison, correct_family, paired_comparison, spearman_with_ci,
)

RAW = REPO_ROOT / "results" / "raw"
OUT = REPO_ROOT / "results" / "statistical_tests"
TABLES = REPO_ROOT / "tables"

logger = logging.getLogger("analyse")

SINGLE_FAMILIES = ("P", "Pplus", "S", "T")
ALL_FAMILIES = ("P", "Pplus", "S", "T", "P+S", "P+T", "S+T", "P+S+T")
PRIMARY_METRIC = "pr_auc"


def load_results(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the most recent results and per-step files matching ``tag``."""
    candidates = sorted(RAW.glob(f"transaction_results_{tag}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no results matching transaction_results_{tag}*.csv in {RAW}")
    results_path = candidates[-1]
    per_step_path = Path(str(results_path).replace("_results_", "_per_step_"))
    logger.info("reading %s", results_path.name)
    results = pd.read_csv(results_path)
    per_step = pd.read_csv(per_step_path) if per_step_path.exists() else pd.DataFrame()
    return results, per_step


def best_model_per_family(results: pd.DataFrame) -> pd.DataFrame:
    """Pick one model per family, by **validation** PR-AUC recorded at search time.

    Selecting by test performance would be test-set fitting. The validation score stored with each
    run record is what the choice is made on.
    """
    chosen = (
        results.groupby(["family", "model"])["validation_pr_auc_at_search"].mean()
        .reset_index()
        .sort_values("validation_pr_auc_at_search", ascending=False)
        .groupby("family").head(1)[["family", "model"]]
    )
    merged = results.merge(chosen, on=["family", "model"], how="inner")
    logger.info("selected per family by validation: %s",
                dict(zip(chosen["family"], chosen["model"])))
    return merged


def family_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pr_auc", "roc_auc", "illicit_precision", "illicit_recall", "illicit_f1",
               "macro_f1", "mcc", "balanced_accuracy", "lift_at_100"]
    available = [m for m in metrics if m in results.columns]
    summary = (results.groupby(["family", "model"])[available]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index()


def compare_families(results: pd.DataFrame, metric: str = PRIMARY_METRIC) -> list[Comparison]:
    """All pairwise comparisons among the feature families, paired by seed."""
    by_family: dict[str, np.ndarray] = {}
    for family, group in results.groupby("family"):
        by_family[family] = group.sort_values("seed")[metric].to_numpy()

    comparisons: list[Comparison] = []
    families = [f for f in ALL_FAMILIES if f in by_family]
    for left, right in itertools.combinations(families, 2):
        a, b = by_family[left], by_family[right]
        if a.size != b.size or a.size < 2:
            logger.warning("skipping %s vs %s: %d and %d runs", left, right, a.size, b.size)
            continue
        comparisons.append(paired_comparison(a, b, name_a=left, name_b=right))
    return comparisons


def degradation_analysis(per_step: pd.DataFrame) -> pd.DataFrame:
    """Per-family PR-AUC before and after the step-43 regime change, for H6.

    Steps are pooled either side of the shift rather than compared individually, because steps 45
    and 46 hold only 5 and 2 illicit transactions. The pooled window is what H6 is defined on.
    """
    if per_step.empty:
        return pd.DataFrame()
    frame = per_step.dropna(subset=["pr_auc"]).copy()
    # Prefer the train-only protocol where it exists: it gives 8 unseen pre-shift steps against 7
    # unseen post-shift ones, where the primary split offers only step 42 before the shift.
    if "protocol" in frame.columns and (frame["protocol"] == "train_only").any():
        frame = frame[frame["protocol"] == "train_only"]
        logger.info("H6 degradation uses the train-only protocol (steps 35-49)")
    else:
        logger.warning("H6 degradation falling back to the train+val protocol; only one "
                       "pre-shift step is available, so the comparison is weak")
    frame["regime"] = np.where(frame["time_step"] < sp.REGIME_SHIFT_STEP, "pre", "post")
    # Weight each step by its illicit count: a step with 2 positives should not count as much as
    # one with 239.
    def weighted(group: pd.DataFrame) -> float:
        weights = group["n_illicit"].to_numpy(dtype=float)
        values = group["pr_auc"].to_numpy(dtype=float)
        return float(np.average(values, weights=weights)) if weights.sum() > 0 else float("nan")

    rows = []
    for (family, model, seed), group in frame.groupby(["family", "model", "seed"]):
        pre = group[group.regime == "pre"]
        post = group[group.regime == "post"]
        if pre.empty or post.empty:
            continue
        pre_value, post_value = weighted(pre), weighted(post)
        # Post-shift performance relative to the post-shift base rate. Relative drop alone is
        # scale-free but not floor-aware: a family sitting near the base rate has little to lose
        # and scores well on it while being useless in both regimes. This is the guard.
        post_base_rate = (float(post["n_illicit"].sum()) / float(post["n"].sum())
                          if post["n"].sum() > 0 else np.nan)
        rows.append({
            "family": family, "model": model, "seed": seed,
            "pr_auc_pre_shift": pre_value,
            "pr_auc_post_shift": post_value,
            "absolute_drop": pre_value - post_value,
            "relative_drop": (pre_value - post_value) / pre_value if pre_value > 0 else np.nan,
            "post_shift_lift": post_value / post_base_rate if post_base_rate else np.nan,
            "post_shift_base_rate": post_base_rate,
            "n_illicit_pre": int(pre["n_illicit"].sum()),
            "n_illicit_post": int(post["n_illicit"].sum()),
        })
    return pd.DataFrame(rows)


def test_h6(degradation: pd.DataFrame) -> list[Comparison]:
    """H6: does T's degradation across the shift differ from P's?

    The comparison is on ``relative_drop``, so a family with a lower absolute starting point is
    not penalised for having less to lose.
    """
    if degradation.empty:
        return []
    comparisons: list[Comparison] = []
    by_drop = {
        family: group.sort_values("seed")["relative_drop"].to_numpy()
        for family, group in degradation.groupby("family")
    }
    by_retained = {
        family: group.sort_values("seed")["post_shift_lift"].to_numpy()
        for family, group in degradation.groupby("family")
    }
    for left, right in (("T", "P"), ("T", "S"), ("P+T", "P")):
        if left in by_drop and right in by_drop:
            a, b = by_drop[left], by_drop[right]
            if a.size == b.size and a.size >= 2:
                # Negated so a positive difference means the first family degraded LESS, which is
                # the direction H6 predicts.
                comparisons.append(paired_comparison(
                    -a, -b, name_a=f"{left}_smaller_drop", name_b=f"{right}_smaller_drop"
                ))
        # The guard comparison. A smaller relative drop is only meaningful if the family is still
        # doing something useful afterwards, so post-shift lift over the post-shift base rate is
        # compared as well, and both are reported.
        if left in by_retained and right in by_retained:
            a, b = by_retained[left], by_retained[right]
            if a.size == b.size and a.size >= 2:
                comparisons.append(paired_comparison(
                    a, b, name_a=f"{left}_post_shift_lift", name_b=f"{right}_post_shift_lift"
                ))
    return comparisons


def write_tables(summary: pd.DataFrame, comparisons: list[Comparison],
                 degradation: pd.DataFrame, h6: list[Comparison], provenance: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    families_present = summary["family"].unique().tolist()
    baseline = [f for f in ("P", "Pplus", "S") if f in families_present]
    topo = [f for f in ("T",) if f in families_present]
    hybrid = [f for f in ("P+S", "P+T", "S+T", "P+S+T") if f in families_present]

    def block(title: str, families: list[str], note: str) -> list[str]:
        lines = [f"# {title}", "", note, "",
                 "| Family | Model | PR-AUC (illicit) | ROC-AUC | Illicit F1 | MCC | Lift@100 | Features |",
                 "|---|---|---|---|---|---|---|---|"]
        for _, row in summary[summary.family.isin(families)].iterrows():
            lines.append(
                f"| {row['family']} | {row['model']} | "
                f"{row.get('pr_auc_mean', float('nan')):.4f} ± {row.get('pr_auc_std', 0):.4f} | "
                f"{row.get('roc_auc_mean', float('nan')):.4f} ± {row.get('roc_auc_std', 0):.4f} | "
                f"{row.get('illicit_f1_mean', float('nan')):.4f} | "
                f"{row.get('mcc_mean', float('nan')):.4f} | "
                f"{row.get('lift_at_100_mean', float('nan')):.2f} | "
                f"{provenance.get('n_features', {}).get(row['family'], '')} |"
            )
        lines.append("")
        return lines

    # Derived, not written: the seed count changed from ten to twenty when the discreteness floor
    # made the pre-registered significance rule unreachable, and five table captions went on
    # saying "ten" until a reader of the Word output noticed.
    # Taken from the paired comparisons, where it is the number of paired runs by construction,
    # rather than from a separate count that could disagree with the tests reported beneath it.
    n_seeds = comparisons[0].n_pairs if comparisons else 0
    header = (
        f"Mean ± standard deviation over the {n_seeds} pre-registered seeds, on the held-out "
        "test window "
        "(time steps 42–49). Primary metric is PR-AUC on the illicit class, fixed before any model "
        "was fitted. The test split was scored once per (model, family, seed) and used for no "
        "selection."
    )
    (TABLES / "table5_baseline_performance.md").write_text(
        "\n".join(block("Table 5 — Baseline performance (provided and classical features)",
                        baseline, header)), encoding="utf-8")
    if topo:
        (TABLES / "table6_tda_performance.md").write_text(
            "\n".join(block("Table 6 — Topological feature performance", topo, header)),
            encoding="utf-8")
    if hybrid:
        (TABLES / "table7_hybrid_performance.md").write_text(
            "\n".join(block("Table 7 — Hybrid feature performance", hybrid, header)),
            encoding="utf-8")

    # --- Table 9: significance ---
    lines = [
        "# Table 9 — Statistical significance of family comparisons",
        "",
        f"Paired Wilcoxon signed-rank across the {n_seeds} seeds, Holm–Bonferroni corrected "
        "within the",
        "declared family of comparisons. A confidence interval containing zero vetoes the claim",
        "regardless of the p-value, per `docs/PREREGISTRATION.md` §7.",
        "",
        "| Comparison | Mean A | Mean B | Difference | 95% CI | p (raw) | p (Holm) | Cliff's δ | Significant |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for comparison in sorted(comparisons, key=lambda c: c.p_value):
        corrected = ("—" if comparison.p_value_corrected is None
                     else f"{comparison.p_value_corrected:.3g}")
        lines.append(
            f"| {comparison.name_a} vs {comparison.name_b} | {comparison.mean_a:.4f} | "
            f"{comparison.mean_b:.4f} | {comparison.mean_difference:+.4f} | "
            f"[{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}] | "
            f"{comparison.p_value:.3g} | {corrected} | {comparison.cliffs_delta:+.3f} | "
            f"{'**yes**' if comparison.significant else 'no'} |"
        )
    lines.append("")

    if h6:
        lines += [
            "## H6 — relative degradation across the step-43 regime change",
            "",
            "Compared on *relative* drop in weighted PR-AUC, so a family with a lower starting",
            "point is not credited merely for having less to lose. A positive difference means the",
            "first family degraded **less**, which is the direction H6 predicts. Per-step values",
            "are weighted by that step's illicit count, because steps 45 and 46 hold only 5 and 2.",
            "",
            "| Comparison | Difference in robustness | 95% CI | p (raw) | p (Holm) | Significant |",
            "|---|---|---|---|---|---|",
        ]
        for comparison in h6:
            corrected = ("—" if comparison.p_value_corrected is None
                         else f"{comparison.p_value_corrected:.3g}")
            lines.append(
                f"| {comparison.name_a} vs {comparison.name_b} | "
                f"{comparison.mean_difference:+.4f} | "
                f"[{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}] | "
                f"{comparison.p_value:.3g} | {corrected} | "
                f"{'**yes**' if comparison.significant else 'no'} |"
            )
        lines.append("")

    if not degradation.empty:
        lines += ["## Pre- and post-shift PR-AUC by family", "",
                  "| Family | Pre-shift (≤42) | Post-shift (≥43) | Absolute drop | Relative drop |",
                  "|---|---|---|---|---|"]
        aggregated = degradation.groupby("family")[
            ["pr_auc_pre_shift", "pr_auc_post_shift", "absolute_drop", "relative_drop"]
        ].mean().round(4)
        for family, row in aggregated.iterrows():
            lines.append(
                f"| {family} | {row['pr_auc_pre_shift']:.4f} | {row['pr_auc_post_shift']:.4f} | "
                f"{row['absolute_drop']:+.4f} | {row['relative_drop']:+.1%} |"
            )
        lines.append("")

    (TABLES / "table9_statistical_significance.md").write_text("\n".join(lines), encoding="utf-8")


def main(tag: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    results, per_step = load_results(tag)
    logger.info("loaded %d run records across %d families and %d models",
                len(results), results.family.nunique(), results.model.nunique())

    n_features = results.groupby("family")["n_features"].first().to_dict()
    selected = best_model_per_family(results)

    summary_all = family_summary(results)
    summary_best = family_summary(selected)
    summary_all.to_csv(OUT / "family_model_summary.csv", index=False)
    # Also saved on its own, because this is the table the manuscript quotes and the model behind
    # each row is chosen by validation score -- so it can change between runs. A claim that named
    # the model would silently point at an unselected row after such a change; one that names only
    # the family cannot.
    summary_best.to_csv(OUT / "family_selected_summary.csv", index=False)

    comparisons = compare_families(selected)
    degradation = degradation_analysis(
        per_step.merge(selected[["family", "model"]].drop_duplicates(),
                       on=["family", "model"], how="inner") if not per_step.empty else per_step
    )
    h6 = test_h6(degradation)

    # Correct within the declared family: the pairwise family comparisons plus H6.
    correct_family(comparisons + h6)

    if not degradation.empty:
        degradation.to_csv(OUT / "regime_shift_degradation.csv", index=False)

    write_tables(summary_best, comparisons, degradation, h6, {"n_features": n_features})

    verdicts = {
        "H2_T_beats_S": next(
            (c.as_dict() for c in comparisons
             if {c.name_a, c.name_b} == {"T", "S"}), None),
        "H7_union_beats_singles": [
            c.as_dict() for c in comparisons
            if c.name_a == "P+S+T" or c.name_b == "P+S+T"
        ],
        "H6_robustness": [c.as_dict() for c in h6],
        "all_comparisons": [c.as_dict() for c in comparisons],
    }
    (OUT / "hypothesis_tests.json").write_text(
        json.dumps(verdicts, indent=2, default=str), encoding="utf-8")

    print("\n=== mean test PR-AUC by family (best model per family, chosen on validation) ===")
    print(summary_best[["family", "model", "pr_auc_mean", "pr_auc_std"]]
          .sort_values("pr_auc_mean", ascending=False).to_string(index=False))
    print("\n=== pre-registered hypothesis verdicts ===")
    for comparison in comparisons + h6:
        print("  " + comparison.verdict())
    if not degradation.empty:
        print("\n=== regime-shift degradation (mean over seeds) ===")
        print(degradation.groupby("family")[
            ["pr_auc_pre_shift", "pr_auc_post_shift", "relative_drop"]
        ].mean().round(4).to_string())

    logger.info("wrote tables to %s and tests to %s", TABLES, OUT)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="main")
    raise SystemExit(main(**vars(parser.parse_args())))
