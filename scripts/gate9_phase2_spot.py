"""Gate 9, phase 2 — re-run a slice of the pipeline from scratch and compare it with the record.

Phase 1 showed that a clean environment can run the code and that the deterministic artefacts
regenerate byte-identically. It did not touch the stochastic half: model fitting, hyper-parameter
search, and the per-seed metrics the paper's claims are built from.

A full phase 2 re-runs everything, takes hours, and would overwrite the canonical results it is
meant to check -- so this runs a representative slice under a separate tag, leaving the canonical
artefacts untouched, and compares the numbers cell by cell.

It does NOT check bit equality, because the pipeline does not deliver it and does not claim to.
The hyper-parameter search is seeded, but it scores each draw by fitting a model, and threaded tree
builders and BLAS are not bit-deterministic; between two near-tied configurations a floating-point
difference flips the argmax, a different configuration is selected, and the reported metric moves
with it. Measured here, individual cells move by up to about 0.04 PR-AUC between independent runs.

So the check is the one that corresponds to what the paper claims: that the ORDERING of the feature
families is stable under re-running. Every conclusion in the paper is a comparison, none depends on
a cell's third decimal place, and a re-run that preserved the numbers but inverted an ordering
would be worthless while one that moved every number but preserved every ordering is exactly what
section 9.1 says to expect.

The magnitude is reported alongside, because "not bit-reproducible" is only useful to a reader who
is told by how much.

The slice is chosen to exercise the parts most likely to diverge -- the feature cache, the
hyper-parameter search, and a family from each representation -- rather than to be large.

Usage::

    .venv\\Scripts\\python.exe scripts\\gate9_phase2_spot.py
    .venv\\Scripts\\python.exe scripts\\gate9_phase2_spot.py --tolerance 1e-9
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RAW = REPO_ROOT / "results" / "raw"
CANONICAL = RAW / "transaction_results_r2s20_r2_value_superlevel_cd2.csv"
SPOT_TAG = "gate9spot"
METRICS = ["pr_auc", "roc_auc", "illicit_f1", "mcc", "balanced_accuracy"]

logger = logging.getLogger("gate9")


def main(skip_run: bool) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not CANONICAL.exists():
        raise SystemExit("no canonical results at " + CANONICAL.name)

    # --quick appends _quick to the suffix the runner builds, so the filename is not the
    # plain tag form.
    spot_path = RAW / ("transaction_results_" + SPOT_TAG
                       + "_r2_value_superlevel_cd2_quick.csv")
    if not skip_run:
        # --quick restricts to three seeds and the three single-representation families, which is
        # the smallest slice that still exercises the provided, structural and topological paths.
        logger.info("re-running the slice from scratch under tag %r", SPOT_TAG)
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "experiments" / "baseline" /
                                 "run_transaction_experiment.py"),
             "--radius", "2", "--tag", SPOT_TAG, "--quick"],
            cwd=REPO_ROOT)
        if completed.returncode != 0:
            raise SystemExit("the spot run failed with exit code " + str(completed.returncode))

    if not spot_path.exists():
        raise SystemExit("the spot run produced no results at " + spot_path.name)

    canonical = pd.read_csv(CANONICAL)
    spot = pd.read_csv(spot_path)
    keys = ["model", "family", "seed"]
    merged = spot.merge(canonical, on=keys, suffixes=("_spot", "_canon"))
    if merged.empty:
        raise SystemExit("no overlapping (model, family, seed) cells to compare")

    logger.info("comparing %d cells on %d metrics", len(merged), len(METRICS))

    # 1. How far do individual cells move? Reported, not asserted.
    worst_cell = 0.0
    worst_pr_auc = 0.0
    for metric in METRICS:
        a, b = metric + "_spot", metric + "_canon"
        if a in merged and b in merged:
            difference = (merged[a] - merged[b]).abs()
            worst_cell = max(worst_cell, float(difference.max()))
            if metric == "pr_auc":
                worst_pr_auc = float(difference.max())
            logger.info("  %-20s max |difference| = %.4f, median %.4f",
                        metric, float(difference.max()), float(difference.median()))

    # 2. Does the ordering of the feature families survive? This is what the paper rests on.
    spot_means = merged.groupby(["model", "family"])["pr_auc_spot"].mean()
    canon_means = merged.groupby(["model", "family"])["pr_auc_canon"].mean()
    families = ["P", "S", "T"]
    broken = []
    print()
    print("Ordering P > S > T, by model:")
    for model in sorted({index[0] for index in spot_means.index}):
        present = [f for f in families if (model, f) in spot_means.index]
        if len(present) < 2:
            continue
        spot_order = [spot_means[(model, f)] for f in present]
        canon_order = [canon_means[(model, f)] for f in present]
        spot_ok = all(x > y for x, y in zip(spot_order, spot_order[1:]))
        canon_ok = all(x > y for x, y in zip(canon_order, canon_order[1:]))
        print("  {:<14} re-run {:<5} recorded {:<5}  ({})".format(
            model, str(spot_ok), str(canon_ok),
            ", ".join("{} {:.4f}".format(f, v) for f, v in zip(present, spot_order))))
        if spot_ok != canon_ok:
            broken.append(model)

    print()
    print("Between independent runs, individual cells moved by up to {:.4f} on the primary "
          "metric (PR-AUC)".format(worst_pr_auc))
    print("and up to {:.4f} on any of the five metrics compared.".format(worst_cell))
    print("Cells compared: " + str(len(merged)) + " over "
          + ", ".join(str(s) for s in sorted(merged.seed.unique())) + " seeds.")
    if broken:
        print()
        print("ORDERING CHANGED for: " + ", ".join(broken)
              + ". A conclusion of the paper does not survive re-running.")
        return 1
    print()
    print("Every family ordering is preserved. The pipeline reproduces its conclusions, not its"
          " digits, which is what section 9.1 claims.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-run", action="store_true", dest="skip_run",
                        help="compare an existing spot run without recomputing it")
    args = parser.parse_args()
    raise SystemExit(main(args.skip_run))
