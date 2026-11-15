"""Equivalence (TOST) tests for the "no improvement" claims.

Exploratory, post hoc: NOT part of the frozen pre-registration. A non-significant
difference is absence of evidence; TOST asks whether the paired difference lies
inside +/- margin. Because the margin was not frozen, the script reports a range of
margins and the smallest margin at which equivalence would be declared, so the
reader can apply their own.

Unit: learner seeds (n=20), same learner for both families. Seed-level spread
understates across-time variation (see the step-level table in the manuscript).

Usage (from the repository root):
    .venv/Scripts/python.exe scripts/equivalence_tests.py --tag r2s20
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

CONTRASTS = [  # (a, b, learner) -> difference a - b
    ("P+T", "P", "lightgbm"),
    ("P+S", "P", "lightgbm"),
    ("Pplus", "P", "lightgbm"),
    ("S+T", "S", "random_forest"),
    ("S+T", "S", "lightgbm"),
    ("P+S+T", "P", "lightgbm"),
]
MARGINS = [0.0025, 0.005, 0.01, 0.02]
ALPHA = 0.05


def tost(diff: np.ndarray, margin: float) -> dict:
    n = diff.size
    mean, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n)
    if se == 0:
        p = 0.0 if abs(mean) < margin else 1.0
    else:
        p_lo = stats.t.sf((mean + margin) / se, n - 1)   # H0: mean <= -margin
        p_hi = stats.t.cdf((mean - margin) / se, n - 1)  # H0: mean >= +margin
        p = float(max(p_lo, p_hi))
    return {"margin": margin, "p_tost": p, "equivalent": bool(p < ALPHA)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r2s20")
    ap.add_argument("--results-dir", default="results/raw")
    ap.add_argument("--out", default="results/statistical_tests/equivalence_tests.json")
    args = ap.parse_args()
    files = glob.glob(f"{args.results_dir}/transaction_results_{args.tag}_*.csv")
    if len(files) != 1:
        raise SystemExit(f"expected one results file for tag {args.tag}, found {files}")
    df = pd.read_csv(files[0])
    out = {"source": Path(files[0]).name, "unit": "learner seed", "alpha": ALPHA,
           "status": "exploratory, not pre-registered", "contrasts": []}
    for a, b, learner in CONTRASTS:
        piv = df[df.model == learner].pivot(index="seed", columns="family", values="pr_auc")
        d = (piv[a] - piv[b]).dropna().to_numpy()
        n = d.size
        se = d.std(ddof=1) / np.sqrt(n)
        half = stats.t.ppf(1 - ALPHA, n - 1) * se       # 90% CI <-> TOST at 5%
        lo, hi = d.mean() - half, d.mean() + half
        out["contrasts"].append({
            "a": a, "b": b, "learner": learner, "n_seeds": int(n),
            "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "ci90": [float(lo), float(hi)],
            "smallest_equivalence_margin": float(max(abs(lo), abs(hi))),
            "tost": [tost(d, m) for m in MARGINS],
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for c in out["contrasts"]:
        flags = " ".join(f"{t['margin']}:{'EQ' if t['equivalent'] else '--'}" for t in c["tost"])
        print(f"{c['a']:>6} - {c['b']:<5} [{c['learner']:<13}] mean {c['mean_diff']:+.4f} "
              f"90% CI [{c['ci90'][0]:+.4f}, {c['ci90'][1]:+.4f}] min margin "
              f"{c['smallest_equivalence_margin']:.4f} | {flags}")


if __name__ == "__main__":
    main()
