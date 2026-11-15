"""R8. Can Hyp5 be tested at all? A label-count feasibility check (no features, seconds to run).

Hyp5 was left untested because the frozen temporal split left one illicit trajectory address
in the test window against the 20 positives the adequacy gate requires. Before spending compute
on trajectories, this counts, for each minimum-observation threshold and each candidate split,
how many illicit and licit cohort addresses would land in the test window (by first appearance)
when the WHOLE cohort is used instead of a 3,000-address sample.

Any re-test chosen from this table is a deviation from the frozen protocol and must be reported
as such (Amendment 5). The table itself is the honest answer to "why not just test it".

Usage:
    .venv/Scripts/python.exe experiments/revision/hyp5_feasibility.py
"""
from __future__ import annotations

import pandas as pd

from common import RAW_PP, setup_logging, write_json, OUT

SPLITS = {"frozen (test 42-49)": 42, "test 35-49": 35, "test 30-49": 30, "test 25-49": 25}


def main() -> None:
    setup_logging()
    w = pd.read_csv(RAW_PP / "Actors Dataset" / "wallets_features.csv", usecols=["address", "Time step"])
    c = pd.read_csv(RAW_PP / "Actors Dataset" / "wallets_classes.csv")
    f = w.merge(c, on="address")
    f = f[f["class"].isin([1, 2])]
    g = f.groupby("address").agg(n_steps=("Time step", "nunique"), first=("Time step", "min"),
                                 cls=("class", "first"))
    rows = []
    for k in (2, 3, 4):
        cohort = g[g.n_steps >= k]
        for name, start in SPLITS.items():
            test = cohort[cohort["first"] >= start]
            rows.append({"min_observations": k, "split": name, "cohort": int(len(cohort)),
                         "cohort_illicit": int((cohort.cls == 1).sum()),
                         "test_n": int(len(test)), "test_illicit": int((test.cls == 1).sum()),
                         "gate_20_positives": bool((test.cls == 1).sum() >= 20)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hyp5_feasibility.csv", index=False)
    write_json("hyp5_feasibility.json", {"rows": rows})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
