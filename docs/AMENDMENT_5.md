# Amendment 5 — revision analyses for the BCRA submission

**Status: drafted [DATE], before any of the analyses below was run on the real data.**
Commit this file and record its commit hash and SHA-256 here, and attach it to the OSF
registration (see the runbook), *before* running `scripts/run_revision.ps1`.

None of the analyses below was in the frozen pre-registration. All are **exploratory**. They are
added because reviewers of the first submission round, and the authors' own limitations section,
identified questions the frozen design could not answer. Every result file they write carries
`"status": "exploratory (Amendment 5), not pre-registered"`, and the manuscript will report them
in a separate subsection, after the frozen verdicts, which they cannot change.

| ID | Question | Script | Decision rule fixed now |
|---|---|---|---|
| E1 | Is P+T equivalent to P, not merely "not different"? | `scripts/equivalence_tests.py` | Report TOST at margins 0.0025, 0.005, 0.01, 0.02 PR-AUC and the smallest margin at which equivalence holds. No margin is claimed as the margin. |
| R1 | Do the family orderings hold at other forecast origins? | `rolling_origin.py` | An ordering "holds" if it has the same sign at ≥ 4 of 5 origins. |
| R2 | Do they survive 5–20% training-label noise? | `label_noise.py` | Ordering of P, S, T unchanged at every noise rate. |
| R3 | Was T undertuned (444 columns)? | `feature_selected_T.py` | T_sel chosen on validation only; T counts as undertuned if T_sel beats S on test. |
| R4 | Does a GNN gain from T? | `gnn_baseline.py` | Report P+T − P over seeds; "gains" requires the seed-level 95% interval to exclude zero. |
| R5 | Does T's usefulness track cycle density when task is held fixed? (synthetic) | `synthetic_density.py` | Account supported if T is at base rate when p_cycle = 0 and T − S rises monotonically over the first four p_cycle values. |
| R6 | Same task, other view: transaction classification from address-neighbourhood topology | `tx_on_address_view.py` | The transaction-level conclusion is reversed if P+T_addr beats P with a seed-level interval excluding zero, or T_addr beats S_tx. |
| R7 | Is the Hyp3 lift neighbourhood overlap or hindsight? | `hyp3_controls.py` | Report lift@10 at exclude_hops 0/1/2 with and without time cut; the frozen Hyp3 verdict is unchanged whatever the outcome. |
| R8 | Can Hyp5 be tested? | `hyp5_feasibility.py` | Re-test only if a split exists with ≥ 20 test positives; any such split is a deviation and is labelled as one. |
| R9 | Prospective test of the screen | `screen.py --predict` on a graph not used in this study | Thresholds are those in `screen.py` at this commit. The prediction file is hashed and registered before the labelled comparison is run. |

**Known circularity, stated now.** The screen thresholds in `screen.py` were chosen with knowledge
of the Elliptic results (22.1% vs 68.5% cyclic share). Only R9 on a new graph tests the screen;
R5 tests the mechanism behind it. Neither Elliptic view can serve as evidence for the screen.
