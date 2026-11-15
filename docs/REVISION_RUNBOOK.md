# Revision runbook — BCRA submission

What has been done for you, what you need to run, and what only you can do. In order.

## 0. Already done (no action)

* `Publication/BCRA/BCRA_Manuscript.docx` — reframed abstract, introduction and contributions
  (screen first, negative result as evidence), four contributions, Elsevier-style section
  numbering (1, 1.1, …) and table numbering (1–13), Elsevier declarations, generative-AI
  declaration rewritten to Elsevier's wording (figures are rendered by code, not by AI), author
  bio removed, TOST result added to §6.4 and Table 12. **No other number changed** — checked by
  diffing every numeric literal against the previous version.
* `Publication/BCRA/BCRA_Supplementary.docx` — four authors, cross-references renumbered.
* `Publication/BCRA/BCRA_Highlights.docx`, `BCRA_CoverLetter.docx`.
* `scripts/equivalence_tests.py` and `results/statistical_tests/equivalence_tests.json` — run on
  your existing r2s20 results. P+T vs P: 90% CI [+0.0005, +0.0059], equivalent within ±0.01.
* `experiments/revision/` — nine new experiments, each smoke-tested on a synthetic stand-in for
  Elliptic++ with the same file layout (not on the real data, which never left your machine).

## 1. Before running anything (you, ~1 hour)

1. **Fill the placeholders** in the manuscript: CRediT roles for the three co-authors, funding.
   Search for `[TO CONFIRM`.
2. **Commit `docs/AMENDMENT_5.md`**, then register it. Create a free OSF account, start an
   "OSF Preregistration" or "Open-Ended Registration", attach `docs/PREREGISTRATION.md`, its
   hash file and `docs/AMENDMENT_5.md`, and submit. You get a DOI with an independent timestamp.
   Do this **before** step 2, so the revision analyses are registered before they run.
3. **Push the repository** (`gh auth login`, `git push -u origin main`) and keep it **private**
   until acceptance.
4. **Zenodo**: link Zenodo to GitHub, make a release `v1.0-bcra`; Zenodo mints a DOI. Put the
   OSF and Zenodo DOIs in the Data availability section.

## 2. Run the experiments (your machine, mostly unattended)

```powershell
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
powershell -ExecutionPolicy Bypass -File scripts\run_revision.ps1 -Jobs 8
```

Resumable: finished steps are skipped. Rough cost on the machine that produced r2s20:

| Step | Cost | Notes |
|---|---|---|
| equivalence, hyp5_feasibility, screens | minutes | |
| synthetic | ~1 h | single process |
| feature_selected, label_noise | 1–3 h each | reuse the r2 feature cache |
| rolling_origin | ~4 h | 5 origins × 6 families × search |
| gnn | ~1 h + one-off feature extraction for all 203k transactions | GPU used if present |
| tx_on_address | several hours | 49 time-cut address graphs; add `--with-classical` only if time allows (~8× slower) |
| hyp3_controls | about as long as the original wallet run, twice | time cut doubles extraction |

**Hyp5:** run `hyp5_feasibility` first and look at its table. A projection from the sample in the
paper (1 illicit of 37 test addresses from 3,000) suggests the whole cohort gives ~2 test positives
under the frozen split, short of the gate of 20. If no row passes, keep Hyp5 as untested and cite
the table; that answers the reviewer better than a forced test.

## 3. Prospective screen (the biggest single upgrade; you choose the dataset)

Pick one labelled graph not used here. Candidates: Elliptic2 (Bitcoin subgraphs, money
laundering), an Ethereum phishing-detection transaction graph, or a later Bitcoin period.
Export its edges as `src,dst,value,time`, then:

```powershell
.venv\Scripts\python.exe experiments\revision\screen.py --edges edges.csv --name eth --predict
```

Register the printed SHA-256 on OSF (an update to the registration) **before** running any
labelled model on that graph. Then run the topology-vs-baseline comparison on it.

## 4. Where each result goes in the paper

Add one subsection after §6.10, "6.11 Revision analyses (exploratory, Amendment 5)", with one
short paragraph per row of `docs/AMENDMENT_5.md`, and a table:
family orderings across rolling origins (R1) and noise rates (R2); T_sel (R3); GraphSAGE P vs P+T
(R4). Put the synthetic result (R5) in §8.2, where it turns the density account from correlation
into a controlled test; it is the best candidate for a new figure (T − S against cyclic share).
R6 goes in §8.2 and removes Limitation 2 from §10 whatever it shows. R7 replaces the two caveats in §6.8.
Then update the abstract's last sentence and §11 to match.

## 5. Before submitting

* Re-run `scripts/build_manuscript.py`, then register the new numbers in `claims.json` the same way
  as the old ones, so the build check covers them.
* Run your own similarity check with the repository private.
* BCRA submission: Editorial Manager → confirm the APC screen shows the sponsored rate.
