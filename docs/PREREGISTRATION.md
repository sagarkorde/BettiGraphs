# Pre-registration — BettiGraphs

**Frozen:** 2026-09-27, before any model was fitted.
**Status:** this document is append-only. Amendments are added as dated entries in §10 with a
reason; nothing above §10 is edited after the freezing commit.

## Why this document exists

The brief forbids writing a result and then constructing an experiment that produces it. The only
robust defence against that — including against doing it unintentionally — is to fix the
hypotheses, the splits, the metric and the statistical procedure in advance, commit them, and
record the commit hash. That hash is the audit trail. Everything reported in the manuscript is
then classified as either **confirmatory** (pre-registered here) or **exploratory** (discovered
afterwards), and the two are never presented as though they carried the same evidential weight.

### What was inspected before freezing, and what was not

Full disclosure, because it bears on how much this pre-registration is worth.

**Inspected:** file formats, column names, node and edge counts, the containment relationship
between Elliptic and Elliptic++, the number of cross-time-step edges, how many time steps each
address spans, the distribution of `total_BTC`, and the **per-time-step count of illicit, licit and
unlabelled transactions**. All recorded in `docs/DATA_FINDINGS.md`.

The per-step label counts were inspected for two specific reasons, both of which had to be settled
before this document could be written honestly: to confirm that each split contains enough
positives to estimate PR-AUC at all, and to establish whether the regime change that hypothesis
**H6** depends on actually exists in the data. It does — illicit counts run 80–340 per step through
step 42 and collapse to 24, 24, 5, 2, 22 across steps 43–47. Pre-registering H6 without checking
would have risked freezing an untestable hypothesis.

**Not inspected:** any relationship between features and labels. No model has been fitted, no
correlation, mutual information or importance computed, no persistence diagram compared across
classes. The first such computation happens after this document is committed.

---

## 1. Research questions, as revised by the data

RQ1, RQ2, RQ3 and RQ5 stand as originally framed. **RQ4 is rewritten**, because the original
phrasing asked a question this data cannot answer: the Elliptic transaction graph has zero
cross-time-step edges, so a transaction has no temporal successor and "the evolution of a
transaction's ego-subgraph" is not a quantity the dataset contains. Recording that now, rather
than discovering it at review, is the point of gating.

| RQ | Question |
|---|---|
| **RQ1** | Do Betti-number profiles of ego-subgraphs differ between illicit and licit entities? |
| **RQ2** | Do persistent-homology representations improve detection over classical graph-structural features? |
| **RQ3** | Does topological distance between wallet ego-subgraph persistence diagrams measure behavioural relatedness better than distance in non-topological feature spaces? |
| **RQ4a** | *(revised)* Do snapshot-level topological summaries β₀(t), β₁(t), persistence entropy(t) vary systematically across the 49 time steps, and do they track the illicit base rate? |
| **RQ4b** | *(revised)* On the subpopulation of addresses observed in ≥ 2 time steps, do entity-level topological trajectories carry label information beyond a single static snapshot? |
| **RQ5** | Does combining topological with provided and classical features outperform every single family? |

RQ4a is **descriptive**: n = 49 supports a figure and a correlation, not a classification claim,
and it will not be presented as one. RQ4b is inferential but runs on a biased subpopulation
(7.5% of addresses), which is stated wherever its results appear.

---

## 2. Hypotheses, with rejection criteria fixed in advance

Each hypothesis has a pre-specified test and a criterion under which it is **rejected**. A
hypothesis whose criterion is met is reported as rejected. None is reframed after the fact.

| ID | Hypothesis | Test | Rejected if |
|---|---|---|---|
| **H1** | Betti-curve and persistence descriptors differ in distribution between illicit and licit entities | Mann–Whitney U per descriptor on the training split, with Cliff's δ | no descriptor has Holm-corrected *p* < 0.05 **and** \|δ\| ≥ 0.15 |
| **H2** | TDA-only features achieve higher illicit PR-AUC than classical-structural-only features | paired across 10 seeds, Wilcoxon signed-rank | the bootstrap 95% CI for the mean difference contains 0 |
| **H3** | Topological wallet distance ranks same-class wallets above **both** control feature spaces | MAP and NDCG@{10,50}, bootstrap CI over query wallets | it fails to beat either control with a CI excluding 0 |
| **H4** | Snapshot-level β₁ and persistence entropy correlate with the per-step illicit base rate | Spearman ρ over 49 steps, with a block-permutation null | 95% CI for ρ contains 0 |
| **H5** | Entity-level temporal topological features add PR-AUC over static TDA on the multi-step cohort | paired across 10 seeds, Wilcoxon | CI for the difference contains 0 |
| **H6** | **Topological features degrade less across the step-43 regime change than the provided features do** | per-step illicit PR-AUC; compare pre-shift (≤ 42) vs post-shift (≥ 43) drop between feature families, paired across seeds | the difference in degradation between families has a CI containing 0 |
| **H7** | The union P+S+T outperforms every single family | paired PR-AUC, Holm-corrected across all family pairs | the union is not significantly best against at least one single family |

**H6 is the primary hypothesis.** It is the one the paper is built around, for a stated reason:
Elliptic's provided features are a very strong baseline under a random split, so a claim of
outright superiority is a weak bet, whereas a claim about *when and why* topology helps is both
more defensible and more useful. If H6 is rejected the paper reports that.

**Pre-committed mechanism check for H6.** A drop in performance after step 43 could arise from
label shift (the illicit base rate falls from roughly 11% to 1–3%), covariate shift, or concept
shift. These are not the same finding. We will decompose it: base rates per step are already
known; covariate shift is measured by a domain classifier trained to separate pre- from
post-shift features; residual change is attributed to concept shift. H6 is a claim about relative
degradation between feature families and is interpreted only alongside this decomposition.

---

## 3. Data and splits — frozen

**Dataset:** Elliptic++ throughout. `P166` denotes the exact Elliptic-equivalent feature subset
(time step + `Local_feature_1…93` + `Aggregate_feature_1…72`), verified identical to Elliptic's
features to 3.6 × 10⁻¹⁵.

**Transaction-level temporal split**, by time step, fixed now and not revisited:

| Split | Time steps | Illicit | Licit |
|---|---|---|---|
| Train | 1–34 | 3,462 | 26,432 |
| Validation | 35–41 | 675 | 7,154 |
| Test | **42–49** | 408 | 8,433 |

The test window deliberately straddles the step-43 regime change, with step 42 as the last
pre-shift step. A secondary **literature-comparable** split (train 1–34, test 35–49) is also
reported so our numbers can be placed beside published ones, but no hypothesis is tested on it.

**Wallet-level split:** by an address's first observed time step, using the same 1–34 / 35–41 /
42–49 boundaries.

**Unlabelled nodes** (77% of transactions, 68% of wallets) are retained in the graph and excluded
from supervised evaluation. Removing them would sever the paths that carry the structure being
measured.

**Leakage controls**, each enforced in code and unit-tested:

1. Zero cross-time-step edges is a *verified* property of this graph, so no ego-subgraph can
   reach another split structurally. This is measured, not assumed.
2. Scalers, imputers and any fitted transformation see the training split only.
3. Persistence-image ranges and Betti-curve grids are **fixed constants** on `[0, 1]` under rank
   normalisation, so they are not fitted to data at all.
4. No target encoding and no statistic pooled across a split boundary.
5. Hyperparameters are selected on validation only. The test split is read **once per model
   family**, after all selection is complete.

---

## 4. Feature families — frozen

| Code | Family | Contents |
|---|---|---|
| **P** | provided | `P166`, plus the 17 named Elliptic++ columns as `P+` where stated |
| **S** | classical structural | in/out/weighted degree, degree asymmetry, local clustering coefficient, triangle count, PageRank, approximate betweenness, closeness, eigenvector centrality, k-core number, ego-net order/size/density/diameter, reciprocity, assortativity, **cycle rank**, 3- and 4-node motif counts |
| **T** | topological | Betti curves β₀, β₁ (and β₂ where the complex admits it), lifetime statistics, persistence entropy, persistence images, persistence landscapes, Euler-characteristic curve — as implemented in `src/features/topological.py` |

**Cycle rank is deliberately placed in S.** For a 1-dimensional complex β₁ equals the cycle rank
exactly, so if the Betti *curve* carries no information beyond that scalar, the S-vs-T comparison
will show it. Putting the scalar in the baseline is the honest way to ask whether the profile
earns its cost.

Combinations evaluated: P, S, T, P+S, P+T, S+T, P+S+T.

---

## 5. Models — frozen

**Tabular:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.
**Graph:** GCN, GraphSAGE, GAT on `G_tx`.
**Temporal graph:** snapshot GCN + GRU.
**Unsupervised, for the topological anomaly score:** Isolation Forest, Local Outlier Factor,
One-Class SVM over T.

Hyperparameter search is a fixed random search of 30 draws per (model, family) from ranges
declared in `configs/`, scored by validation illicit PR-AUC. Class imbalance is handled by class
weighting; resampling is an ablation, not the default, because it alters the base rate that H6 is
about.

---

## 6. Metrics — frozen

**Primary: PR-AUC on the illicit class.** Chosen before seeing any result, because measured
imbalance is 9.25 : 1 at transaction level and 17.60 : 1 at wallet level.

Also reported: illicit precision, recall and F1; macro-F1; ROC-AUC; MCC; balanced accuracy.
Accuracy appears for completeness only and supports no claim.

For RQ3: Precision@K, Recall@K, MAP, NDCG@{10,50}; plus ARI, NMI and silhouette for clustering.

**No metric may be substituted after results are seen.** If the primary metric proves unsuitable
for a reason discovered later, that is recorded as an amendment in §10 with the reason, and both
the original and the replacement are reported.

---

## 7. Statistical procedure — frozen

- **10 seeds**, fixed as `(0,…,9)` in `src/utils/seeds.py`. Not drawn at runtime.
- Mean ± standard deviation, and bootstrap 95% CIs (10,000 resamples) for every reported metric.
- Paired comparisons across seeds by **Wilcoxon signed-rank**.
- **Holm–Bonferroni** correction across the declared family of comparisons. The family is defined
  now: the 21 pairwise comparisons among the 7 feature combinations, plus the 7 within-T ablation
  comparisons, plus H6's degradation comparison. Corrections are applied within this family, and
  exploratory tests are reported separately and uncorrected, labelled as exploratory.
- Effect size by **Cliff's δ** for distributional comparisons and Cohen's *d* for paired means.
- **DeLong's test** for ROC-AUC comparisons on a shared test set.
- Per-step results are always reported **with the illicit count for that step**, because steps 45
  and 46 hold only 5 and 2 illicit transactions and a PR-AUC there is nearly meaningless on its
  own. Post-shift conclusions rest on the pooled 43–49 window, not on single steps.

**No superiority claim is made from a point difference whose CI includes zero**, however large it
looks.

---

## 8. Filtration selection protocol — frozen

Five candidates: `value_superlevel`, `temporal_sublevel`, `core_sublevel`, `geodesic_rips`,
`resistance_rips`. Selection is scored on the **validation split only**, by:

1. discriminative power (validation illicit PR-AUC with a fixed LightGBM configuration),
2. stability under random edge deletion at 5% and 10% (mean L2 drift of the signature),
3. wall-clock cost per subgraph.

The full comparison table is reported whatever it shows. The test split plays no part in choosing
a filtration, a radius, a resolution or a complex dimension.

Ego radius k ∈ {1, 2, 3}, Betti-curve resolution R ∈ {16, 32, 64}, complex dimension ∈ {1, 2, 3}
and persistence-image resolution ∈ {6, 8, 12} are selected by the same procedure.

---

## 9. What we commit *not* to do

1. Not change the splits after seeing test results.
2. Not switch the primary metric to one that looks better.
3. Not drop a seed, a time step or a model because it is unfavourable.
4. Not present an exploratory finding as confirmatory.
5. Not report β₂ on the bipartite view, where it is zero by construction rather than by finding.
6. Not claim novelty beyond "our search of the surveyed literature did not identify…".
7. Not interpret topological similarity between wallets as evidence of common ownership.
8. Not report a tuned TDA model against an untuned baseline; the search budget is equal per
   family.
9. Not quietly narrow an experiment until it produces a positive result. If the topological
   features add nothing, the paper says so, with the statistics that establish it.

---

## 10. Amendments

Amendments are appended here with date, what changed, and why — never by editing above.

---

### Amendment 1 — 2026-09-27: ego radius, node bound, and topology view

**Still no model has been fitted and no label has been used.** Every measurement behind this
amendment is a structural property of the graphs — degree, component count, cycle rank, subgraph
size, wall-clock cost. None of it touches the labels, so this is not a result-driven change. It is
recorded here rather than applied silently because it changes the search space §8 commits to.

#### What was measured

`G_tx` (transactions): 203,769 nodes, 234,355 undirected edges, **mean degree 2.30**, exactly
**49 connected components** — independently confirming the disjoint-snapshot finding — and a
global cycle rank of 30,635. The graph is extremely sparse, and 79% of nodes have degree ≤ 2.

Fraction of ego-subgraphs containing **any** cycle, over 3,000 random transactions:

| radius | median nodes | median edges | fraction with a cycle | mean cycle rank |
|---|---|---|---|---|
| 1 | 3 | 2 | **3.5%** | 0.13 |
| 2 | 6 | 5 | 21.7% | 3.49 |
| 3 | 16 | 16 | 42.2% | 19.72 |
| 4 | 53 | 54 | 57.3% | 63.68 |

`G_addr` (addresses): 822,942 nodes, 2,776,827 undirected edges, **mean degree 6.75**, one giant
component holding 100% of nodes, global cycle rank **1,953,887** — sixty-four times that of
`G_tx`. Maximum degree 22,996.

| radius | median nodes | median edges | fraction with a cycle | mean cycle rank |
|---|---|---|---|---|
| 1 | 3 | 2 | 7.8% | 0.90 |
| 2 | 76 | 114 | **66.7%** | 605.96 |
| 3 | 407 | 751 | 84.4% | 1266.22 |

#### Why this forces a change

At radius 1 on `G_tx`, H₁ is zero for 96.5% of entities. A β₁ descriptor that is identically zero
for almost every case cannot carry information, so the originally pre-registered radius set
{1, 2, 3} would have confined the strongest topological descriptor to a regime where it is
structurally incapable of saying anything. That is a defect in the plan, not a finding about
topology, and it should be fixed before the experiment rather than discovered in the results.

#### Changes

1. **Ego radius search space extended from {1, 2, 3} to {1, 2, 3, 4}.** Selection is still on the
   validation split by the §8 procedure; only the candidate set grows.

2. **Node bound fixed at 25,000, with a principled truncation rule.** Measured truncation rates at
   radius 2 on `G_addr` were 26.8% at a 1,000-node cap, 22.8% at 2,000, 15.5% at 5,000, 5.3% at
   10,000 and **0.0% at 25,000** (largest observed radius-2 ego-net: 23,389 nodes). Cost at 25,000
   is 26 ms mean, 139 ms at the 95th percentile, about 1 s for the largest hub, and roughly 4
   minutes for all 265,354 labelled wallets across 30 cores at under 1 GiB resident. Since
   truncation at the earlier bound would have been a size-dependent selection effect on a fifth of
   all cases, and removing it costs four minutes, it is removed. Where the bound still bites, the
   frontier is subsampled **uniformly at random with a seed derived from the seed node** rather
   than by array order, which is unbiased with respect to node properties and exactly
   reproducible; the affected cases are counted and reported.

3. **Topology view added to the search space.** The label sits on the entity, but the topology need
   not be computed on the same graph view. For transaction-level classification the following
   three views are candidates, selected on validation by the §8 procedure:
   - `tx_ego` — ego-subgraph in `G_tx` (as originally planned),
   - `bip_ego` — ego-subgraph in `G_bip`, the bipartite address-transaction view,
   - `addr_ego` — the union ego-subgraph in `G_addr` over the addresses the transaction touches.

   The motivation is structural and stated above: `G_tx` is too sparse for H₁ to be informative at
   small radii, while `G_addr` is not. Adding the view as a selectable factor is preferable to
   silently switching to whichever one works, and preferable to persisting with a view that a
   label-free measurement says cannot carry the signal.

#### What does not change

The hypotheses, the splits, the primary metric, the statistical procedure, the feature families
and the §9 commitments are all untouched. H6 remains primary. The transaction-level task remains
the benchmark on which results are compared to published work, precisely because that is what
prior work reports.


---

### Amendment 2 - 2026-09-27: a size-matched control is added to H1, reported alongside the pre-registered test

**This amendment makes a test stricter after seeing its result, so it is set out in full and the
pre-registered verdict is reported regardless.**

#### What happened

H1 as frozen in section 2 is: Mann-Whitney U per descriptor with Cliff's delta, rejected unless
some descriptor achieves Holm-corrected *p* < 0.05 **and** |delta| >= 0.15. Run as written, on 428
topological descriptors over the training window, **22 descriptors pass**. By the pre-registered
criterion, **H1 is supported**.

But the H4 result, obtained earlier the same day, had already shown what that kind of pass can be
worth. There, four descriptors cleared an equivalent bar and every one turned out to be an
elementary graph count wearing a topological name; two returned partial correlations of exactly
0.000 because they are algebraically determined by the vertex and edge counts.

The same confound is present here in a different form, and it was measured before this amendment
was written: illicit ego-subgraphs are **smaller** than licit ones (median order 21 against 26 at
radius 3), and beta_1 grows with order. A raw beta_1 difference can therefore be a size difference.

#### The change

Every descriptor is additionally tested **within strata of ego-subgraph order and edge count**,
with per-stratum Cliff's delta pooled by stratum weight. Three levels of evidence are now reported,
and the manuscript reports all three rather than whichever one suits:

| Level | Requirement | Topological descriptors passing |
|---|---|---|
| **L1, as pre-registered** | Holm *p* < 0.05 and abs(delta) >= 0.15 | **22 of 428** |
| **L2** | L1, plus pooled size-matched abs(delta) >= 0.15 with an interval excluding zero | **14** |
| **L3** | L2, plus the effect keeping its sign in every size stratum | **0** |

For comparison, on the 43 classical descriptors: best size-matched abs(delta) = **0.41**, and
**10 pass L3**, all of them transaction-value statistics (`ego_weight_std` -0.32, `ego_weight_sum`
-0.31, `ego_weight_mean` -0.29, `seed_weighted_degree` -0.27). The best topological size-matched
effect is **0.18**, with a standard error of 0.083.

#### Why this is not moving the goalposts to manufacture a negative

Three reasons, and the reader is entitled to weigh them.

1. The control is applied **identically to both feature families**, and it is the classical family
   that survives it. A criterion invented to disadvantage topology would not have that property.
2. L3 is strict - it requires a consistent sign across all eight strata - and
   `ego_degree_two_fraction`, the single largest raw effect in the study at 0.41, fails it too.
3. The pre-registered L1 verdict is reported first and without qualification wherever H1 appears.

The honest summary is therefore: **H1 is supported as pre-registered and does not survive a
size-matched control, while the classical comparison family does survive it.** That is what goes in
the manuscript, in those words.

#### Not changed

No other hypothesis is amended. H6 remains primary and its criterion is untouched.


---

### Amendment 3 - 2026-09-27: H6 gains a second evaluation protocol with a usable pre-shift baseline

**Made before H6 was tested.** No topological result for H6 exists yet; only a single baseline cell
had been computed, and it is what exposed the problem.

#### The problem

H6 compares how much performance degrades across the regime change at step 43. As frozen, it was to
be measured on the primary test window, steps 42-49. That window contains exactly **one** pre-shift
step and seven post-shift steps. A degradation estimate whose baseline is a single time step is far
too fragile to carry the study's primary hypothesis, and the fragility is structural rather than a
matter of sample size within the step.

This surfaced when the first completed cell (provided features, random forest) reported a 70%
relative drop -- PR-AUC 0.9221 before the shift against 0.2766 after. The effect is real and large,
but 0.9221 rests on step 42 alone.

#### The change

H6 is now evaluated under **two** protocols, both reported:

1. **train+val**, as originally frozen: fit on steps 1-41, score steps 42-49. One pre-shift step.
2. **train-only** (new): fit on steps 1-34 alone, score every step from 35 to 49. This gives
   **8 unseen pre-shift steps against 7 unseen post-shift steps**, with both sides out-of-sample and
   drawn from the same contiguous span, so the comparison isolates the regime change instead of
   confounding it with a split boundary.

The statistical test, the metric, and the requirement that a confidence interval exclude zero are
all unchanged. The primary reported figure for H6 becomes the train-only protocol, because it is the
one with a defensible baseline; the original protocol is reported beside it, and any disagreement
between them is itself reported rather than resolved silently.

#### Why this is not selecting a protocol to suit a result

No H6 result for any topological family existed when this was written. The only computed cell was a
baseline, and the change was prompted by the shape of the split rather than by the direction of any
comparison. Both protocols are reported for every family, so a reader can see whether the choice
mattered.

#### Consequence

The two already-computed cells of the transaction experiment lack the train-only records and were
discarded so that all 32 cells are produced under identical conditions.

#### Not changed

Hypotheses H1 to H5 and H7, the splits, the primary metric, the correction family and the section 9
commitments are untouched.


---

### Amendment 4 - 2026-09-28: the pre-registered significance rule is unreachable at ten seeds

**This corrects a defect in section 7 of this document.** The procedure as frozen cannot declare
any comparison significant, whatever the data show. That is a property of the test, not of the
results, and it was found by running the analysis and seeing an obviously wrong verdict.

#### The defect

Section 7 specifies paired Wilcoxon signed-rank tests across ten seeds, Holm-Bonferroni corrected
across a declared family of comparisons.

The signed-rank statistic is **discrete**. With `n` paired samples all differing in the same
direction -- the strongest possible evidence -- the exact two-sided p-value is `2 / 2**n`. At the
pre-registered ten seeds:

```
floor                       = 2 / 2**10          = 0.00195312
Holm's strictest threshold  = 0.05 / 31          = 0.00161290
```

The floor sits above the threshold, so **no comparison in a 31-member family can reach significance
at any effect size**. Run as written, the analysis reported "no significant difference" for a
PR-AUC gap of **0.6549** whose 95% interval was **[+0.6521, +0.6574]** — nowhere near zero. Every
one of the 31 comparisons returned the same verdict, which is what made the defect visible.

#### The correction

The family-wise decision moves to a **Bonferroni-adjusted bootstrap interval**: each comparison's
interval over the same paired per-seed differences is recomputed at level `1 - alpha/m`, and
significance requires it to exclude zero. This controls the family-wise error rate, uses the same
data and the same pairing, and is not subject to a discreteness floor.

The Wilcoxon p-value and its Holm adjustment are still computed and reported for every comparison,
now alongside the floor for that sample size, so a reader can see both the original procedure and
why it was insufficient.

#### What this does not change

The metric, the splits, the seeds, the pairing, the family of comparisons, and the requirement that
an interval exclude zero are all unchanged. The correction is strictly about which interval the
threshold is applied to.

#### Why not simply add seeds

Twenty seeds would lower the floor to 7.7e-6 and the original rule would work. It was rejected for
this round because it doubles the cost of the most expensive stage for no gain in what the study can
conclude: the effects that matter here are between 0.4 and 0.65 in PR-AUC against a seed-to-seed
standard deviation under 0.005, so the intervals are decisive either way. A future replication
should use twenty seeds and keep the original rule, and section 7 of any successor pre-registration
should check the discreteness floor against the intended correction **before** freezing.

#### Disclosure

This amendment was written after seeing that every comparison returned null, which is the point at
which the defect became apparent. It cannot favour any particular result: the correction is applied
identically to all 31 comparisons, and the comparisons it newly permits to reach significance
include several that count **against** the topological features -- among them P vs T at a gap of
0.65 in the conventional features' favour.
