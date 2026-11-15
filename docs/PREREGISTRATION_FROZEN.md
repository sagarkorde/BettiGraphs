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

None. Amendments are appended here with date, what changed, and why — never by editing above.
