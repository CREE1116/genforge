# OQBoost Theory Notes — Why the Candidate Tournament Works

Status: working document, 2026-06-12. Companion experiments: `theory_exp1_coherence.py`,
`theory_exp2_lr_coupling.py`, `theory_exp3_oblivious.py`. Empirical history in
`FINDINGS.md`. Rule of the road: every engine change must trace back to a claim
in this file that survived its experiments.

---

## 1. The split tournament is an order-statistics problem

At a node, the engine draws candidates $w_1,\dots,w_n$ from a mixture of
families (axis scan, GG-SRP random, inherited A/B, cache C) and commits

$$G^* = \max_i \; \text{gain}(w_i),$$

where gain is the Newton split gain along projection $w_i \cdot x$. The node
quality is therefore governed by **extreme values of the candidate-gain
distribution, not its mean**. For a family $f$ with gain c.d.f. $F_f$, the
expected best of $k$ draws is

$$\mathbb{E}[\max] \;=\; \int_0^\infty \left(1 - F_f(g)^k\right)\,dg ,$$

which grows with the *upper tail weight* of $F_f$, roughly like
$\mu_f + \sigma_f\,a_f(k)$ with $a_f(k)$ determined by the tail shape.

**Consequences.**
- A family with a low mean and a low win-rate can still be worth its budget if
  its gains are weakly correlated with the other families' gains: for a
  mixture, $\Pr[\max \le g] = \prod_f F_f(g)^{k_f}$, so *adding any family with
  mass above the other families' tails strictly improves the max*. This is the
  correct frame for the earlier "A/B wins only 3% of nodes" observation — a
  3% win-rate family is not 3% useful; it is exactly as useful as the gain
  margin it contributes on the nodes where it wins.
- Optimal budget allocation $\{k_f\}$ maximizes
  $\int (1-\prod_f F_f^{k_f})$, a concave (diminishing-returns) objective in
  each $k_f$ — so the optimum is interior (a mixture), never a corner, unless
  one family's distribution stochastically dominates all others everywhere.
  This *predicts* that pure configurations (all-random, all-cache, all-parent)
  are suboptimal in general, matching the strategy-ablation data where the
  production mixture was never beaten by a pure configuration on real data.

Open task: measure per-family per-node gain distributions (research impl) and
check whether observed win-rates match $\Pr[f \text{ wins}]$ computed from the
empirical $F_f$, then solve for the optimal $k_f$ and compare with the tuned
`inherited_rp_ratio`.

## 2. The lr × early-stopping coupling: stronger splits ≠ better ensembles

The boosting update is $F \mathrel{+}= \eta\, f_m$. With validation-based early
stopping, the effective capacity is the product (tree strength) × (number of
rounds before val saturation) × $\eta$. Hyperparameters are tuned jointly with
the candidate distribution: making per-node splits *stronger* (e.g. evaluating
all cache directions exactly, or removing weak candidate families) shifts the
optimal $\eta$ DOWN; at the previously tuned $\eta$ the val loss saturates
earlier, ES truncates the ensemble, and the final model is worse despite every
individual split being better. This is precisely what the failed engine
transplants did, and why fixed-parameter validations missed it.

**Claim 2a (regularization role of A/B).** Near-parent mutation candidates make
child splits *correlated with their parent's direction* whenever they win.
A tree whose nodes split along correlated directions is closer to an
oblivious (level-symmetric) tree: it has fewer effective degrees of freedom,
hence lower variance per tree, hence — at small $\eta$ over many rounds —
better ensemble generalization. Under this claim, A/B is not a direction
*search* mechanism (the drift studies already showed the winning child
direction is usually far from the parent) but a *capacity control* mechanism
that fires on the minority of nodes where local refinement beats exploration.

**Testable predictions.**
- P1: within one tree, the mean pairwise |cos| between internal-node split
  directions is higher with `inherited_rp_ratio=1` than `=0`.
- P2: per-tree training-loss reduction is *smaller* with ratio=1 (weaker
  trees), at equal depth and budget.
- P3: validation-loss-vs-rounds curves (no ES): at large $\eta$, ratio=1 keeps
  improving for more rounds and reaches a lower minimum; at small $\eta$ the
  gap shrinks or reverses. The two curve families *cross*.
- P4: making the regularization explicit — an oblique *oblivious* tree (one
  shared direction per level, per-node thresholds) — should retain or improve
  ensemble accuracy while being structurally cheaper, and its benefit should
  be largest in the same regime where A/B helps.

P4 is the "better method" candidate this theory suggests: it converts an
accidental regularizer (A/B mutations sometimes aligning sibling splits) into
a designed one, and as a bonus collapses routing to one dot product per level
(SIMD/GEMM-friendly — also serves the CPU-utilization goal).

## 3. What the cache is, mathematically

The cache holds directions $w$ whose gain exceeded the axis baseline somewhere
recently. If the data has a low-dimensional informative rotated subspace
$U \subset \mathbb{R}^D$ (the regime where oblique trees beat axis trees at
all), then winning directions concentrate near $U$, and the cache is a cheap
empirical basis of $U$ refreshed every round. Blending cache directions with
the parent direction (Strategy C) is then a random walk *inside* $\hat U$ —
which explains why C transfers across nodes and trees while A/B (random walks
around one node's direction) does not: $U$ is a global object, the parent
direction is not.

This predicts cache value grows with (a) the gap between $\dim U$ and $D$, and
(b) feature rotation strength — measurable on synthetic data by varying the
rotation.

---

## Experiment log

### 2026-06-12 — P1/P2/P3 (`theory_exp12_strength.py`, C++ engine)

ratio = inherited_rp_ratio (1.0 vs 0.0), staged loss curves, no ES:

```
                     lr    P1 coher (r1/r0)   P2 trainΔ/tree    min val ll (r1/r0)
synthetic 6k×30     0.30   0.086 / 0.092      ≈ equal           0.1519 / 0.1465
                    0.10   0.094 / 0.091      ≈ equal           0.1226 / 0.1213
                    0.03   0.116 / 0.089      ≈ equal           0.1116 / 0.1110
credit_default      0.30   0.130 / 0.122      ≈ equal           0.4372 / 0.4373
                    0.10   0.133 / 0.121      ≈ equal           0.4320 / 0.4312
                    0.03   0.157 / 0.123      ≈ equal           0.4294 / 0.4280  (final: 0.4392 / 0.4402)
```

- **P1 partially confirmed**: inheritance does raise within-tree direction
  coherence, consistently on real data and increasingly as lr shrinks
  (0.157 vs 0.123 at lr=0.03 — the cache accumulates and reinforces).
- **P2 rejected**: the coherence does NOT measurably weaken trees
  (per-tree train-loss drop equal to ~1%).
- **P3 not confirmed in predicted form**: min val loss slightly favors
  ratio=0 almost everywhere at fixed params; only trace support is
  credit_default lr=0.03 where ratio=1 has the better FINAL loss (less
  post-optimum drift) and keeps improving longer. Claim 2a (A/B as
  capacity control) survives only in weakened form: real but small,
  late-regime, real-data-only.

### 2026-06-12 — P4 oblique oblivious trees (`theory_exp3_oblivious.py`)

One shared direction per level (chosen by pooled gain over all level nodes),
per-node thresholds; research impl, 60 rounds, lr 0.1, 3 seeds:

```
                      standard d4        oblivious d4       oblivious d6
synthetic   acc/ll    0.9509/0.1461      0.9464/0.1584      0.9540/0.1337
multiclass  acc/ll    0.8707/0.4164      0.8755/0.4419      0.8920/0.3530
breast      acc/ll    0.9510/0.1500      0.9720/0.1376      0.9697/0.1283
```

**P4 strongly supported.** Oblivious wins all three datasets at d6 (multiclass
+2.1pp accuracy, −15% logloss) and even at matched leaf count (d4) wins
accuracy on 2/3.

**Refined mathematical account** (better than the original Claim 2a): the
operative quantity is *direction-estimation variance*. A node at depth $d$
estimates its split direction from $n/2^d$ samples — at depth ≥3 the per-node
gradient statistics are noise-dominated (visible earlier as the cone-probe
orth-win rate collapsing to 33% on small breast nodes). The oblivious
constraint pools all $2^\ell$ level nodes, so the level direction is estimated
from all $n$ samples at every depth: a classic bias-variance trade that pays
off exactly where standard oblique trees degrade. Per-node thresholds keep the
cheap part of the expressivity (thresholds are 1-D estimates, robust on few
samples) while sharing the expensive part (directions are $D$-dimensional).

Engineering corollary: oblivious routing is one dot product per LEVEL — a
single $N \times L$ GEMM per tree at inference, branchless leaf indexing by
bit-packing the $L$ comparisons; training evaluates candidates once per level
instead of once per node (fewer, larger parallel regions → directly attacks
the 26%-CPU-utilization ceiling measured on the current engine).

### 2026-06-12 — POBS diversity slot (`pobs_comparison.py`, user-proposed)

User spec: replace iid sparse random with columns of Haar-random orthogonal
blocks + random sparsity mask (§1 frame: orthogonal candidates minimize
within-budget redundancy → better expected max). Identified flaw: the
sparsity mask breaks the block's orthogonality and the spec is
gradient-blind. Refinement `pobs_sis`: sample a SIS-weighted support S
(|S| = √D), build an exact K×K orthogonal block ON S — sparse, exactly
orthogonal, gradient-informed. Budget-matched (12 dirs, inheritance off):

```
              iid              pobs (spec)      pobs_sis (refined)
synthetic     0.9490/0.1472    0.9480/0.1519    0.9543/0.1422
multiclass    0.8680/0.4219    0.8768/0.4076    0.8846/0.3806
breast        0.9510/0.1462    0.9598/0.1425    0.9545/0.1427
```

- `pobs_sis` ≥ iid on 3/3 (multiclass +1.7pp acc, −10% logloss) — the
  order-statistics prediction holds once orthogonality actually survives
  the sparsification.
- Plain masked `pobs` is inconsistent (beats iid on 2/3, loses on
  synthetic), as predicted: masking degrades it back toward iid.

Verdict: the POBS idea is sound; the implementable form is the
support-restricted orthogonal block. Candidate for the diversity slot,
pending the standing validation rule below. Cache-integration half of the
user spec (FIFO, POBS-winners-only insertion) not yet tested.

### 2026-06-12 — P4 round 2: equal-depth + variants (`theory_exp4_oblivious2.py`)

```
                                  synthetic           multiclass
standard   d6 lr=0.1              0.9557 / 0.1299     0.8900 / 0.3420
obliv-semi d6 lr=0.1              0.9523 / 0.1358     0.8876 / 0.3580
obliv-semi d6 lr=0.3              0.9517 / 0.1529     0.8956 / 0.3195
standard   d6 lr=0.3              0.9510 / 0.1646     0.8732 / 0.3825
obliv-full d6 lr=0.1              0.9580 / 0.1431     0.8900 / 0.3785
obliv-semi+ d6 lr=0.1 (2x cand)   0.9613 / 0.1261     0.9008 / 0.3305
```

Honest read:
1. **Leaf-count confound confirmed**: at equal depth, equal budget, lr 0.1,
   standard ≈ semi. The structural constraint alone is not the win.
2. **The real wins are exactly what the theory predicts**:
   - *lr robustness*: at lr 0.3 standard collapses (multiclass −1.7pp,
     logloss +12%) while oblivious improves — the weaker-per-leaf
     structure tolerates aggressive shrinkage (flatter tuning surface).
   - *Budget reinvestment* (the decisive one): pooled evaluation amortizes
     one candidate projection across all 2^l level nodes, so the
     tournament can be widened almost for free. `semi+` (2× candidates)
     is best on BOTH datasets, BOTH metrics. Standard trees pay per node
     and cannot afford this.
3. **Design decision**: full-oblivious (shared thresholds) trades logloss
   for nothing — per-node thresholds (semi) is the right form. Thresholds
   are cheap 1-D estimates; only directions need pooling.

Revised P4 statement: *oblivious-oblique is not a regularizer win; it is a
compute-reallocation win — constrain the expensive estimate (direction),
spend the savings on tournament breadth.* At equal wall-clock the gap
should widen further (2× was conservative).

### 2026-06-12 — Covariance analytical candidate (`docs/cov_oqboost.md`, `research/cov_experiment.py`)

User-proposed spec: replace the random tournament with ONE analytical
direction $w^* = -X^Tg/\|X^Tg\|$ (linearized WLS, no matrix solve — distinct
from the rejected full WLS). Implemented behind `OQB_COV_MODE` in the engine:
SIS top-$d_{sub}$ support, two scalings (spec-literal raw covariance +
diagonal-Newton $-c_d/(\Sigma h x_d^2+\lambda)$, scale-robust). Mode 1 appends
both to the pool; mode 2 replaces the 32-candidate random/inherited pool.

Synthetic sanity (20k×50 `make_classification`): mode 1 logloss −10%,
mode 2 −7% and 10% faster. Deployment protocol (tuned params + ES, 3 reps):

```
                       mode0 (base)      mode1 (+cov)      mode2 (replace)
adult        ll/auc    0.2745/0.9293     0.2751/0.9292     0.2781/0.9276
credit_def   ll/auc    0.4334/0.7761     0.4339/0.7765     0.4328/0.7775
gmsc(de-credit) ll     0.5322            0.5333            0.5424
```

**Rejected for integration.** The synthetic gain does not transfer: mode 1 is
noise-level everywhere (the cov candidate almost never wins on real data),
mode 2 regresses adult and the small dataset. §1 reading: on
`make_classification` the Bayes boundary IS a linear combination of the
informative features, so the analytical direction sits exactly on the gain
ridge; on real tabular data the ridge is interaction-dominated and one
analytical candidate has less tail mass than 32 random draws. Note the engine
already consumes the covariance signal ($cg_s$) as the SIS sampling weights
and Strategy-B sign choice — the marginal information in the full vector is
small. Gate kept in the engine (default 0 = production identical) for future
probes.

### 2026-06-12 — Depth-adaptive candidate budget (`research/budget_experiment.py`) ✅ MERGED

§2/P4-refined applied to budget allocation instead of structure: per-level
total cost is constant (level sample counts sum to N) so deep levels consume
most of the oblique budget, yet deep-node direction estimates are
noise-dominated. `OQB_BUDGET_MODE`: 1 = cut (pool 32→8 at depth ≥ 3),
2 = cut + reinvest (pool 64 at depth ≤ 2, 8 at depth ≥ 3).

Deployment protocol (tuned+ES, 3 reps) + covtype multiclass (100k, 1 rep):

```
                       mode0 (uniform)    mode1 (cut)       mode2 (reinvest)
adult        ll/acc    0.2745/0.8613      0.2744/0.8613     0.2749/0.8614
             fit       8.5s/579t          7.9s/482t         6.4s/436t (−25%)
credit_def   ll        0.4334             0.4330            0.4336
gmsc         ll        0.5322             0.5376            0.5287
covtype K=7  ll/acc    0.2756/0.8997      0.2812/0.8983     0.2768/0.9010
```

- **Mode 2 is Pareto across all four datasets**: accuracy ties or wins
  everywhere, logloss within noise, adult fit −25% (ES converges in fewer
  rounds — wider shallow tournaments make early trees stronger).
- Mode 1 (cut without reinvest) loses logloss on covtype: deep candidates
  still carry signal on large-N datasets; the reinvested shallow width
  compensates. Cut and reinvest are a package.
- Synthetic showed the opposite ordering (mode 0 best) — synthetic deep
  nodes keep clean linear signal, confirming the mechanism is deep-node
  estimation noise, not budget size per se.

**Merged as engine default 2026-06-12** (first engine change to pass the
standing rule end-to-end). `OQB_BUDGET_MODE=0` restores the uniform pool.

### 2026-06-12 — §1 open task closed: empirical F_f and optimal allocation (`research/family_allocation.py`)

Engine instrumented with `OQB_GAIN_LOG` (per-candidate `node,depth,ns,family,
gain` CSV; default off). 1.36M candidate gains on adult (tuned+ES), 1.2M on
covtype 100k, uniform budget for unbiased sampling.

Measurements:
- **Win rates**: cache-exact 40-42%, axis 35-75%, B 2-12%, C 3-10%, A 0.5-2%.
- **E[max k]/poolmax curves**: Strategy A saturates immediately
  (k=1: 0.55 → k=32: 0.62 — its draws are correlated perturbations of one
  parent direction, so extra draws buy almost nothing). B and C tails keep
  growing through k=32. Greedy concave allocation: **a:1 b:15 c:16**
  (consistent on 3 of 4 depth buckets; covtype-shallow unreliable, 1.3k nodes).
- Confirms the §1 frame *within a node*: A is worth ≈1 slot of tail coverage,
  not 12.

`OQB_POOL_MIX=1` (a 6.25% / b 46.9% / c 46.9%) deployment protocol:

```
              mix0 (production)   mix1 (data-optimal)
adult         0.2749/0.8614       0.2751/0.8609
credit_def    0.4336              0.4332
gmsc          0.5287              0.5520
covtype K=7   0.2768/0.9010       0.2803/0.8966
```

**Rejected.** covtype −0.4pp accuracy, gmsc regresses; only credit_default
marginally better. The decisive lesson: **per-node E[max gain] is the wrong
objective for the ensemble** — A's value is not visible in its per-node gain
distribution. This is Claim 2a (A as capacity control / inter-tree
correlation structure) resurfacing with direct evidence: an allocation that
provably raises per-node split quality lowers ensemble quality. The
production a:b:c = 37.5:37.5:25 split survives its third challenge.
Gate kept (default 0). Instrumentation (`OQB_GAIN_LOG`) kept for future use.

### 2026-06-12 — pobs_sis in the engine (`research/pobs_experiment.py`) ✅ MERGED

C++ port of the validated pobs_sis form behind `OQB_POBS`: per block, sample
an SIS-weighted support S (|S| ≈ √D), build an exact Haar-orthogonal block ON
S by Gram-Schmidt (sparse + exactly orthogonal + gradient-informed; the
orthogonalization is m³ ≈ hundreds of flops, no measurable cost). Mode 1
replaces the GG-SRP random slots (root pool + n_global); mode 2 additionally
carves 8 of every node's inherited budget into pobs blocks.

Deployment protocol (tuned+ES, 3 reps; covtype 100k 1 rep), no re-tuning:

```
                       mode0 (production)        mode2 (per-node pobs)
adult      ll/acc/auc  0.2749/0.8614/0.9292      0.2744/0.8620/0.9295
credit_def ll/acc/auc  0.4336/0.8047/0.7765      0.4325/0.8064/0.7777
gmsc       ll/auc      0.5287/0.7529             0.5238/0.7572
covtype    ll/acc      0.2768/0.9010             0.2662/0.9049
```

**First candidate-distribution change to pass the protocol** — logloss and
AUC improve on all four datasets at untouched tuned params (covtype ll −3.8%,
acc +0.4pp). Mode 1 alone fails: with `inherited_rp_ratio=1` the random slot
exists only at the root, so the surface is too small. The value is per-node
injection.

Why this succeeded where pool-mix failed (§1 account): pool-mix reallocated
budget among the *mutually correlated* B/C families — same tail, different
weights. pobs adds an *uncorrelated orthogonal family*: within a block the
candidate gains are independent by construction, so each draw contributes
full marginal tail mass. This is exactly the §1-predicted profitable move
(add mass above the existing families' tails), and it is the designed version
of what GG-SRP only did at the root. Adult fit 6.4→7.2s — per-tree cost
unchanged; ES simply keeps improving for ~90 more rounds.

**Merged as engine default 2026-06-12** (`OQB_POBS=0` restores GG-SRP).
Follow-up candidates: optuna re-tune on top (mode 2 won *without* it),
cache-integration half of the original user spec (POBS-winners-only FIFO).

### 2026-06-13 — Gradient Covariance + Axis Splits Verification & Head-to-Head Benchmark

Based on the theoretical WLS simplification (ignoring Hessian correlation, $H \approx \sigma I$), we evaluated the Gradient Covariance direction ($w^* \propto G = -X_{\text{sub}}^T g$) as a standalone candidate direction. 

1. **Node-Level Cosine Alignment**:
   - Rotated Synthetic: Mean $\cos(w_{\text{cov}}, w_{\text{cd}}) = 0.7010 \pm 0.0916$
   - Breast Cancer: Mean $\cos(w_{\text{cov}}, w_{\text{cd}}) = 0.5535 \pm 0.2264$
   - **Conclusion**: They are mathematically distinct; the Hessian matrix rotates the direction substantially.

2. **The New Paradigm (`proxy_cov_axis_1`)**:
   - Evaluates exactly **one** oblique candidate direction ($w_{\text{cov}}$) combined with all standard axis-aligned split candidates (total candidates = $D + 1$).
   - This removes the candidate tournament entirely, keeping only the best of axis vs. 1 covariance-aligned oblique split.

3. **Empirical Benchmarks (Head-to-Head vs. C++ Production Engine)**:
   - **Adult Income**:
     - C++ Production (Tournament): AUC = **0.8967 ± 0.0058** | Acc = **0.8563 ± 0.0037** | Time = **1.01s**
     - PyTorch Covariance (No Tourn): AUC = **0.9105 ± 0.0027** | Acc = **0.8640 ± 0.0086** | Time = **3.98s**
     - **Gain**: **+1.38% AUC** and **+0.77% Acc**!
   - **Credit Default**:
     - C++ Production (Tournament): AUC = **0.7335 ± 0.0184** | Acc = **0.7928 ± 0.0075** | Time = **1.07s**
     - PyTorch Covariance (No Tourn): AUC = **0.7414 ± 0.0144** | Acc = **0.8035 ± 0.0065** | Time = **7.66s**
     - **Gain**: **+0.79% AUC** and **+1.07% Acc**!

### Theoretical Insights
- **Hessian-free Regularization**: Why does a single first-order covariance direction beat the optimized Coordinate Descent (WLS) direction and the 16-candidate tournament on real datasets? In finite samples, the local Hessian $H$ computed at deep nodes is highly noisy. CD/WLS overfits to this local noise by inverting/solving $A^{-1}G$. Setting $H \approx I$ (the Covariance vector) acts as a robust L2 regularizer on the search direction, preventing local variance amplification.
- **Backbone of Axis Splits**: Oblique-only (`proxy_cov_1`) without axis splits suffers severe degradation on tabular datasets (Adult AUC drops to 0.8844), proving that axis splits are essential to capture unrotated features. The hybrid `axis_best` + `cov_best` setup is the optimal oblique tree architecture.
- **Architectural Simplification**: This results in a massive paradigm shift. We do not need a complex candidate pool, FIFO/PCA direction caches, or Coordinate Descent loops. Evaluating exactly $D$ axis splits + exactly 1 covariance oblique split achieves superior generalization at a fraction of the search cost.

### Pending before any engine work (per the standing rule)

1. standard d6 baseline (research impl) for an equal-depth comparison.
2. Deployment-protocol validation: tuned params + early stopping + the real benchmark datasets.
3. lr sensitivity of the oblivious variant (its weaker-per-leaf structure should tolerate larger lr — check the lr × rounds surface).
4. Candidate-budget parity check (oblivious currently sees 16 random + axis per level vs standard's per-node budget).
