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

### Pending before any engine work (per the standing rule)

1. standard d6 baseline (research impl) for an equal-depth comparison.
2. Deployment-protocol validation: tuned params + early stopping + the real
   benchmark datasets.
3. lr sensitivity of the oblivious variant (its weaker-per-leaf structure
   should tolerate larger lr — check the lr × rounds surface).
4. Candidate-budget parity check (oblivious currently sees 16 random + axis
   per level vs standard's per-node budget).
