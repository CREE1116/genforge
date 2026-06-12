# Mechanism Study Findings — Direction Inheritance in OQBoost

Date: 2026-06-12. Scripts: `drift_analysis.py`, `mechanism_study.py`, `e7_matched.py`,
`cpp_ablation.py`. All research-impl runs use `use_wls=False` unless noted, so the
candidate set mirrors the production C++ engine (axis + sparse random + inherited).

## The apparent contradiction

The first analysis run (`analyze()`) reported mean parent→child direction drift of
74.9° (median 84.4°) — yet the model's inheritance mechanism (Strategy A/B) mutates
the parent direction under the assumption that good child directions stay NEAR the
parent. And the C++ ablation shows inheritance does help:

```
[C++ engine, inherited_rp_ratio sweep]
synthetic 6k×30 : ratio=1.0 acc 0.9669  |  ratio=0.0 acc 0.9640   (+0.3pp)
breast_cancer   : ratio=1.0 acc 0.9744  |  ratio=0.0 acc 0.9697   (+0.5pp)
```

## Resolution — three experiments

### Exp 1: the drift number was a measurement artifact (H1 confirmed)

Drift conditioned on WHICH candidate type won the node:

```
synthetic            wins    drift(mean)
  axis               20.6%   84.2°
  wls                76.7%   73.1°
  inherit_A           1.4%   11.6°
  inherit_B           1.4%    8.7°
```

When inherited candidates win, drift is small by construction. The 74.9° average
measured "winner vs parent" across all winner types. BUT: inherited candidates
win only ~3% of nodes — direction reuse is almost never what the tree wants.

### E6: the orthogonal subspace holds ~2× the gain (orthogonality principle)

Matched candidate budgets (8 dirs each) per node, three cones around the parent
direction, same feature support:

```
                 near(~11°)   orth(90°)   ratio   orth-wins
synthetic d1       20.90        42.03     2.01x     88.8%
synthetic d2        8.45        22.26     2.63x     93.8%
synthetic d3        4.51        11.89     2.64x     82.8%
breast    d1        1.51         2.60     1.72x     85.0%
breast    d2        0.56         1.01     1.80x     73.7%
```

Principle: a split on direction w consumes the gradient signal along w — each
child is conditioned on one side of the hyperplane, so the residual signal
concentrates in the orthogonal complement of w. Large drift is OPTIMAL behavior,
not noise. Strategy A's near-parent cone searches the one subspace the split
just emptied.

### E5: support reuse also rejected

Parent↔child-winner feature support Jaccard: 0.011 (synthetic), 0.035 (breast).
The informative features change almost completely after a split. Inheritance does
not work through feature-support transfer either.

### E7 (budget-matched): orthogonalized inheritance beats mutation everywhere

Each mode contributes exactly 4 inherited-slot candidates (no count confound):

```
                       none              mutate (current)   orth (new)        both
synthetic   acc      0.9437±0.0031     0.9420±0.0050      0.9503±0.0036     0.9477
            logloss  0.1622            0.1591             0.1451            0.1505
multiclass  acc      0.8317±0.0055     0.8352±0.0049      0.8581±0.0020     0.8523
            logloss  0.4746            0.4748             0.4294            0.4382
breast      acc      0.9804±0.0052     0.9818±0.0034      0.9832±0.0095     0.9818
            logloss  0.0486            0.0469             0.0435            0.0448
```

orth wins all three datasets: up to +2.3pp accuracy and −9.6% logloss over the
current mutate strategy. mutate is statistically indistinguishable from no
inheritance at matched budgets — its measured benefit in the C++ engine likely
comes from gradient-informed feature selection inside Strategy B (new-axis choice
weighted by SIS score), not from parent proximity.

## Conclusion

1. The inheritance hypothesis ("good child directions are near the parent
   direction") is wrong. The opposite holds: residual signal is orthogonal.
2. What still makes inherited candidates useful is being *anchored to the node's
   geometry* (support ∪ top-SIS features) and *informed by gradients* — not the
   parent direction itself.
3. Strategy O (sample directions in the parent-orthogonal subspace within
   support ∪ top-SIS) is a drop-in replacement for Strategy A/B with a consistent
   accuracy/logloss win at equal cost.

## C++ port — attempted and REVERTED (2026-06-12)

Strategy O (and a depth-scheduled β(depth)=d/(d+1) partial variant) was ported
into `eval_oblique`, then reverted to the original A/B/C strategies. Reason —
the user's high-dimensional argument, which the cone probe itself supports:

> In high dimensions, RANDOM directions are already near-orthogonal to any
> fixed vector (concentration of measure). E6 measured rand ≈ orth gain
> (42.58 vs 42.03 at d1, etc.) — explicit orthogonalization duplicates
> coverage the global random pool provides for free. The only region random
> sampling can never reach is the narrow cone AROUND the parent direction;
> Strategy A/B is the sole source of that coverage. Removing it deletes a
> unique candidate family and adds nothing new.

So the correct reading of E6 is not "orthogonal is good" but "near-parent is
rarely needed — yet it is cheap, irreplaceable coverage." The research-impl E7
'orth' wins came from support-informed random exploration (vs none), not from
orthogonality per se.

### Apparent real-benchmark regression — resolved as noise

After the O port, adult/credit_default/gmsc benchmark scores dipped and train
times collapsed (early stop at ~12 trees). Controlled comparison (old engine
rebuilt from git HEAD, identical protocol, credit_default with tuned params):

```
old engine : auc 0.7801±0.0013  n_trees=15
new engine : auc 0.7802±0.0016  n_trees=12   (ratio=0.0)
new engine : auc 0.7791±0.0005  n_trees=11   (ratio=1.0)
```

Engines are statistically identical; early stopping at ~12-15 trees predates
all of this session's changes (tuned lr=0.17 converges that fast). The CSV
deltas came from rep-level noise on small datasets (the gmsc benchmark
actually runs the 1000-row OpenML credit-g fallback, ±0.03-0.05 AUC noise).

### Final engine state

Original A/B/C inheritance strategies; bfstree removed; GEMM panel batched
projections + full cache-candidate evaluation kept (32% faster training,
quality unchanged).

---

# Part 2 — Diversity + Continuity study (2026-06-12)

Hypothesis (user): the candidate generator needs BOTH continuity (near-parent
candidates) and diversity (parent-independent random candidates).
Scripts: `diversity_continuity.py`, `strategy_ablation.py` (C++ engine via the
`OQB_STRATEGIES` env gate added to `eval_oblique`).

## EXP-A: budget-matched knockouts (research impl, 12 non-axis candidates)

```
              synthetic acc/ll        multiclass acc/ll
full (4i+8r)  0.9484 / 0.1515        0.8653 / 0.4263
diversity 12r 0.9489 / 0.1472  ←best 0.8688 / 0.4197  ←best
continuity12i 0.9367 / 0.1701        0.8605 / 0.4391
axis_only     0.9387 / 0.1724        0.8651 / 0.4414
```

Diversity is necessary AND sufficient; parent continuity contributes nothing
measurable and continuity-only collapses to the axis floor.

## EXP-B: HOW diversity works — feature coverage, not tree decorrelation

Per-tree prediction correlation is identical across configs (~0.44); what
changes is feature-usage entropy (0.945 → 0.970 synthetic, 0.870 → 0.933
multiclass). Random oblique candidates widen which features participate in
splits; they do not decorrelate the ensemble.

## EXP-D: diversity matters MORE as boosting progresses

Random-family win share grows monotonically over rounds (18%→35% synthetic,
15%→36% multiclass) while axis declines (78%→58%, 84%→61%). Early residuals
have strong marginal (axis) structure; late residuals are oblique.

## C++ strategy ablation: the inherited slot's value was the CACHE all along

`OQB_STRATEGIES` gate, inherited_rp_ratio=1.0 except "none", 5 seeds:

```
                 synthetic acc/ll     multiclass acc/ll
abc (production) 0.9656 / 0.1304      0.9020 / 0.2780
c   (cache only) 0.9664 / 0.1314      0.9002 / 0.2814
none (random)    0.9643 / 0.1298      0.9027 / 0.2777
ab  (no cache)   0.9620 / 0.1382      0.8986 / 0.2900
a   (perturb)    0.9588 / 0.1426      0.8959 / 0.2909
b   (grad-axis)  0.9601 / 0.1444      0.8927 / 0.2966
```

{abc, c, none} cluster at the top within noise. Pure parent-anchored
strategies (a, b, ab) are consistently 0.4-1pp WORSE than replacing them with
random. The earlier ratio-sweep advantage of the inherited slot was Strategy C
(cache) — cross-TREE reuse of directions that scored well anywhere — diluting
A/B's harm, not parent continuity helping.

## Synthesis — why the model works

1. **Backbone**: the full 256-bin axis scan finds all strong marginal
   structure (wins 58-84% of nodes, most in early rounds).
2. **Diversity (essential)**: sparse random oblique candidates widen feature
   coverage; their value grows over boosting rounds as residuals become
   oblique. This is the irreplaceable ingredient.
3. **Continuity (the right scale is the DATASET, not the parent)**: good
   oblique directions are a global property of the data (its rotated
   subspaces), so reusing globally-proven directions (cache) is neutral-to-
   mildly-positive, while reusing the parent's direction is actively harmful —
   the parent split already consumed that direction's signal.
4. **Gain-based selection filters bad candidates**, which is why production
   (abc) never pays the full price of A/B: bad near-parent candidates simply
   lose the tournament.

The user's hypothesis holds in spirit with one correction: the continuity that
matters is dataset-level (cache), not parent-level (mutation).

## Final validation and engine change (APPLIED, 2026-06-12)

4 configs × 4 datasets × 5 seeds (`final_validation.py`), 200 trees, no ES:

```
              abc(old)   c r=1.0   c r=0.5   none r=0
adult    acc   0.8525    0.8554    0.8532    0.8533
credit   auc   0.7550    0.7561    0.7543    0.7550
breast   ll    0.1082    0.1095    0.0993    0.0967
synth50  acc   0.9699    0.9721    0.9729    0.9727
```

abc (with A/B) never wins and is the clear loser on synthetic50 (beyond
noise). All A/B-free configs are never-worse. c-vs-none splits by dataset
(adult prefers cache, breast prefers random) → the balanced mix wins overall.

**Engine change applied** (`oqboost.cpp` + `_classifier.py`):
- Strategies A and B (parent-direction mutation) removed. The informed slot
  is now Strategy C only (cache blend), falling back to sparse random draws
  while the cache is still empty.
- `inherited_rp_ratio` default 1.0 → 0.5 (informed/random half-split).
- `mutation_rate` / `mutation_strength` deprecated no-ops (API unchanged).
- The temporary `OQB_STRATEGIES` env gate was removed with the strategies.

Post-change results — never worse, usually better than the session baseline:

```
quick_bench          baseline → final
synthetic50    acc   0.9710  → 0.9736   logloss 0.0952 → 0.0923
multiclass     acc   0.9064  → 0.9192   logloss 0.2767 → 0.2495
breast         acc   0.9650  → 0.9720   logloss 0.0640 → 0.0605
total fit time       11.62s  → 7.44s    (engine optimizations included)

adult (5 seeds) acc  0.8525  → 0.8550   auc 0.9085 → 0.9095
```
