# HypForge — Algorithm Reference

## Overview

HypForge (Hypothesis Pool Forge) is a gradient-boosted decision tree (GBDT) variant that uses **oblique splits** — each tree node splits on a learned linear combination of features rather than a single axis-aligned feature. The split directions are drawn from an evolving pool of *hypotheses* managed by a UCB (Upper Confidence Bound) bandit algorithm.

The core idea: instead of searching over individual features at each split, HypForge maintains a shared pool of projection directions that are continuously refined by gradient information. Better directions survive, weak directions are pruned, and new directions are proposed by following the gradient.

---

## Key Concepts

### Hypothesis

A **Hypothesis** `h` defines one oblique split direction:

| Type | Projection |
|---|---|
| `linear` | `X @ w` |
| `square` | `(X @ w)²` |
| `abs` | `\|X @ w\|` |

`w` is a sparse unit-norm weight vector with at most `k = sqrt(D_num)` non-zero entries. The `square` and `abs` types apply a pointwise nonlinearity after projection, letting the pool represent non-linear feature interactions without multiplying features together.

Each hypothesis also carries:
- **UCB statistics**: `n_obs`, `mu_fitness`, `M2_fitness` (Welford online mean/variance)
- **Usage stats**: `use_count`, `rounds_since_last_use`
- **Projection caches**: `full_cache_cpu`, `thresholds_cpu` (used during training only; cleared before save)

### Pool

The **HypForgePool** is initialized with D axis-aligned hypotheses (one per feature), marked `is_base = True` and immortal. During training, the pool grows by evolving new hypotheses and shrinks via diversity pruning and survival pressure.

### UCB Scoring

Each hypothesis is scored by a UCB1-style formula after every round:

```
score = μ + (C_UCB × σ + C_NOVEL) / √n_obs − η × complexity
```

- `μ` — Welford online mean of observed split-gain fitness values
- `σ` — Welford online standard deviation (zero until n_obs ≥ 2)
- `C_UCB = 1.0` — confidence interval width; rewards hypotheses with high variance (exploration)
- `C_NOVEL = 0.5` — novelty bonus that decays as 1/√n_obs; newly admitted hypotheses start with a boost
- `η = 0.002` — complexity penalty; discourages dense projection vectors
- `complexity` — number of non-zero weights in `w`

Hypotheses with more observations converge toward their true mean minus the complexity penalty. New hypotheses start with a high novelty bonus and are given a fair chance before being judged.

---

## Training Algorithm

### Initialization

```
pool ← D axis-aligned base hypotheses (one per feature, immortal)
F    ← log-prior scores (balanced class log-probabilities)
```

### Per-Round Loop (for m = 1 … n_estimators)

**1. Compute gradients (softmax cross-entropy)**

```
P  = softmax(F)
G  = sample_weight ⊙ (P − one_hot(y))      # shape [N, K]
H  = sample_weight ⊙ P ⊙ (1 − P)           # shape [N, K]
```

**2. Evolve the pool** (every `evolve_every` rounds, on a random subsample of ≤10,000 rows)

See [Pool Evolution](#pool-evolution) below.

**3. Build tree (C++)**

```
Z          = stack(h.full_cache_cpu for h in pool)   # [P, N]
thresholds = stack(h.thresholds_cpu for h in pool)   # [9, P]

tree.build(Z, thresholds, G[sub], H[sub], max_depth, reg_lambda)
```

The C++ BFS tree stores integer indices into `pool.pop`. Each node selects the hypothesis index and threshold that maximizes the XGBoost-style split gain:

```
gain = 0.5 × (G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ))
```

**4. Update scores**

```
F ← F + learning_rate × tree.predict(Z)
pool.pop[idx].use_count++ for each split index idx in tree
```

**5. Validation & early stopping**

If `eval_set` is provided, validate and track `best_val_loss`. On improvement, snapshot `best_trees`, `best_pool_snaps`, `best_pool`. If `no_improv ≥ early_stopping_rounds`, restore best state and halt.

---

## Pool Evolution

`pool.evolve()` runs every round on a random subsample of rows:

### Step 1 — Observe fitness & UCB update

Compute split-gain fitness for every current hypothesis on the subsample, then run Welford online update:

```python
delta          = fit_val − mu
mu            += delta / n_obs
M2            += delta × (fit_val − mu)   # after updated mu
sigma          = sqrt(M2 / n_obs)
score          = mu + (C_UCB×sigma + C_NOVEL)/sqrt(n_obs) − η×complexity
```

Sort pool by UCB score (descending).

### Step 2 — Generate candidates

**Gradient flow** — one direction per output class:

```
V   = X_sub.T @ G_sub                  # [D, K]
w_k = sparsify(V[:, k])  for k in 0..K-1
```

Emit 3 hypotheses per direction: `linear`, `square`, `abs` — totalling 3K candidates.

**Gradient ascent** — refine top-20 pool members:

```python
for h in pool[:20]:
    p      = X_sub @ h.w
    v      = G_sub.T @ h.cache          # gradient-aligned score
    g      = G_sub @ v
    # type-specific gradient w.r.t. w:
    if linear:  grad_w = X_sub.T @ (g − z)
    if square:  grad_w = 2 × X_sub.T @ ((g − z) × p)
    if abs:     grad_w = X_sub.T @ ((g − z) × sign(p))
    emit Hypothesis(h.hyp_type, w=sparsify(h.w + 0.1×grad_w))
```

### Step 3 — Score candidates & admit

Compute split-gain for all candidates in a single batched call. Admit those with `fit_val > 1e-5`, initialize their projections via a single batched matmul:

```
W_new  = stack([c.w for c in admitted])   # [D, M]
projs  = X_full @ W_new                   # [N, M] — one matmul for all M
```

Compute quantile thresholds from `projs` per hypothesis.

### Step 4 — Diversity pruning (type-aware cosine)

Compute the absolute cosine similarity matrix over sub-indexed projections. Greedy selection: keep the highest-UCB hypothesis first; skip any later hypothesis that is too similar to an already-kept one.

```
threshold = 0.90  if same type
threshold = 0.98  if different type
```

`linear` vs `square` on the same `w` are genuinely distinct (x vs x²), so the looser threshold applies across types.

### Step 5 — Survival filter

Base hypotheses (`is_base = True`) are immortal. Others survive if:

```
h.n_obs < 3                        # too new to judge
OR
h.mu_fitness + h.score > 1e-6      # UCB upper bound still meaningful
```

Low-fitness hypotheses are naturally displaced by the cap enforced in Step 4 before this filter runs.

---

## Inference (`predict_proba`)

```python
F = tile(F_init, (N, 1))

eval_cache = {}   # deduplication: same Hypothesis object → computed once
for tree, snap in zip(trees_, pool_snaps_):
    Z_rows = [eval_cache.setdefault(id(h), h.eval(X)) for h in snap]
    Z      = stack(Z_rows)           # [P_snap, N]
    F     += learning_rate × tree.predict(Z)

return softmax(F)
```

Each tree is paired with its own pool snapshot (`pool_snaps_`) because the C++ tree stores *integer indices* into the pool as it existed at that round. Using the wrong pool (e.g. the final pool for all trees) would map indices to wrong hypotheses.

---

## sparsify

```python
k = max(2, int(sqrt(D_num)))     # keep at most sqrt(D) non-zero weights
w = zero out all but top-k by |w|
w = w / ||w||
```

Sparse projections reduce complexity, improve interpretability, and provide a natural regularization that prevents the pool from collapsing to dense dense dense directions.

---

## Performance (SDSS Stellar Classification, N=577k, K=3)

| Model | Val Acc | Balanced Acc | Train Time |
|---|---|---|---|
| LightGBM | 0.9664 | 0.9536 | 23s |
| XGBoost | 0.9649 | 0.9508 | 30s |
| CatBoost | 0.9589 | 0.9409 | 56s |
| **HypForge** | **~0.956** | **~0.960** | ~28min |
| HypForge (pre-bugfix) | 0.9439 | 0.9454 | — |

HypForge achieves a **higher balanced accuracy** than all axis-aligned competitors (including XGBoost and LightGBM) because oblique splits learn to separate minority classes (QSO, STAR) more precisely. On axis-aligned features the majority class (GALAXY, 65%) dominates XGBoost's splits. On a 2D spiral dataset — where the optimal boundary is oblique — HypForge's boundaries are visibly smoother and more class-aligned than XGBoost/LightGBM/CatBoost.

The main current weakness is **training time**: Python-level per-round pool management + large dataset → ~28 minutes vs 30 seconds for XGBoost.

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `n_estimators` | 500 | Max boosting rounds (use with early stopping) |
| `learning_rate` (η) | 0.1 | Shrinkage; lower → more trees needed, better generalization |
| `max_depth` | 6 | Tree depth; 6 → 127 nodes |
| `reg_lambda` (λ) | 1.0 | L2 regularization on leaf weights |
| `pool_size` | 150 | Max hypotheses in pool; empirically saturates at 80–90 |
| `n_hyp_leaf` | 50 | Not currently used at tree level (all pool members evaluated) |
| `evolve_every` | 1 | Evolve pool every N rounds |
| `early_stopping_rounds` | 30 | Patience |
| `subsample` | 0.8 | Row fraction per tree |

---

## Architecture Notes

### Why per-tree pool snapshots?

The C++ BFS tree stores **integer indices** into `pool.pop` at build time. If the pool grows or reorders between round 1 and round 300, index 5 at round 1 refers to a different hypothesis than index 5 at round 300. Each tree must be paired with the exact pool state at its build time — stored in `pool_snaps_`.

### Why Welford online variance?

Welford's algorithm computes running mean and variance in O(1) memory and O(1) time per observation with numerically stable updates. No forgetting factor (like EMA alpha) is required; the algorithm is parameter-free once C_UCB and C_NOVEL are set.

### Why type-aware cosine thresholds?

`linear(w)` and `square(w)` on the same direction `w` are functionally distinct — one produces a symmetric parabola, the other a hyperplane. A looser threshold across types (0.98) allows both to coexist even when `w` is nearly identical. Same-type hypotheses with nearly identical `w` (cosine > 0.90) genuinely duplicate each other.

### Why no product hypotheses?

Early experiments with product hypotheses `h1.eval(X) × h2.eval(X)` showed they dominated the pool (84% of slots) and caused class imbalance in predictions. Gradient flow + gradient ascent with square/abs types captures the same nonlinearity with better diversity control.

---

## Planned Improvements

1. **C++ rewrite of `evolve()`** — the bottleneck is Python-level pool management. The algorithm is now simple enough (two candidate sources, one UCB formula, one diversity prune) that a full C++ port would reduce per-round cost by 10–50×.

2. **Fitness normalization** — split-gain values are non-stationary (large in early rounds, small in late rounds). Dividing by the per-round mean fitness before Welford update would make UCB scores comparable across rounds and hypotheses.

3. **More gradient flow directions** — currently K directions (one per class). Using top-M singular vectors of `G` would generate more diverse candidates without adding tunable parameters.

4. **XGBoost pipeline** — the final hypothesis pool can be exported to XGBoost as engineered features: `clf.embed(X)` produces an `[N, P]` matrix of learned projections that a downstream XGBoost can use.