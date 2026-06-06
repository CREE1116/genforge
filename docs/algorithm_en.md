# HypForge — Algorithm Reference (English)

## Overview

HypForge (Hypothesis Pool Forge) is a gradient-boosted decision tree framework that uses **oblique splits**: each tree node splits on a learned linear combination of features rather than a single axis. Split directions are drawn from an evolving pool of *hypotheses* managed by a UCB (Upper Confidence Bound) bandit.

Core idea: instead of searching over individual features at every split, HypForge maintains a shared pool of projection directions that are continuously refined through crossover breeding and fitness pressure. Better directions survive; redundant or weak directions are pruned; new directions are proposed by blending the best existing directions.

---

## Hypothesis Types

A **Hypothesis** `h` defines one oblique split direction via a sparse unit-norm weight vector `w ∈ ℝᴰ`:

| Type | ID | Projection | Notes |
|------|----|-----------|-------|
| `linear` | 0 | `X @ w` | Standard oblique split |
| `leaky_relu` | 1 | `max(X@w, 0.01·X@w)` | Asymmetric; emphasizes positive projections |
| `product` | 2 | `h₁(X) · h₂(X)` | Composite of two existing hypotheses |

Ablation result: using both `linear` and `leaky_relu` (op_mode="all") outperforms either type alone. `product` hypotheses are supported but rarely survive diversity pruning against the two primary types.

Each hypothesis carries:
- **UCB statistics**: `n_obs`, `mu_fitness`, `M2_fitness` (Welford online mean/variance)
- **Usage stats**: `use_count`, `rounds_since_last_use`
- **Projection cache**: `full_cache[N]` (training only; cleared before save), `thresholds[9]`
- **Genealogy & Lineage**:
  - `parent1`, `parent2`: global history IDs of parent hypotheses (crossover or component components, -1 if base/gradient)
  - `birth_round`: boosting round index at which this hypothesis was born
  - `family_id`: unique lineage identifier (inherited from fitter parent during crossover/product)
  - `family_fitness`: average fitness of all descendant hypotheses
  - `breeding_value`: average fitness of all direct child hypotheses

Sparse weights: at most `k = max(2, floor(√D_num))` non-zero entries, ℓ2-normalised.

---

## Pool Architecture

The **HypForgePool** is initialised with `D` axis-aligned base hypotheses (one per feature), marked `is_base=True` and immortal. During training, the pool grows by crossover breeding and shrinks via diversity dedup, elitism, MAP-Elites type quota, and survival pressure.

### Confirmed Defaults (ablation-validated)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `pool_size` | 400 | Larger candidate buffer → better diversity selection |
| `map_elites_slots` | 100 | Enforces 50/50 linear/lrelu balance (with pool_size=400) |
| `elitism_k` | 20 | Top-20 immune to similarity eviction |
| `alps_mode` | True | Young hypotheses (n_obs < 5) skip eviction |
| `crossover_top_k` | 3 | Top-3 pairs breed; beyond 3 dilutes quality |
| `subsample` | 0.8 | Row subsampling per tree; regularisation benefit |
| `max_depth` | 4 | Sweet spot; depth 5+ overfits |
| `family_max_size` | 30 | Lineage-based MAP-Elites quota; prevents single-family dominance |
| `meta_evolution` | True | Self-adaptive online UCB bandit selection among 4 candidate policies |
| `family_lambda` | 0.1 | Weight of family fitness bonus in final UCB score |
| `breeding_beta` | 0.3 | Weight of breeding value in parent selection score |

Parameters hardcoded after ablation (no user exposure needed):

| Parameter | Fixed Value | Why |
|-----------|-------------|-----|
| `mutation_mode` | none | Gradient/random mutation consistently harmful |
| `feedback_mode` | scan_only | Split-gain re-observation adds overhead without benefit |
| `fitness_norm_mode` | none | UCB exploration bonus already handles scale variation |
| `recency_bonus_rounds` | 0 | Protects stale hypotheses; hurts diversity |
| `use_rate_bonus` | 0.0 | Confirmed harmful across all lambda values |
| `novelty_lambda` | 0.0 | Novelty re-scoring post-dedup is harmful |
| `evolve_mode` | crossover | Standard and novelty modes both underperform |

---

## UCB Scoring

After every evolve round, each hypothesis is scored:

```
score = ucb_score + family_lambda × family_fitness
```
where `ucb_score` is computed as:
```
ucb_score = μ_fitness + (σ_fitness + 0.5) / √n_obs − η · complexity
```

- `μ_fitness` — Welford online mean of observed split-gain values
- `σ_fitness` — Welford online standard deviation (zero until n_obs ≥ 2)
- `0.5` — novelty constant that decays as 1/√n_obs; gives new hypotheses an initial boost
- `η = 0.002` — complexity penalty
- `complexity` — number of non-zero entries in `w`
- `family_fitness` — average fitness of all descendant hypotheses, which rewards propagating high-quality genetic lines

The pool is sorted descending by score before diversity pruning and crossover selection.

---

## Training Loop

### Initialisation

```
pool  ← D axis-aligned base hypotheses (immortal)
F     ← log-prior scores (balanced class log-probabilities), shape [N, K]
```

### Per-Round (m = 1 … n_estimators)

**1. Gradients (softmax cross-entropy)**

```
P = softmax(F)
G = sample_weight ⊙ (P − one_hot(y))    # [N, K]
H = sample_weight ⊙ P ⊙ (1 − P)        # [N, K]
```

**2. Evolve pool** (every `evolve_every` rounds, on a random subsample ≤ 10,000 rows)

See [Pool Evolution](#pool-evolution) below.

**3. Build oblique tree (C++)**

```
Z          = pool.eval(X)          # [P, N] — all hypothesis projections
thresholds = pool.thresholds       # [9, P] — precomputed quantile thresholds

tree.bfs_build(Z, thresholds, G[sub], H[sub], max_depth, reg_lambda)
```

Each node selects the (hypothesis, threshold) pair maximising XGBoost-style gain:

```
gain = 0.5 × (G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ))
```

**4. Update**

```
F += learning_rate × tree.predict(Z)
pool.increment_use_counts(tree.split_indices)
```

**5. Validation & early stopping**

If `eval_set` provided: compute val loss, track `best_val_loss`. On improvement, snapshot `(best_trees, best_pool_snaps)`. Stop if `no_improv ≥ early_stopping_rounds`, restore best snapshot.

---

## Pool Evolution

`pool.evolve()` runs on a random subsample of rows each round.

### Step 1 — Fitness scan & UCB update

For every hypothesis in the pool, compute split-gain fitness on the subsample (feedback_mode="scan_only"), then apply Welford update. Sync stats to the global history DAG, compute genealogy metrics (family fitness as average descendant fitness, breeding value as average direct child fitness) over all history elements, and set:
```
score = ucb_score + family_lambda × family_fitness
```

Sort pool descending by score.

### Step 2 — Crossover

1. Sort active hypotheses by parent selection score: `parent_selection_score = fitness + breeding_beta * breeding_value`.
2. Extract the top `crossover_top_k` parents.
3. Form all possible parent pairs and compute crossover selection weights proportional to their historical combination survival rate ( Laplace smoothed):
```
weight(Ta, Tb) = (survivors[Ta][Tb] + 1) / (births[Ta][Tb] + 2)
```
4. Sample pairs via Roulette Wheel Selection based on these weights. For each sampled pair:
```
α         ~ Uniform(0.3, 0.7)
w_child   = α × wᵢ + (1−α) × wⱼ
w_child   = sparsify(w_child)
type      = type of higher-scoring parent
family_id = family_id of higher-scoring parent
```

### Step 3 — Candidate fitness scan & admission

Compute split-gain for each candidate on the subsample. Admit those with `gain > 1e-5` and append them to the global history (assigning a new `family_id = global_id` if gradient-aligned). Cache their full projections: `full_cache[N] = apply_op(X @ w, type)`.

### Step 4 — Re-score & sort

Recompute scores (UCB score + family bonus) for all hypotheses, sort descending.

### Step 5 — ALPS priority boost

If `alps_mode=True`: temporarily add `+1e6` to the score of any hypothesis with `n_obs < 5` before sorting for eviction. Score boost is removed after re-sort.

### Step 6 — Diversity dedup & Family Quota

Greedy selection on sub-indexed projections. Keep the highest-scoring hypothesis. For each subsequent candidate, verify family quota and cosine similarity:
1. **Family Quota**: skip if the number of kept hypotheses with the same `family_id` already reaches `family_max_size` (default 30).
2. **Cosine Similarity**: compute absolute cosine similarity to already-kept ones. Skip if it exceeds `0.90` (same type) or `0.98` (different type).

Stop when `|kept| = pool_size (400)`.

### Step 7 — Elitism

Top `elitism_k = 20` hypotheses by score are immune to similarity and family quota eviction.

### Step 8 — MAP-Elites type quota

After dedup, apply per-type slot limit: at most `map_elites_slots = 100` hypotheses of each type.

### Step 9 — Survival filter & Transition Update

Remove non-base hypotheses with `n_obs ≥ 3` and `mu_fitness + score ≤ 1e-6`. Update the transition matrix births and survivors based on crossover candidates that were admitted in this round and whether they survived eviction.


---

## Inference

```python
F = tile(log_prior, (N, 1))

for tree, pool_snap in zip(trees_, pool_snaps_):
    Z  = pool_snap.eval(X)         # [P_snap, N]
    F += learning_rate × tree.predict(Z)

return softmax(F)
```

Each tree is paired with its own pool snapshot because C++ trees store *integer indices* into the pool as it existed at build time. Using the final pool for all trees would map indices to wrong hypotheses.

---

## C++ Interface

All pool operations run in `bfstree.cpp` (C++17, OpenMP). Python calls via ctypes (`_pool.py`).

| Export | Purpose |
|--------|---------|
| `pool_create(D, max_size, evolve_mode)` | Allocate pool with D base hypotheses |
| `pool_set_options(…)` | Set pool configuration options (op_mode, crossover_top_k, elitism_k, alps_mode, map_elites_slots, family_max_size, enable_meta_evolution, family_lambda, breeding_beta) |
| `pool_evolve(…)` | Full evolve pipeline |
| `pool_eval(…)` | Batch Z = apply all hypotheses to X |
| `pool_get_caches_and_thresholds(…)` | Return precomputed projections and thresholds for tree build |
| `pool_update_use_counts(…)` | Increment use_count for selected split indices |
| `pool_export(…)` | Serialize pool state to numpy arrays (including parent/family stats) |
| `pool_import(…)` | Deserialize pool state from numpy arrays |
| `pool_get_policy_stats(…)` | Retrieve Meta-Evolution bandit telemetry |
| `pool_get_transition_matrix(…)` | Retrieve crossover operator combination success rates |
| `pool_free(handle)` | Release C++ pool memory |

---

## Performance

Ablation benchmark (3 datasets: spiral-2D, checker-2D, hiD-20D; 3 seeds each):

| Configuration | Avg Accuracy |
|--------------|-------------|
| Baseline (standard mode, no pool features) | 0.9309 |
| + crossover (top_k=3) | 0.9354 |
| + elitism_k=20 | 0.9358 |
| + alps_mode=True | 0.9361 |
| + map_elites_slots=100 | 0.9364 |
| + subsample=0.8 | 0.9376 |
| + pool_size=400 | **0.9394** |

Final comparison (same 3 datasets, n_estimators=300):

| Model | Avg Accuracy |
|-------|-------------|
| **HypForge** | **0.9338** |
| XGBoost | 0.9318 |
| LightGBM | 0.9308 |
| ExtraTrees | 0.9306 |
| GradBoost-sklearn | 0.9282 |
| RandomForest | 0.9236 |

HypForge leads on high-dimensional data (hiD-20D: 0.9160 vs XGBoost 0.8967) where oblique splits uncover linear combinations of features that axis-aligned splits cannot.

---

## Planned Improvements

See separate design documents:

- [`plan_contribution_based_survival_en.md`](plan_contribution_based_survival_en.md) — Track per-hypothesis validation gain; penalise overfitting directions in UCB scoring.
- [`plan_latent_pool_en.md`](plan_latent_pool_en.md) — Separate primitive (base) and compound (crossover/product) hypothesis pools for independent evolution.
