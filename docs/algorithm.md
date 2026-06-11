# GenForge: Algorithm and Theory

## Overview

GenForge is a gradient-boosted oblique decision tree ensemble built on three principles:

1. **GG-SRP** (Gradient-Guided Sparse Random Projection) — oblique directions derived from gradient/Hessian statistics rather than expensive numerical optimization.
2. **Hereditary direction inheritance** — child nodes inherit and mutate their parent's split direction, exploiting the fact that parent-filtered subsets are already partially separated.
3. **Parent-cache crossover with depth-decayed mutation** — global high-performing directions (stored in a ring buffer) are blended with local parent directions; mutation strength decays with depth to shift from exploration to exploitation.

---

## Boosting Framework

GenForge uses Newton-Raphson gradient boosting. Given ensemble prediction $F^{(m)}$, the $(m+1)$-th tree minimizes:

$$\sum_i \left[ g_i f(x_i) + \tfrac{1}{2} h_i f(x_i)^2 \right] + \Omega(f), \quad \Omega(f) = \lambda \|w\|^2$$

where $g_i, h_i$ are first- and second-order gradients of the loss. For $K$-class problems, the model maintains $K$ output heads with softmax + cross-entropy.

---

## Stage 1: GG-SRP — Gradient-Guided Sparse Random Projection

**Why:** Per-node numerical Coordinate Descent (Gauss-Seidel with Gram matrix + WLS panels) is computationally prohibitive for deep forests. GG-SRP replaces it with a gradient-informed probabilistic direction that is $O(D)$ to compute.

**How:**

1. Compute SIS feature importance scores on the node subsample:
$$s_f = \frac{|\sum_i x_{if} g_i|}{\sqrt{\sum_i h_i x_{if}^2 + \lambda}}$$

2. Sample a random subspace $\mathcal{F}$ of size $b$ with probability $\propto s_f$ (higher gradient correlation = more likely to be included).

3. For each selected feature $f \in \mathcal{F}$, set weight sign to match the steepest descent direction:
$$\tilde{w}_f = -\operatorname{sign}\!\left(\sum_i x_{if} g_i\right) \cdot |r_f|, \quad r_f \sim \mathcal{N}(0,1)$$

4. Normalize $\tilde{w}$ to unit norm.

This produces a random oblique direction that is aligned with the gradient on the dominant informative subspace, without solving any linear system.

---

## Stage 2: Hereditary Direction Inheritance

**Why:** Samples reaching a child node have already been filtered by the parent split. The parent's weight vector $w_\text{parent}$ is therefore a meaningful warm start for the child's direction — the local geometry is already partially aligned.

Two mutation strategies:

**Strategy A — Axis-Maintaining Mutation:**  
Inherit parent weights and add calibrated noise:

$$w_\text{A} = w_\text{parent} + \text{rate} \cdot \epsilon, \quad \epsilon_f \sim \mathcal{U}(-1, 1)$$

This tilts the inherited boundary while preserving its orientation.

**Strategy B — New-Axis Borrowing:**  
Borrow one new feature $f^*$ that has high gradient correlation at the current node and extend the parent's sparse direction:

$$w_\text{B} = w_\text{parent} \;\oplus\; \{\,f^* \mapsto \pm\,\text{strength}\,\}$$

This expands the split from the parent subspace into a new dimension without discarding the inherited structure.

**Ratio control:** `inherited_rp_ratio` sets the fraction of direction candidates drawn from Strategies A/B vs. fresh GG-SRP candidates (default 1.0 = all inherited).

---

## Stage 3: Parent-Cache Crossover and Depth-Decayed Mutation

**Strategy C — Global-Local Crossover:**  
The ring buffer `dir_cache` stores up to 32 high-performing directions from previous boosting rounds. Each round, a cache direction $w_\text{cache}$ is blended with the current parent direction:

$$w_\text{C} = \alpha \, w_\text{parent} + (1 - \alpha) \, w_\text{cache}, \quad \alpha \sim \mathcal{U}(0, 1)$$

This prevents local convergence to a suboptimal split plane by injecting globally validated directions.

**Depth-Decayed Mutation:**  
Shallow nodes see a large feature space and benefit from wide exploration; deep nodes operate on small, nearly-pure subsets where large perturbations overfit. Mutation parameters decay with depth:

$$\text{rate}_d = \frac{\text{rate}_0}{\sqrt{1 + d}}, \qquad \text{strength}_d = \frac{\text{strength}_0}{1 + d}$$

where $d$ is the node depth, `rate_0 = mutation_rate`, `strength_0 = mutation_strength`.

---

## Ablation: Evolution of the Direction-Finding Strategy

Internal ablation on a held-out classification benchmark (100 000 samples, 50 features, 10 informative):

| Configuration | Balanced Acc | F1-Macro | Log Loss |
|---------------|-------------|----------|----------|
| GG-SRP only (no inheritance) | 0.96308 | — | 0.11101 |
| + Parent inheritance 75% | 0.96415 | — | 0.10798 |
| + Parent inheritance 100% | 0.96373 | — | 0.10382 |
| + Crossover + Depth Decay (final) | **0.96336** | **0.95386** | **0.10016** |

Key findings:
- Parent inheritance improves Log Loss consistently (better calibration from finer gradient alignment).
- Depth-decayed mutation achieves the best Log Loss (0.10016), confirming that shallow exploration + deep exploitation improves generalization.
- Full GG-SRP removal of CD retains competitive Balanced Accuracy while being significantly faster per node.

---

## Context-Cached Binning (v9)

Computed once per dataset, reused across all rounds:

**Numeric:** Column mean imputation ($x_{\text{NaN}} \leftarrow \mu_f$), 256-bin global histogram codes.

**Categorical:** Value dictionaries (raw int → dense ID). NaN is its own category.

Per round, only categorical **gradient-rank re-encoding** is recomputed:
- Score each category: $s_c = \sum G_{ic} / (\sum H_{ic} + \lambda)$
- Sort by score → replace with rank
- NaN and unseen test categories fall back to the NaN rank

---

## Best-First Tree Growth

Nodes are expanded in decreasing upper-bound gain order (A* / lazy beam). Growth stops at $2^{\text{max\_depth}}$ leaves. Smaller child gets a fresh histogram; larger child = parent − smaller (histogram subtraction).

---

## Complexity

| Step | Cost |
|------|------|
| Context creation | $O(N \cdot D)$ |
| Cat re-encoding per round | $O(N \cdot D_{\text{cat}})$ |
| GG-SRP direction per node | $O(D + b)$ |
| Histogram build (smaller child) | $O(n_t \cdot D \cdot 256)$ |
| Prediction (sparse routing) | $O(N \cdot \text{depth} \cdot s)$ |

$b \leq 16$ (direction subspace), $s \approx 2$–6 (split nnz).
