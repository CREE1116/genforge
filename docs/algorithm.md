# OQBoost: Algorithm and Theory

OQBoost is an optimized gradient-boosted oblique decision tree ensemble designed for high accuracy, robust generalization, and fast C++ execution. This document details the mathematical and architectural design of OQBoost, including the recent C++ engine optimizations.

---

## 1. Newton-Raphson Boosting Framework

OQBoost follows the Newton-Raphson boosting formulation. Given an ensemble prediction $F^{(m)}(x)$ at iteration $m$, the next tree $f^{(m+1)}(x)$ minimizes the second-order Taylor expansion of the loss function:

$$\mathcal{L}^{(m+1)} \approx \sum_{i=1}^{N} \left[ g_i f^{(m+1)}(x_i) + \frac{1}{2} h_i \left(f^{(m+1)}(x_i)\right)^2 \right] + \Omega(f^{(m+1)})$$

where:
*   $g_i = \left. \frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)} \right|_{F = F^{(m)}}$ is the first-order gradient.
*   $h_i = \left. \frac{\partial^2 \ell(y_i, F(x_i))}{\partial F(x_i)^2} \right|_{F = F^{(m)}}$ is the second-order Hessian.
*   $\Omega(f) = \lambda \sum_{\text{leaves}} \|w_\text{leaf}\|^2$ is the L2 regularization penalty.

For $K$-class classification, the ensemble maintains $K$ output heads and uses a multiclass Softmax loss function.

---

## 2. Split Strategy: Oblique Decision Rules

Unlike axis-aligned trees that split on a single feature $x_j < \theta$, OQBoost uses **oblique splits** (linear combinations of features) at each decision node:

$$w^T x = \sum_{j \in \mathcal{S}} w_j x_j < \theta$$

where $w$ is a sparse weight vector (restricting the active feature subspace size $|\mathcal{S}| \le D_{\text{SUB\_MAX}} = 16$), and $\theta$ is the split threshold. This allows the model to capture oblique relationships directly at the node level.

---

## 3. Direction Generation & Optimization Pipeline

OQBoost avoids computationally expensive Coordinate Descent (CD) or Gram matrix solver operations at each node by using three randomized, gradient-aligned stages.

### Stage 1: GG-SRP (Gradient-Guided Sparse Random Projection)
Instead of searching all dimensions, OQBoost identifies candidate features using Sure Independence Screening (SIS) scores, computed on the node-local subset of samples $\mathcal{I}_t$:

$$s_f = \frac{\left| \sum_{i \in \mathcal{I}_t} x_{if} g_i \right|}{\sqrt{\sum_{i \in \mathcal{I}_t} h_i x_{if}^2 + \lambda}}$$

1.  **Subspace Selection**: A sparse random subspace $\mathcal{S}$ is sampled, where feature $f$ is selected with probability proportional to its screening score $s_f$.
2.  **Active Feature Cap**: To avoid exceeding the C++ `SparseVec` capacity, the number of non-zero features is strictly capped at $D_{\text{SUB\_MAX}} = 16$.
3.  **Sign Alignment**: For each selected feature $f \in \mathcal{S}$, the projection weight matches the steepest gradient descent sign:
    $$w_f = -\operatorname{sign}\left( \sum_{i \in \mathcal{I}_t} x_{if} g_i \right) \cdot |r_f|, \quad r_f \sim \mathcal{N}(0, 1)$$
4.  **Normalization**: The weight vector is normalized to unit L2 norm: $w \leftarrow \frac{w}{\|w\|_2}$.

### Stage 2: Dataset-Level Direction Cache (Strategy C)
High-performing split directions from previous boosting rounds are stored in a global ring buffer `dir_cache` (up to 32 directions, near-duplicates rejected at |cos| > 0.95). At non-root nodes, the informed candidate slot blends a cached direction with the current parent direction:

$$w_{\text{blend}} = \alpha w_{\text{parent}} + (1 - \alpha) w_{\text{cache}}, \quad \alpha \sim \mathcal{U}(0.2, 0.8)$$

While the cache is still empty (early trees), the slot falls back to fresh GG-SRP-style sparse random draws. The `inherited_rp_ratio` parameter (default 0.5) splits the oblique candidate budget between this informed pool and the global random pool.

**Why no parent-direction mutation?** Earlier versions also mutated the parent node's own direction (Strategy A: weight perturbation; Strategy B: new-axis borrowing). Controlled strategy ablations (see `research/FINDINGS.md`) showed both are consistently 0.4–1pp worse than replacing them with random draws: a split on $w$ consumes the gradient signal along $w$, so the residual signal inside each child concentrates away from the parent direction. Good oblique directions are a *global* property of the dataset (its rotated subspaces) — which is exactly what the cache captures across trees — not a local property of the parent node. Both strategies were removed; `mutation_rate` / `mutation_strength` remain as deprecated no-ops.

*   **RNG Seed Propagation**: To ensure tree diversity and prevent redundant splits across boosting rounds, the Python classifier passes a unique random seed (e.g., `rng.integers(1 << 30)`) to the C++ engine for each tree. The C++ engine then seeds its node-level generators with `seed + t`, ensuring full determinism and maximum diversity.

---

## 4. Memory-Optimized C++ Engine

### Object-Pool Histogram Recycling (`hist_pool`)
Building histograms is the primary computational bottleneck in GBDT training. To prevent frequent heap memory allocation (`malloc`) and deallocation (`free`) during best-first node growth, OQBoost uses a lightweight object pool for histogram buffers inside `gf_build`:

```cpp
std::vector<std::vector<float>> hist_pool;

auto get_hist = [&]() -> std::vector<float> {
  if (!hist_pool.empty()) {
    auto h = std::move(hist_pool.back());
    hist_pool.pop_back();
    std::fill(h.begin(), h.end(), 0.0f);
    return h;
  }
  return std::vector<float>(HSZ, 0.0f);
};

auto recycle_hist = [&](std::vector<float>& h) {
  if (h.size() == HSZ) {
    hist_pool.push_back(std::move(h));
  }
  h.clear();
};
```

This pool recycles 256-bin feature histograms, reducing heap allocation overhead to zero once the pool reaches steady state.

### Dynamic Hybrid Histogram Parallelization
OQBoost dynamically selects between two OpenMP parallelization strategies depending on the number of features $D$ relative to the CPU thread count $T$:

1.  **Block-Wise Feature-Parallelism (when $D \ge T$)**:
    *   Features are divided into contiguous blocks of size $\lfloor D/T \rfloor$.
    *   Each thread processes its block, writing directly into its assigned slice of the global histogram buffer.
    *   This eliminates thread-local buffers and histogram merging steps, ensuring cache-contiguous reads/writes and maximum cache performance.
2.  **Sample-Parallelism (when $D < T$)**:
    *   When $D$ is small, feature-parallelism would leave CPU cores under-utilized.
    *   Samples are divided among threads. Each thread computes a small local histogram, which is merged using a parallel reduction loop.
    *   Since $D$ is small, the local buffers fit entirely within the L1/L2 cache, preventing cache thrashing.

---

## 5. Algorithmic Complexity

| Phase | Time Complexity | Notes |
| :--- | :--- | :--- |
| **Context Creation** | $O(N \cdot D)$ | Done once; computes bin thresholds. |
| **Categorical Re-encoding** | $O(N \cdot D_{\text{cat}})$ | Done once per boosting round. |
| **GG-SRP Projection** | $O(D + b)$ | Done per node; $b \le 16$. |
| **Histogram Construction** | $O(n_t \cdot D \cdot 256)$ | Scaled down using histogram subtraction. |
| **Inference Routing** | $O(N \cdot \text{depth} \cdot s)$ | $s \le 6$ (average active features per split). |

---

## 6. Ablation & Empirical Findings

Ablation studies on a classification benchmark (100,000 samples, 50 features) demonstrate the impact of each algorithmic design:

*   **GG-SRP vs. Coordinate Descent**: Replacing exact Gauss-Seidel CD coordinate search with GG-SRP maintains balanced accuracy parity while speeding up node evaluation by over **10x**.
*   **Diversity is the essential ingredient**: budget-matched knockouts show removing the random oblique pool collapses accuracy to the axis-only floor, while the random pool alone matches the full candidate set. Its mechanism is wider feature coverage (feature-usage entropy 0.945 → 0.970), not tree decorrelation, and its win share grows monotonically over boosting rounds (18% → 35%) as residuals become oblique.
*   **Continuity matters at the dataset scale, not the parent scale**: strategy ablations show the direction cache (Strategy C) is the only inherited-slot component that earns its place; parent-direction mutation (former Strategies A/B) consistently underperformed random replacement and was removed. Full study: `research/FINDINGS.md`.
*   **Balanced Argmax**: OQBoost predictions use a prior-corrected argmax formula to account for class imbalance, keeping predictions well-calibrated without altering the C++ Newton-Raphson gradients.
