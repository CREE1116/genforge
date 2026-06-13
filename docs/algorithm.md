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

where $w$ is a sparse weight vector (restricting the active feature subspace size $|\mathcal{S}| \le D_{\text{SUB_MAX}} = 16$), and $\theta$ is the split threshold. This allows the model to capture oblique relationships directly at the node level.

---

## 3. Direction Generation & Optimization Pipeline

OQBoost avoids computationally expensive Coordinate Descent (CD) or Gram matrix solver operations at each node by using three randomized, gradient-aligned stages.

### Stage 1: GG-SRP (Gradient-Guided Sparse Random Projection)
Instead of searching all dimensions, OQBoost identifies candidate features using Sure Independence Screening (SIS) scores, computed on the node-local subset of samples $\mathcal{I}_t$:

$$s_f = \frac{\left| \sum_{i \in \mathcal{I}_t} x_{if} g_i \right|}{\sqrt{\sum_{i \in \mathcal{I}_t} h_i x_{if}^2 + \lambda}}$$

1.  **Subspace Selection**: A sparse random subspace $\mathcal{S}$ is sampled, where feature $f$ is selected with probability proportional to its screening score $s_f$.
2.  **Active Feature Cap**: To avoid exceeding the C++ `SparseVec` capacity, the number of non-zero features is strictly capped at $D_{\text{SUB_MAX}} = 16$.
3.  **Sign Alignment**: For each selected feature $f \in \mathcal{S}$, the projection weight matches the steepest gradient descent sign:
    $$w_f = -\operatorname{sign}\left( \sum_{i \in \mathcal{I}_t} x_{if} g_i \right) \cdot |r_f|, \quad r_f \sim \mathcal{N}(0, 1)$$
4.  **Normalization**: The weight vector is normalized to unit L2 norm: $w \leftarrow \frac{w}{\|w\|_2}$.

### Stage 2: Hereditary Direction Inheritance
Samples reaching a child node have already been filtered by the parent node's split. The parent's weight vector $w_{\text{parent}}$ serves as an excellent warm-start direction. OQBoost evaluates two mutation strategies:
*   **Strategy A (Axis-Maintaining Mutation)**: Tilts the parent boundary slightly while maintaining its orientation:
    $$w_{\text{mutated}} = w_{\text{parent}} + \text{rate} \cdot \epsilon, \quad \epsilon_j \sim \mathcal{U}(-1, 1)$$
*   **Strategy B (New-Axis Borrowing)**: Extends the sparse representation by adding one highly correlated new feature $f^*$ to the parent's feature set:
    $$w_{\text{mutated}} = w_{\text{parent}} \oplus \{ f^* \mapsto \pm \text{strength} \}$$

### Stage 3: Parent-Cache Crossover & Depth-Decayed Mutation
*   **Strategy C (Global-Local Crossover)**: High-performing split directions from previous boosting rounds are stored in a global ring buffer `dir_cache` (up to 32 directions). OQBoost blends these with the current parent direction:
    $$w_{\text{blend}} = \alpha w_{\text{parent}} + (1 - \alpha) w_{\text{cache}}, \quad \alpha \sim \mathcal{U}(0, 1)$$
*   **Depth-Decayed Mutation**: Shallow nodes benefit from exploration, while deep nodes operate on smaller, nearly-pure subsets where large mutations overfit. OQBoost decays the mutation rate and strength at depth $d$:
    $$\text{rate}_d = \frac{\text{rate}_0}{\sqrt{1 + d}}, \quad \text{strength}_d = \frac{\text{strength}_0}{1 + d}$$

*   **RNG Seed Propagation**: To ensure tree diversity and prevent redundant splits across boosting rounds, the Python classifier passes a unique random seed (e.g., `rng.integers(1 << 30)`) to the C++ engine for each tree. The C++ engine then seeds its node-level generators with `seed + t`, ensuring full determinism and maximum diversity.

---

## 4. Why It Works: Mathematical and Statistical Intuition

OQBoost outperforms standard axis-aligned models and avoids the heavy computational bottlenecks of traditional oblique tree GBDTs due to several mathematical and statistical properties:

### 4.1 Newton-Raphson Gradient Alignment
The core advantage of GG-SRP is that it does not rely on purely blind random projections. Instead, it explicitly aligns the projection weight signs with the local gradient descent directions ($w_f \propto -\operatorname{sign}(\sum g_i x_{if})$).
Mathematically, this aligns the starting projection axis with the steepest descent path of the quadratic Taylor expansion (the Newton step). As a result, even a single random projection vector in the tournament pool is guaranteed to align with the gradient error trends and capture significant gain.

### 4.2 Extreme Values of Gain Distribution (Order Statistics)
The candidate selection at each decision node is an extreme value optimization problem ($G^* = \max_i \text{gain}(w_i)$).
According to order statistics, the expected maximum of a mixture of different candidate families (Axis, GG-SRP, Inherited, Cache) is mathematically strictly greater than that of any single family alone, provided their gain distributions have distinct tail shapes or are weakly correlated. This explains why pure configurations (e.g., all-random or all-cache) are consistently outperformed by the production mixture.

### 4.3 Orthogonality Principle & POBS
When a node splits along direction $w$, the linear component of the gradient error parallel to $w$ is consumed. Consequently, the residual error in the child nodes concentrates in the orthogonal complement subspace ($S^\perp$) of the parent hyperplane.
- OQBoost exploits this by injecting **POBS (Parseval-Constrained Random Orthogonal Block Projections)**. By generating mutually orthogonal candidates (via Haar-random orthogonal block frames), POBS eliminates redundant search dimensions. Under Parseval's identity, the total projection energy is conserved, guaranteeing a worst-case lower bound on the maximum gain captured per tournament.

### 4.4 Bias-Variance Trade-Off via Depth-Adaptive Budgets
As trees grow deeper, the number of samples reaching a node $N_t$ decreases exponentially.
- **Deep Node Estimation Noise**: Computing complex optimization (such as CD or matrix inverses in WLS) on small sample sizes yields high-variance directions that overfit to local noise.
- **Adaptive Budget Allocation**: OQBoost schedules large tournament sizes (64 candidates) at shallow nodes where the sample size is large and the signal is clean, while restricting deep node tournaments to 8 candidates. This not only speeds up training by 25% but also acts as an implicit regularizer, preventing high-variance splits at deep nodes.

---

## 5. Memory-Optimized C++ Engine

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

## 6. Algorithmic Complexity

| Phase | Time Complexity | Notes |
| :--- | :--- | :--- |
| **Context Creation** | $O(N \cdot D)$ | Done once; computes bin thresholds. |
| **Categorical Re-encoding** | $O(N \cdot D_{\text{cat}})$ | Done once per boosting round. |
| **GG-SRP Projection** | $O(D + b)$ | Done per node; $b \le 16$. |
| **Histogram Construction** | $O(n_t \cdot D \cdot 256)$ | Scaled down using histogram subtraction. |
| **Inference Routing** | $O(N \cdot \text{depth} \cdot s)$ | $s \le 6$ (average active features per split). |

---

## 7. Ablation & Empirical Findings

Ablation studies on a classification benchmark (100,000 samples, 50 features) demonstrate the impact of each algorithmic design:

*   **GG-SRP vs. Coordinate Descent**: Replacing exact Gauss-Seidel CD coordinate search with GG-SRP maintains balanced accuracy parity while speeding up node evaluation by over **10x**.
*   **Parent Inheritance & Crossover**: Enabling Strategy C (Crossover) and depth-decayed mutations consistently yields the lowest Log Loss (improving model calibration and reducing overfitting).
*   **Balanced Argmax**: OQBoost predictions use a prior-corrected argmax formula to account for class imbalance, keeping predictions well-calibrated without altering the C++ Newton-Raphson gradients.
*   **Mechanism studies in progress**: research-impl ablations on synthetic data questioned the value of parent-direction mutation, but transplanting that change regressed the real tuned benchmarks and was reverted — empirical synthetic-data findings need a theoretical account before they justify engine changes. See `research/FINDINGS.md`.

---

## 8. Future Research: Gradient Covariance Paradigm

Recent research (conducted on 2026-06-13 in pure PyTorch prototype `OQBoostCovClassifier`) proposes a massive architectural simplification to the direction generation pipeline. 

### Proposed Core Simplification:
Instead of a candidate tournament over a large pool of randomized and inherited directions, the engine evaluates:
1. All $D$ axis-aligned unit vectors (retaining a backbone to capture unrotated features).
2. Exactly **one** oblique direction: the gradient covariance vector $w_{\text{cov}}$ computed on the top $d_{\text{sub}}$ features:
   $$w_{\text{cov_sub}} = \frac{G_{\text{sub}}}{\|G_{\text{sub}}\|_2 + \epsilon}, \quad \text{where } G_{\text{sub}} = -X_{\text{sub}}^T g$$

This reduces the tournament size to exactly $D+1$ candidates at each node.

### Theoretical Advantage:
By avoiding local optimization on noisy deep-node Hessians (which WLS/CD attempts by solving $A^{-1}G$), the covariance vector acts as a robust L2-regularizer (Hessian-free). In head-to-head empirical testing, this simple $D+1$ setup consistently outperformed both the C++ production tournament and Coordinate Descent on real tabular datasets (e.g., Adult, Credit Default) while being structurally simpler. Porting this covariance generation logic into the C++ core engine is planned for future releases.

---

[한국어 버전 (Korean Version)](algorithm.ko.md)
