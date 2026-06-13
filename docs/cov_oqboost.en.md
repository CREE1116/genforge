## Technical Specification: Covariance-Based Oblique Split Generation (OQBoost-Cov Core)

This specification defines the mathematical design and algorithm pipeline of the covariance-based split generation algorithm. It shifts the paradigm of oblique branch searching in gradient boosting trees (GBDT) from an "optimization problem ($O(B \times N)$)" to an "analytical statistic calculation problem ($O(N)$)."

---

## 1. Overview

Traditional oblique tree models generate a massive pool of random candidates and perform a brute-force tournament search to find a split direction $w$ that maximizes the Newton-Raphson Gain.

This method leverages the covariance vector between node-level gradients and feature values to directly compute (analytically generate) the optimal oblique axis passing through the gain ridge in a single pass. This bounds the number of candidate searches $B$ to a constant ($O(1)$), dramatically reducing computational complexity while maximizing generalization performance through implicit geometric regularization.

---

## 2. Mathematical Formulation

### 2.1 Linear Relaxation of Weighted Least Squares (WLS)

The quadratic Newton-Raphson gain objective function that a boosting tree model optimizes at a local node is equivalent to a weighted least squares problem:

$$\min_{w, \|w\|_2=1} \sum_{i=1}^{N} h_i \left( w^T x_i + \frac{g_i}{h_i} \right)^2$$

Substituting the local Hessian variations with a constant ($h_i \approx \bar{h}$) and assuming feature independence ($X^T X \approx I$), the expansion converges to a linear term optimization problem:

$$\min_{w, \|w\|_2=1} 2 w^T X^T g \iff \max_{w, \|w\|_2=1} w^T (-X^T g)$$

### 2.2 Covariance Direction Vector Formula

By the Cauchy-Schwarz inequality, the inner product of two vectors is maximized when they are parallel. Therefore, the optimal oblique split axis $w^*$ is directly derived as the normalized covariance vector between the feature matrix $X$ and the gradient vector $g$:

$$w^* = \frac{-X^T g}{\|-X^T g\|_2}$$

### 2.3 Implicit Regularization

- By removing the matrix inverse term $(X^T HX)^{-1}$ present in the exact WLS solution, we prevent eigenvalue distortion and local noise amplification (high variance) at deep nodes where sample counts drop significantly.
- It acts as a statistical filter that constrains variance and only captures the major principal trends of the gradient error.

---

## 3. Architecture Pipeline

The process executed at each node split decision follows a 4-stage hybrid search pipeline:

```
[Sample Data Input]
        │
        ▼
[Stage 1: SIS Feature Screening] ───► Extract top d_sub features contributing to error
        │
        ▼
[Stage 2: Covariance Axis Derivation] ─► Compute and normalize X_sub.T @ g to create w_cov (1 candidate)
        │
        ▼
[Stage 3: Hybrid Tournament] ──────► Merge axis-aligned candidates (D) + covariance oblique candidate (1)
        │
        ▼
[Stage 4: Batched Projection & Scan] ──► Tensor GEMM (X @ W.T) followed by a single optimal gain extraction
```

### Detailed Stage Description

- **Stage 1: SIS Feature Screening (Space Compression)**
  To prevent the curse of dimensionality, features are compressed to a smaller dimension `d_sub` based on the gradient distribution of the dominant class.
- **Stage 2: Covariance Axis Derivation (Analytical Target)**
  A single vector-matrix multiplication ($X_{\text{sub}}^T g$) is performed on the selected subspace to generate the covariance axis candidate `w_cov`.
- **Stage 3: Safe Backbone Ring Construction (Hybrid Subspace)**
  To safeguard against feature scale collapse and multicollinearity in tabular data, $D$ standard axis-aligned basis vectors and 1 `w_cov` are merged into a final **$D+1$ candidate tournament pool**.
- **Stage 4: Batched Projection and Optimal Split (Batched Scan)**
  The sample matrix $X$ is projected onto the candidate matrix $W$ in a single Tensor GEMM operation (`X @ W.T`), and the split threshold ($\theta$) and axis achieving the maximum gain are determined.

---

## 4. Complexity Profile

| Search Mechanism | Split Generation Complexity | Histogram Building Complexity | Total Node Search Complexity |
| :--- | :--- | :--- | :--- |
| **Traditional Oblique Tree** | $O(B \cdot D)$ (Random) | $O(B \cdot N \cdot \text{Bins})$ | $O(B \cdot N \cdot \text{Bins})$ (Proportional to $B$) |
| **Covariance Model (`proxy_cov_axis_1`)** | $O(N \cdot d_{\text{sub}})$ (Analytical) | $O((D + 1) \cdot N \cdot \text{Bins})$ | $\mathbf{O(D \cdot N)}$ (Linear acceleration by removing $B$) |

---

## 5. Runtime Control and Exception Handling

### 5.1 Defense against Node-Scale Energy Illusion

- Even if the cross-node correlation is high ($\ge 0.96$), the within-node correlation can behave independently near 0.
- Since the gain landscape is a sharp ridge rather than a smooth bowl, the gain can drop vertically if the covariance angle is slightly off.
- To safeguard against this, standard axis-aligned (`axis`) backbone candidates remain in the tournament pool (the actual win ratio in benchmarks converges to roughly `Axis 80% : Oblique 20%`).

### 5.2 Defense against GPU Hardware Synchronization Bottlenecks (Tensor-Level Optimization)

- Python scalar conversions and copies (`.item()`, `float()`) are strictly avoided inside the gain optimization loop.
- All threshold scans and gain tournaments are executed via **pure tensor-level argmax** operations directly in GPU memory (`cuda` / `mps` layouts). A CPU interrupt is triggered only at the final split decision to minimize synchronization latency.

---

[Korean Version (한국어 버전)](cov_oqboost.md)
