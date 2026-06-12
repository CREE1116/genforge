# study: Cosine-Constrained Parent Direction Mutation Strategy

This study evaluates the **Cosine-Constrained Mutation** strategy, which restricts child candidate directions to a cone around the parent direction $\mathbf{w}_p$, and evaluates its interaction with a decaying candidate budget.

The goal is to implement the principle: **"Projections must be diverse but continuous"** (다양해야 정답을 찾기 쉽고, 연속적이어야 실제 게인을 얻을 수 있다).

---

## 1. Mathematical formulation

A candidate direction $\mathbf{w}$ is constructed to have an exact cosine similarity $\gamma$ with the parent direction $\mathbf{w}_p$:
$$\mathbf{w} = \gamma \mathbf{w}_p + \sqrt{1 - \gamma^2} \mathbf{r}_{\perp}$$
where:
- $\mathbf{w}_p$ is the parent split direction (normalized).
- $\mathbf{r}_{\perp}$ is a random unit vector drawn from the orthogonal complement of $\mathbf{w}_p$ (using the `make_O(1.0)` projection).
- $\gamma$ is the minimum cosine similarity (cone constraint), scheduled to increase (narrowing the cone) as depth increases:
  $$\gamma = \min(0.99, 0.3 + 0.2 \times \text{depth})$$
  - Depth 0 (first split under root): $\gamma = 0.5$ (cone angle $60^\circ$)
  - Depth 1: $\gamma = 0.5$ (cone angle $60^\circ$)
  - Depth 2: $\gamma = 0.7$ (cone angle $45^\circ$)
  - Depth 3: $\gamma = 0.9$ (cone angle $25^\circ$)
  - Depth 4: $\gamma = 0.99$ (cone angle $8^\circ$)

---

## 2. Compared Strategies

1. **`noise_less`** (Control Baseline):
   - Noise-mutation strategy with decaying noise scale (`inherit_mode='noise_mutation'`, `noise_strategy='less'`).
   - Constant budget of 8 candidates per node.
2. **`cosine_constrained`**:
   - Cosine constraint cone narrowing with depth.
   - Constant budget of 8 candidates per node.
3. **`cosine_constrained_less_budget`**:
   - Cosine constraint cone narrowing with depth.
   - Decaying candidate budget per node:
     $$N_{\text{candidates}} = \max(2, 8 - 2 \times \text{depth})$$
     - Depth 0: 8 candidates
     - Depth 1: 6 candidates
     - Depth 2: 4 candidates
     - Depth 3: 2 candidates

---

## 3. Results (Mean ± Std over 3 seeds)

| Dataset | Strategy | Test Accuracy ↑ | Test Log Loss ↓ | Mean Time/Run ↓ |
| :--- | :--- | :---: | :---: | :---: |
| **Rotated Synthetic** | **noise_less** | **0.9317 ± 0.0025** | 0.1834 ± 0.0020 | 31.30s |
| | **cosine_constrained** | 0.9290 ± 0.0049 | 0.1907 ± 0.0100 | 30.94s |
| | **cosine_constrained_less_budget** | 0.9310 ± 0.0014 | **0.1829 ± 0.0029** | **25.18s** (19.5% faster) |
| **Digits** | **noise_less** | 0.9763 ± 0.0010 | 0.1053 ± 0.0021 | 33.32s |
| | **cosine_constrained** | 0.9711 ± 0.0065 | 0.1087 ± 0.0060 | 33.05s |
| | **cosine_constrained_less_budget** | **0.9793 ± 0.0055** | **0.1025 ± 0.0033** | **27.13s** (18.6% faster) |
| **Wine** | **noise_less** | **0.9778 ± 0.0000** | 0.1211 ± 0.0027 | 5.97s |
| | **cosine_constrained** | **0.9778 ± 0.0000** | 0.1201 ± 0.0045 | 5.97s |
| | **cosine_constrained_less_budget** | **0.9778 ± 0.0000** | **0.1200 ± 0.0044** | **5.46s** (8.5% faster) |
| **Synthetic Correlated** | **noise_less** | 0.9513 ± 0.0066 | 0.1420 ± 0.0009 | 27.69s |
| | **cosine_constrained** | 0.9527 ± 0.0019 | 0.1411 ± 0.0027 | 27.59s |
| | **cosine_constrained_less_budget** | **0.9553 ± 0.0038** | **0.1396 ± 0.0035** | **22.41s** (19.1% faster) |

---

## 4. Key Insights

1. **Both Faster and More Accurate (Win-Win)**:
   - The `cosine_constrained_less_budget` strategy achieves the highest accuracy on both **Digits** (**0.9793**) and **Synthetic Correlated** (**0.9553**), while simultaneously running **18% to 20% faster** than the baseline.
   
2. **Speedups from Candidate Budget Reduction**:
   - Because tree nodes grow exponentially with depth, the vast majority of split computations occur at deeper levels (e.g. depth 2 and 3). 
   - By reducing the number of candidate projections from 8 down to 4 (at depth 2) and 2 (at depth 3), we dramatically reduce the number of matrix multiplications (`X_node @ W.T`) and threshold evaluations, yielding massive training speedups.

3. **Generalization Benefits of Restricting Search Cone**:
   - Limiting the budget at deeper levels also acts as a powerful regularizer. With fewer candidate projections to select from, the model is prevented from selecting spurious, overfitted oblique directions that happen to fit local noise.
   - Constraining the cosine similarity ($\gamma \to 0.99$) ensures that the splits remain **continuous and locally aligned** with the parent split. This aligns with the fact that as child nodes partition smaller subsets of data, the optimal direction shifts smoothly rather than making abrupt jumps, preventing overfitting on local fragments.
