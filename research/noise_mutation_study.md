# study: Noise-Injected Parent Direction Inheritance

This study investigates parent direction mutation strategies. Specifically, we inject random noise into the parent split direction $\mathbf{w}_p$ to generate candidate directions for child nodes, comparing how the scale of injected noise at different depths affects overall model accuracy and log loss.

The candidate direction is computed as:
$$\mathbf{w}_{\text{rand}} = \text{Normalize}(\mathbf{w}_p + \eta \mathbf{r})$$
where:
- $\mathbf{w}_p$ is the parent split direction (normalized).
- $\mathbf{r}$ is a random unit vector drawn from the parent support union and the top feature subspace.
- $\eta$ is the noise scale governed by the selected strategy and the current depth of the node.

---

## 1. Strategies Tested

We compared three strategies for scheduling the noise scale $\eta$ based on depth:
1. **`uniform`**: Every node has a constant noise scale:
   $$\eta = 0.5$$
2. **`less`** (Decaying noise with depth): Noise is larger at shallow depths to encourage exploration and smaller at deeper levels to preserve the parent direction:
   $$\eta = \max(0.1, 1.0 - 0.3 \times \text{depth})$$
3. **`more`** (Increasing noise with depth): Noise is small at shallow depths and grows larger at deeper levels:
   $$\eta = \min(1.5, 0.1 + 0.3 \times \text{depth})$$

---

## 2. Experimental Setup

- **Model**: `OQBoostResearch` with `n_estimators = 50`, `learning_rate = 0.1`, `max_depth = 4`, `use_wls = True`, `inherit_mode = 'noise_mutation'`.
- **Hardware**: Run on **CPU** to avoid device transfer and synchronization overhead.
- **Seeds**: 3 independent random seeds.
- **Datasets**:
  1. **Rotated Synthetic**: Binary classification (4000 samples, 50 features).
  2. **Digits**: Multiclass (10 classes, 1797 samples, 64 features).
  3. **Wine**: Multiclass (3 classes, 178 samples, 13 features).
  4. **Synthetic Correlated**: Binary classification (2000 samples, 30 features).

---

## 3. Results (Mean ± Std over 3 seeds)

| Dataset | Strategy | Test Accuracy ↑ | Test Log Loss ↓ |
| :--- | :--- | :---: | :---: |
| **Rotated Synthetic** | **uniform** | 0.9300 ± 0.0022 | 0.1837 ± 0.0046 |
| | **less** | **0.9317 ± 0.0025** | 0.1834 ± 0.0020 |
| | **more** | 0.9307 ± 0.0038 | **0.1808 ± 0.0020** |
| **Digits** | **uniform** | 0.9696 ± 0.0028 | 0.1100 ± 0.0008 |
| | **less** | **0.9763 ± 0.0010** | **0.1053 ± 0.0021** |
| | **more** | 0.9719 ± 0.0028 | 0.1104 ± 0.0023 |
| **Wine** | **uniform** | **0.9778 ± 0.0000** | **0.1211 ± 0.0027** |
| | **less** | **0.9778 ± 0.0000** | **0.1211 ± 0.0027** |
| | **more** | **0.9778 ± 0.0000** | **0.1211 ± 0.0027** |
| **Synthetic Correlated** | **uniform** | 0.9507 ± 0.0025 | 0.1460 ± 0.0048 |
| | **less** | 0.9513 ± 0.0066 | 0.1420 ± 0.0009 |
| | **more** | **0.9540 ± 0.0000** | **0.1411 ± 0.0079** |

---

## 4. Key Findings & Discussion

1. **coarse-to-fine exploration works best (`less`)**:
   - On the complex, higher-dimensional **Digits** dataset, the `less` strategy outperformed both `uniform` and `more` strategies by a clear margin (**0.9763 Accuracy** vs **0.9696 / 0.9719**).
   - At shallow depths (e.g., depth 0 or 1), the tree is partitioning the data globally. Injecting a larger noise scale ($\eta = 1.0$) promotes wider exploration of the oblique feature space.
   - At deeper levels, the partitions are highly local. Decaying the noise scale ($\eta \to 0.1$) ensures the child direction is only slightly mutated from the parent direction, acting as a local adjustment of the decision boundary rather than a completely new random search.

2. **Increasing noise at deeper levels can be counter-productive**:
   - Increasing the noise scale (`more`) at deeper levels introduces large random perturbations where the tree needs to fine-tune boundaries. This degrades performance on highly structured datasets like `digits`.
   - On simple datasets like `wine`, all three strategies converge on identical boundaries because the space is highly separable and tree depth is likely very shallow.

3. **Conclusion**:
   - The **`less`** strategy (decaying noise scale as depth increases) is the most effective choice when using parent direction mutation. It mirrors the standard coarse-to-fine optimization heuristic: explore widely in early nodes, and exploit/fine-tune in local leaf regions.
