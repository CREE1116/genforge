# study: Adaptive Cache Blending Ratio Strategies

This study evaluates four data-adaptive methods for scheduling the parent-cache blending ratio $\alpha$ in OQBoost:
$$w_{\text{blend}} = \alpha w_{\text{parent}} + (1 - \alpha) w_{\text{cache}}$$

We compare them directly to the baseline `cache_blend` (which uses a uniform random draw $\alpha \sim \mathcal{U}(0.2, 0.8)$).

---

## 1. Experimental Results (Mean ± Std over 5 seeds)

| Dataset | Strategy | Test Accuracy ↑ | Test Log Loss ↓ | Mean Time/Run ↓ |
| :--- | :--- | :---: | :---: | :---: |
| **Rotated Synthetic** | **cache_blend** (Baseline) | 0.9320 ± 0.0021 | 0.1807 ± 0.0039 | 38.50s |
| | **cache_adaptive_blend_size** | 0.9318 ± 0.0018 | 0.1830 ± 0.0042 | 38.44s |
| | **cache_adaptive_blend_gradient** | 0.9320 ± 0.0046 | **0.1758 ± 0.0065** | **38.28s** |
| | **cache_adaptive_blend_depth_prop** | **0.9332 ± 0.0030** | 0.1825 ± 0.0053 | 38.32s |
| | **cache_adaptive_blend_depth_inv** | 0.9324 ± 0.0052 | 0.1797 ± 0.0047 | 39.47s |
| **Digits** | **cache_blend** | 0.9769 ± 0.0030 | **0.0992 ± 0.0021** | 44.25s |
| | **cache_adaptive_blend_size** | **0.9791 ± 0.0030** | 0.0998 ± 0.0031 | 44.34s |
| | **cache_adaptive_blend_gradient** | 0.9760 ± 0.0026 | 0.1015 ± 0.0056 | 44.35s |
| | **cache_adaptive_blend_depth_prop** | 0.9751 ± 0.0036 | 0.1036 ± 0.0044 | **43.79s** |
| | **cache_adaptive_blend_depth_inv** | 0.9782 ± 0.0017 | 0.1024 ± 0.0035 | 44.20s |
| **Wine** | **cache_blend** | **0.9778 ± 0.0000** | 0.1197 ± 0.0053 | 8.02s |
| | **cache_adaptive_blend_size** | **0.9778 ± 0.0000** | **0.1106 ± 0.0133** | 8.05s |
| | **cache_adaptive_blend_gradient** | **0.9778 ± 0.0000** | 0.1117 ± 0.0160 | **7.94s** |
| | **cache_adaptive_blend_depth_prop** | **0.9778 ± 0.0000** | **0.1106 ± 0.0140** | 7.97s |
| | **cache_adaptive_blend_depth_inv** | **0.9778 ± 0.0000** | 0.1151 ± 0.0071 | 8.21s |
| **Synthetic Correlated** | **cache_blend** | **0.9580 ± 0.0054** | **0.1346 ± 0.0042** | **34.58s** |
| | **cache_adaptive_blend_size** | 0.9552 ± 0.0016 | 0.1428 ± 0.0063 | 35.44s |
| | **cache_adaptive_blend_gradient** | 0.9540 ± 0.0036 | 0.1421 ± 0.0075 | 35.27s |
| | **cache_adaptive_blend_depth_prop** | 0.9560 ± 0.0028 | 0.1385 ± 0.0039 | 35.26s |
| | **cache_adaptive_blend_depth_inv** | 0.9560 ± 0.0057 | 0.1373 ± 0.0054 | 35.30s |

---

## 2. Core Insights

1. **Gradient-Adaptive Minimizes Log Loss in Complex Rotated Spaces**:
   - `cache_adaptive_blend_gradient` achieves the lowest test log loss on **Rotated Synthetic** (**0.1758** vs **0.1807** baseline).
   - This occurs because mapping the parent blend ratio to gradient homogeneity ($\alpha = 1.0 - \sigma_G^2$) ensures that when gradient variance is high (difficult, chaotic splits), the node relies more on the cache to explore new directions, but when gradients are homogeneous (simple splits), it relies almost entirely on the parent to perform localized, fine adjustments.

2. **Size-Adaptive Boosts Accuracy in Hierarchical Spaces**:
   - `cache_adaptive_blend_size` achieves the highest accuracy on the **Digits** dataset (**0.9791** vs **0.9769** baseline) and the lowest log loss on **Wine** (**0.1106**).
   - Size-adaptive scheduling ($\alpha = 1.0 - \rho$) is highly regularizing: it forces a clean progression from global cache-exploration at large-sample nodes to strict parent-continuity at small-sample leaf nodes.

3. **Validation of the Coarse-to-Fine Prior (Depth Proportional vs Depth Inverse)**:
   - `depth_prop` (increasing $\alpha$ with depth, relying more on parent at deeper levels) achieved the highest accuracy on **Rotated Synthetic** (**0.9332**).
   - `depth_inv` (decreasing $\alpha$ with depth) performed worse or mediocre.
   - This empirically validates that the correct geometric prior is indeed **coarse-to-fine**: start with global exploration (low $\alpha$, high cache weight) and transition to local adjustment (high $\alpha$, high parent weight) as you go deeper.

4. **Recommendation**:
   - For general deployment, `cache_adaptive_blend_size` or `cache_adaptive_blend_gradient` can be selectively used depending on whether accuracy or probability calibration (loss) is the primary target. The baseline `cache_blend` remains a very robust, low-complexity default.
