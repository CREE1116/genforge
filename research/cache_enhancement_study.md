# study: Cache Strategy Enhancements in Oblique Direction Inheritance

This study evaluates two proposed enhancements to the global direction cache (Strategy C) in OQBoost, comparing them directly to the baseline `cache_blend` ported from the production C++ engine:

1. **`cache_blend` (Baseline)**:
   - FIFO cache replacement (max size 32, similarity filter 0.95).
   - Parent blend: $w_{\text{blend}} = \alpha w_{\text{parent}} + (1 - \alpha) w_{\text{cache}}$.
2. **`cache_orb` (Orthogonal Random Blending)**:
   - FIFO cache replacement.
   - Project cached direction orthogonal to parent.
   - Blend with a fresh sparse random direction instead of the parent direction to maximize diversity:
     $$w_{\text{blend}} = (1 - \alpha) w_{\text{cache},\perp} + \alpha w_{\text{rand}}$$
   - Final projection orthogonal to the parent direction.
3. **`cache_evolutionary`**:
   - Evolutionary Cache Replacement (ECR): tracks the cumulative split gain of each cache item, and replaces the direction with the lowest score when full (instead of FIFO).
   - Uses ORB for blending.

---

## 1. Experimental Results (Mean ± Std over 3 seeds)

| Dataset | Strategy | Test Accuracy ↑ | Test Log Loss ↓ | Mean Time/Run ↓ |
| :--- | :--- | :---: | :---: | :---: |
| **Rotated Synthetic** | **cache_blend** | **0.9333 ± 0.0017** | 0.1784 ± 0.0028 | 39.90s |
| | **cache_orb** | 0.9320 ± 0.0033 | **0.1777 ± 0.0042** | 39.11s |
| | **cache_evolutionary** | 0.9263 ± 0.0019 | 0.1831 ± 0.0001 | **38.77s** |
| **Digits** | **cache_blend** | 0.9770 ± 0.0028 | **0.0976 ± 0.0009** | 43.44s |
| | **cache_orb** | 0.9756 ± 0.0048 | 0.1023 ± 0.0060 | 43.59s |
| | **cache_evolutionary** | **0.9778 ± 0.0048** | 0.0991 ± 0.0053 | **42.39s** |
| **Wine** | **cache_blend** | **0.9778 ± 0.0000** | 0.1173 ± 0.0036 | 7.97s |
| | **cache_orb** | **0.9778 ± 0.0000** | **0.1089 ± 0.0101** | **7.84s** |
| | **cache_evolutionary** | **0.9778 ± 0.0000** | **0.1089 ± 0.0101** | 8.02s |
| **Synthetic Correlated** | **cache_blend** | **0.9567 ± 0.0057** | **0.1369 ± 0.0036** | **32.99s** |
| | **cache_orb** | 0.9533 ± 0.0009 | 0.1415 ± 0.0065 | 33.81s |
| | **cache_evolutionary** | 0.9547 ± 0.0041 | 0.1470 ± 0.0043 | 34.15s |

---

## 2. Key Findings & Theoretical Insights

1. **The Parent Direction acts as a Local Coordinate Anchor**:
   - The baseline `cache_blend` outperforms the orthogonalized random blend (`cache_orb`) on `synthetic` and `rotated_synthetic` accuracy.
   - Even though pure parent-direction mutation is harmful (as the parent split consumes the gradient along its normal), blending the global cache direction with the parent direction acts as a **local coordinate helper**. It aligns the global rotated subspace direction to the parent node's local region. Without blending with the parent direction, the cache direction loses local continuity.

2. **FIFO acts as a Sliding Window for Shifting Residual Targets**:
   - The evolutionary cache (`cache_evolutionary`), which retains high-gain directions by accumulating scores, performed worse than the simple FIFO baseline on several datasets.
   - In gradient boosting, the target residual shifts dynamically round-by-round. A direction that was extremely high-gain in round 1 (fitting global structure) becomes useless by round 40. 
   - A score-accumulating cache becomes stale because early-stage directions stay in the cache forever. In contrast, **FIFO acts as a sliding window** that naturally purges old directions and adapts the cache to the shifting residuals of the current boosting rounds.

3. **Conclusion & Recommendation**:
   - The current C++ engine implementation (`cache_blend`) is remarkably well-balanced. Blending the global cache with the parent direction provides local orientation, and FIFO provides a sliding window that matches the dynamics of boosting. No changes are recommended to the production cache.
