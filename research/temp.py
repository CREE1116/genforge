from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from typing import Optional


# ─── Data structures ──────────────────────────────────────────────────────────

class Node:
    """Represents a single node in our OQBoost Covariance Tree."""
    def __init__(
        self,
        is_leaf: bool,
        leaf_value: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        threshold: float = 0.0,
        left: Optional[Node] = None,
        right: Optional[Node] = None,
        gain: float = 0.0,
    ):
        self.is_leaf = is_leaf
        self.leaf_value = leaf_value
        self.w = w
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain


# ─── Low-level helper functions ───────────────────────────────────────────────

def _dominant_class(G: torch.Tensor) -> int:
    return int(G.abs().sum(0).argmax().item())


def _sis_scores(X: torch.Tensor, G: torch.Tensor, H: torch.Tensor, k: int, reg_lambda: float) -> torch.Tensor:
    g = G[:, k]
    h = H[:, k]
    cg = (X * g.unsqueeze(1)).sum(0)
    add = (h.unsqueeze(1) * X.pow(2)).sum(0)
    return cg.abs() / (add + reg_lambda + 1e-8).sqrt()


def _best_threshold_tensor(proj: torch.Tensor, G: torch.Tensor, H: torch.Tensor, lam: float, n_bins: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    N = len(proj)
    order = proj.argsort()
    proj_s = proj[order]
    G_s = G[order]
    H_s = H[order]

    G_tot = G.sum(0)
    H_tot = H.sum(0)
    root = (G_tot.pow(2) / (H_tot + lam)).sum() / 2

    G_cum = torch.cumsum(G_s, dim=0)
    H_cum = torch.cumsum(H_s, dim=0)

    step = max(1, N // n_bins)
    range_t = torch.arange(step, N - step, step, dtype=torch.long, device=proj.device)
    if len(range_t) == 0:
        return (proj_s[0] if N > 0 else torch.tensor(0.0, device=proj.device)), torch.tensor(-1e18, device=proj.device)

    idxs = range_t - 1
    G_left = G_cum[idxs]
    H_left = H_cum[idxs]

    G_right = G_tot.unsqueeze(0) - G_left
    H_right = H_tot.unsqueeze(0) - H_left

    score_left = (G_left.pow(2) / (H_left + lam)).sum(dim=-1) / 2
    score_right = (G_right.pow(2) / (H_right + lam)).sum(dim=-1) / 2

    gains = score_left + score_right - root

    valid_mask = (H_left.sum(dim=-1) >= 0.1) & (H_right.sum(dim=-1) >= 0.1)
    gains = torch.where(valid_mask, gains, torch.tensor(-1e18, dtype=gains.dtype, device=gains.device))

    best_idx = gains.argmax()
    best_gain = gains[best_idx]
    
    i = range_t[best_idx]
    best_t = (proj_s[i - 1] + proj_s[i]) / 2

    return best_t, best_gain


# ─── OQBoostEmpiricalAdvancedAnalyzer ─────────────────────────────────────────

class OQBoostEmpiricalAdvancedAnalyzer(BaseEstimator, ClassifierMixin):
    """
    Advanced Research Framework evaluating Alpha-Blending dynamics
    and Sample Size (N) Break-even boundaries simultaneously.
    """
    def __init__(
        self,
        n_estimators: int = 20,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        reg_lambda: float = 1.0,
        subsample: float = 1.0,
        d_sub: int = 16,
        random_state: int = 42,
        n_bins: int = 64
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.d_sub = d_sub
        self.random_state = random_state
        self.n_bins = n_bins

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.trees_: list[Node] = []
        self.F_init_: Optional[torch.Tensor] = None
        self.K_: int = 0

        # Empirical Logger Database
        # Stores pairs of (Node Sample Size N, Winning Alpha Value)
        self.empirical_db_: list[tuple[int, float]] = []

    def _leaf_value(self, G: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return -G.sum(0) / (H.sum(0) + self.reg_lambda)

    def _find_split(
        self,
        X_node: torch.Tensor,
        G_node: torch.Tensor,
        H_node: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        N, D = X_node.shape
        k_dom = _dominant_class(G_node)

        # 1. Standard Axis Scan to find the absolute best Axis baseline
        axis_gains = torch.empty(D, device=X_node.device)
        axis_thresholds = torch.empty(D, device=X_node.device)
        for f in range(D):
            thresh, gain = _best_threshold_tensor(X_node[:, f], G_node, H_node, self.reg_lambda, n_bins=self.n_bins)
            axis_gains[f] = gain
            axis_thresholds[f] = thresh

        best_axis_idx = axis_gains.argmax().item()
        best_axis_gain = max(0.0, axis_gains[best_axis_idx].item())

        w_axis_best = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
        w_axis_best[best_axis_idx] = 1.0

        # 2. Compute Analytical Covariance Direction (w_cov)
        scores = _sis_scores(X_node, G_node, H_node, k_dom, self.reg_lambda)
        d_sub_val = min(self.d_sub, D)
        top_feat = scores.topk(d_sub_val).indices.tolist()

        X_sub = X_node[:, top_feat]
        g_node = G_node[:, k_dom]
        G_vec = X_sub.T @ g_node
        w_sub = G_vec.clone()
        w_sub = w_sub / (w_sub.norm() + 1e-8)
        
        w_cov = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
        w_cov[top_feat] = w_sub

        # 3. Alpha-Blending Setup: Interpolating Between Optimization (Axis) & Regularization (Cov)
        # alpha = 0.0 -> Pure Axis | alpha = 1.0 -> Pure Covariance
        alpha_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        blend_candidates = []
        for alpha in alpha_grid:
            # Linear combination formulation
            w_blend = alpha * w_cov + (1.0 - alpha) * w_axis_best
            w_blend = w_blend / (w_blend.norm() + 1e-8)
            blend_candidates.append(w_blend)

        # Batch evaluate the blending candidates
        W_blend = torch.stack(blend_candidates)
        proj_blend = X_node @ W_blend.T # (N, len(alpha_grid))

        blend_gains = np.zeros(len(alpha_grid))
        blend_thresholds = np.zeros(len(alpha_grid))

        for idx, alpha in enumerate(alpha_grid):
            b_thresh, b_gain = _best_threshold_tensor(proj_blend[:, idx], G_node, H_node, self.reg_lambda, n_bins=self.n_bins)
            blend_gains[idx] = max(0.0, b_gain.item())
            blend_thresholds[idx] = b_thresh.item()

        # 4. Extract Winner and Log Metrics mapped to Sample Size N
        best_alpha_idx = int(np.argmax(blend_gains))
        winning_alpha = alpha_grid[best_alpha_idx]
        final_gain = blend_gains[best_alpha_idx]
        final_thresh = blend_thresholds[best_alpha_idx]
        final_w = blend_candidates[best_alpha_idx].detach()

        # Append to empirical research database (N, winning_alpha)
        self.empirical_db_.append((N, winning_alpha))

        return final_w, final_thresh, final_gain

    def _build_tree(self, X: torch.Tensor, G: torch.Tensor, H: torch.Tensor, idx: torch.Tensor, depth: int) -> Node:
        n = len(idx)
        if depth >= self.max_depth or n < 6:
            val = self._leaf_value(G[idx], H[idx])
            return Node(is_leaf=True, leaf_value=val)

        w, thresh, gain = self._find_split(X[idx], G[idx], H[idx])

        if gain <= 0:
            val = self._leaf_value(G[idx], H[idx])
            return Node(is_leaf=True, leaf_value=val)

        proj = X[idx] @ w
        left_mask = proj <= thresh
        right_mask = ~left_mask
        left_idx = idx[left_mask]
        right_idx = idx[right_mask]

        if len(left_idx) == 0 or len(right_idx) == 0:
            val = self._leaf_value(G[idx], H[idx])
            return Node(is_leaf=True, leaf_value=val)

        left_node = self._build_tree(X, G, H, left_idx, depth + 1)
        right_node = self._build_tree(X, G, H, right_idx, depth + 1)

        return Node(is_leaf=False, w=w, threshold=thresh, left=left_node, right=right_node, gain=gain)

    def fit(self, X: np.ndarray, y: np.ndarray) -> OQBoostEmpiricalAdvancedAnalyzer:
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        y_t = torch.tensor(np.asarray(y, dtype=np.int64), device=self.device)
        N, D = X_t.shape
        self.K_ = int(y_t.max().item()) + 1

        cnt = torch.bincount(y_t, minlength=self.K_).float()
        lp = torch.log(cnt / N + 1e-8)
        self.F_init_ = lp - lp.mean()

        F = self.F_init_.unsqueeze(0).expand(N, -1).clone()
        oh = torch.zeros(N, self.K_, dtype=torch.float32, device=self.device)
        oh[torch.arange(N, device=self.device), y_t] = 1.0

        self.empirical_db_ = []

        for m in range(self.n_estimators):
            Fs = F - F.max(1, keepdim=True).values
            P = Fs.exp()
            P = P / P.sum(1, keepdim=True)
            G = P - oh
            H = P * (1 - P)

            root = self._build_tree(X_t, G, H, torch.arange(N, device=self.device), depth=0)
            self.trees_.append(root)
            F = F + self.learning_rate * self._predict_tree(root, X_t)

        self._print_empirical_break_even_report()
        return self

    def _print_empirical_break_even_report(self):
        db = self.empirical_db_
        if not db:
            return

        # Define Sample Size Buckets based on N
        # Large (N > 500) | Medium (100 < N <= 500) | Small (N <= 100)
        buckets = {
            "Large Node Scale (N > 500)       ": [],
            "Medium Node Scale (100 < N <= 500)": [],
            "Small Node Scale (N <= 100)      ": []
        }

        for N, alpha in db:
            if N > 500:
                buckets["Large Node Scale (N > 500)       "].append(alpha)
            elif N > 100:
                buckets["Medium Node Scale (100 < N <= 500)"].append(alpha)
            else:
                buckets["Small Node Scale (N <= 100)      "].append(alpha)

        print("\n" + "="*80)
        print("          OQBOOST EMPIRICAL ADVANCED ANALYSIS: BREAK-EVEN REPORT          ")
        print("="*80)
        print(f" Total Decision Splits Tracked: {len(db)}")
        print("-"*80)
        print("  Node Scale Bracket       |  Total Splits  | Mean Alpha (α) Winner | Mode Alpha")
        print("-"*80)

        for b_name, alphas in buckets.items():
            count = len(alphas)
            if count > 0:
                mean_alpha = np.mean(alphas)
                # Compute mode value manually
                vals, counts = np.unique(alphas, return_counts=True)
                mode_alpha = vals[counts.argmax()]
                print(f"  {b_name} |     {count:4d}       |         {mean_alpha:.4f}         |    {mode_alpha:.1f}")
            else:
                print(f"  {b_name} |        0       |         N/A            |    N/A")

        print("="*80)
        print("\n>>> DEDUCTION MATRIX DIRECTIONS:")
        print(" - If α -> 0.0 dominantly at Large N: Optimization guidelines are strictly required at root.")
        print(" - If α -> 1.0 dominantly at Small N: Regularization theory holds true at sparse limits.")
        print("="*80 + "\n")

    def _predict_tree(self, root: Node, X: torch.Tensor) -> torch.Tensor:
        N = len(X)
        out = torch.zeros(N, self.K_, dtype=X.dtype, device=X.device)
        self._route(root, X, torch.arange(N, device=X.device), out)
        return out

    def _route(self, node: Node, X: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> None:
        if node.is_leaf:
            out[idx] += node.leaf_value.unsqueeze(0)
            return
        proj = X[idx] @ node.w
        left = idx[proj <= node.threshold]
        right = idx[proj > node.threshold]
        if len(left) > 0:
            self._route(node.left, X, left, out)
        if len(right) > 0:
            self._route(node.right, X, right, out)


# ─── Execution Pipeline ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # 데이터셋 생성 및 인위적인 회전 주입
    X_raw, y_raw = make_classification(
        n_samples=3000, 
        n_features=30, 
        n_informative=20, 
        n_redundant=10, 
        random_state=42
    )
    np.random.seed(42)
    R, _, _ = np.linalg.svd(np.random.randn(30, 30))
    X_rotated = X_raw @ R

    # 분석기 실행 (20개 에스티메이터, 최대 깊이 5)
    analyzer = OQBoostEmpiricalAdvancedAnalyzer(n_estimators=20, max_depth=5)
    analyzer.fit(X_rotated, y_raw)