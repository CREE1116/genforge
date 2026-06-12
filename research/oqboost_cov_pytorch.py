from __future__ import annotations

import time
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional


# ─── Data structures ──────────────────────────────────────────────────────────

class Node:
    """Represents a single node in our OQBoost Covariance Tree."""
    def __init__(
        self,
        is_leaf: bool,
        leaf_value: Optional[torch.Tensor] = None,  # (K,) tensor
        w: Optional[torch.Tensor] = None,           # (D,) unit-norm direction tensor
        threshold: float = 0.0,
        left: Optional[Node] = None,
        right: Optional[Node] = None,
        winner_type: str = 'axis',
        gain: float = 0.0,
    ):
        self.is_leaf = is_leaf
        self.leaf_value = leaf_value
        self.w = w
        self.threshold = threshold
        self.left = left
        self.right = right
        self.winner_type = winner_type
        self.gain = gain


# ─── Low-level helper functions ───────────────────────────────────────────────

def _dominant_class(G: torch.Tensor) -> int:
    """Class with the largest absolute gradient sum."""
    return int(G.abs().sum(0).argmax().item())


def _sis_scores(
    X: torch.Tensor,
    G: torch.Tensor,
    H: torch.Tensor,
    k: int,
    reg_lambda: float
) -> torch.Tensor:
    """
    Sure Independence Screening score per feature.
    s_d = |Σ_i x_id g_ik| / sqrt(Σ_i h_ik x_id² + λ)
    """
    g = G[:, k]
    h = H[:, k]
    cg = (X * g.unsqueeze(1)).sum(0)
    add = (h.unsqueeze(1) * X.pow(2)).sum(0)
    return cg.abs() / (add + reg_lambda + 1e-8).sqrt()


def _best_threshold_tensor(
    proj: torch.Tensor,
    G: torch.Tensor,
    H: torch.Tensor,
    lam: float,
    n_bins: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure tensor-level threshold scanning to avoid GPU-CPU sync bottlenecks.
    Returns (best_threshold_tensor, best_gain_tensor) on the same device.
    """
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


# ─── OQBoostCovClassifier ─────────────────────────────────────────────────────

class OQBoostCovClassifier(BaseEstimator, ClassifierMixin):
    """
    OQBoostCovClassifier: Extended to measure Gain(w_cov) / Gain(w_tournament)
    to mathematically verify proxy fidelity at every node and depth.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        reg_lambda: float = 1.0,
        subsample: float = 0.8,
        d_sub: int = 16,
        random_state: int = 42,
        verbose: bool = False,
        device: str = 'auto',
        n_bins: int = 64,
        n_tournament_candidates: int = 32  # Number of exhaustive search candidates to simulate
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.d_sub = d_sub
        self.random_state = random_state
        self.verbose = verbose
        self.device_setting = device
        self.n_bins = n_bins
        self.n_tournament_candidates = n_tournament_candidates

        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.trees_: list[Node] = []
        self.F_init_: Optional[torch.Tensor] = None
        self.classes_: Optional[np.ndarray] = None
        self.K_: int = 0

        # Fidelity Ratio Tracker
        self.fidelity_tracker_ = {}

    def _init_fidelity_tracker(self):
        self.fidelity_tracker_ = {
            d: {
                'ratios': [],
                'w_cov_gains': [],
                'w_tour_gains': []
            } for d in range(self.max_depth)
        }

    def _leaf_value(self, G: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return -G.sum(0) / (H.sum(0) + self.reg_lambda)

    def _find_split(
        self,
        X_node: torch.Tensor,
        G_node: torch.Tensor,
        H_node: torch.Tensor,
        depth: int
    ) -> tuple[torch.Tensor, float, float, str]:
        N, D = X_node.shape
        k_dom = _dominant_class(G_node)

        # SIS feature screening
        n_scr = min(N, 1000)
        if N > n_scr:
            scr_perm = torch.randperm(N, device=X_node.device)[:n_scr]
            X_scr = X_node[scr_perm]
            G_scr = G_node[scr_perm]
            H_scr = H_node[scr_perm]
        else:
            X_scr, G_scr, H_scr = X_node, G_node, H_node

        scores = _sis_scores(X_scr, G_scr, H_scr, k_dom, self.reg_lambda)
        d_sub_val = min(self.d_sub, D)
        top_feat = scores.topk(d_sub_val).indices.tolist()

        # 1. Generate exact Covariance Proxy candidate
        X_sub = X_node[:, top_feat]
        g_node = G_node[:, k_dom]
        G_vec = X_sub.T @ g_node
        w_sub = G_vec.clone()
        w_sub = w_sub / (w_sub.norm() + 1e-8)
        
        w_cov = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
        w_cov[top_feat] = w_sub

        # 2. Simulate Exhaustive Tournament Pool (Old GG-SRP Style)
        # Generate random sign-aligned directions within the same subspace
        tour_candidates = []
        sign_base = -torch.sign(G_vec)
        
        for _ in range(self.n_tournament_candidates):
            r = torch.randn(d_sub_val, dtype=X_node.dtype, device=X_node.device)
            w_rand_sub = sign_base * r.abs()
            w_rand_sub = w_rand_sub / (w_rand_sub.norm() + 1e-8)
            
            w_rand = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
            w_rand[top_feat] = w_rand_sub
            tour_candidates.append(w_rand)

        # 3. Standard Axis Candidates for structural fallback
        axis_candidates = []
        eye_mat = torch.eye(D, dtype=X_node.dtype, device=X_node.device)
        for f in range(D):
            axis_candidates.append(eye_mat[f])

        # 4. Batch Evaluate All Categories to compute Gains
        all_candidates = axis_candidates + [w_cov] + tour_candidates
        W = torch.stack(all_candidates)
        all_proj = X_node @ W.T

        num_total = len(all_candidates)
        best_gains = torch.empty(num_total, device=X_node.device)
        best_thresholds = torch.empty(num_total, device=X_node.device)

        for idx_cand in range(num_total):
            proj = all_proj[:, idx_cand]
            thresh, gain = _best_threshold_tensor(proj, G_node, H_node, self.reg_lambda, n_bins=self.n_bins)
            best_gains[idx_cand] = gain
            best_thresholds[idx_cand] = thresh

        # ─── Compute the Fidelity Ratio Metric ──────────────────────────────────
        w_cov_gain = max(0.0, best_gains[D].item()) # Index D is w_cov
        
        tour_gains = best_gains[D+1:] # Indices after D are tournament candidates
        w_tour_max_gain = max(0.0, tour_gains.max().item())

        if w_tour_max_gain > 0 and w_cov_gain > 0:
            ratio = w_cov_gain / w_tour_max_gain
            if depth in self.fidelity_tracker_:
                self.fidelity_tracker_[depth]['ratios'].append(ratio)
                self.fidelity_tracker_[depth]['w_cov_gains'].append(w_cov_gain)
                self.fidelity_tracker_[depth]['w_tour_gains'].append(w_tour_max_gain)
        # ────────────────────────────────────────────────────────────────────────

        # Determine true split winner among Axis + Cov (Production rule)
        prod_gains = best_gains[:D+1]
        winner_idx = prod_gains.argmax().item()

        final_w = all_candidates[winner_idx].detach()
        final_thresh = float(best_thresholds[winner_idx].item())
        final_gain = float(best_gains[winner_idx].item())
        final_type = 'axis' if winner_idx < D else 'oblique_cov'

        return final_w, final_thresh, final_gain, final_type

    def _build_tree(
        self,
        X: torch.Tensor,
        G: torch.Tensor,
        H: torch.Tensor,
        idx: torch.Tensor,
        depth: int
    ) -> Node:
        n = len(idx)
        if depth >= self.max_depth or n < 4:
            val = self._leaf_value(G[idx], H[idx])
            return Node(is_leaf=True, leaf_value=val)

        w, thresh, gain, winner_type = self._find_split(X[idx], G[idx], H[idx], depth)

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

        return Node(
            is_leaf=False,
            w=w,
            threshold=thresh,
            left=left_node,
            right=right_node,
            winner_type=winner_type,
            gain=gain
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list[tuple]] = None,
        early_stopping_rounds: Optional[int] = None
    ) -> OQBoostCovClassifier:
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        y_t = torch.tensor(np.asarray(y, dtype=np.int64), device=self.device)
        N, D = X_t.shape
        self.K_ = int(y_t.max().item()) + 1
        self.classes_ = np.unique(y)

        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.random_state)

        # Initialize trackers
        self._init_fidelity_tracker()

        # Log-prior initialization
        cnt = torch.bincount(y_t, minlength=self.K_).float()
        lp = torch.log(cnt / N + 1e-8)
        lp = lp - lp.mean()
        self.F_init_ = lp.clone()

        F = lp.unsqueeze(0).expand(N, -1).clone()
        oh = torch.zeros(N, self.K_, dtype=torch.float32, device=self.device)
        oh[torch.arange(N, device=self.device), y_t] = 1.0

        X_val_t, y_val_t, F_val = None, None, None
        if eval_set:
            X_val, y_val = eval_set[0]
            X_val_t = torch.tensor(np.asarray(X_val, dtype=np.float32), device=self.device)
            y_val_t = torch.tensor(np.asarray(y_val, dtype=np.int64), device=self.device)
            F_val = lp.unsqueeze(0).expand(X_val_t.shape[0], -1).clone()

        self.trees_ = []
        best_val_loss = float('inf')
        best_trees = []
        no_improv = 0

        for m in range(self.n_estimators):
            Fs = F - F.max(1, keepdim=True).values
            P = Fs.exp()
            P = P / P.sum(1, keepdim=True)
            G = P - oh
            H = P * (1 - P)

            if self.subsample < 1.0:
                n_sub = max(4, int(N * self.subsample))
                sub = torch.randperm(N, generator=rng, device=self.device)[:n_sub]
            else:
                sub = torch.arange(N, device=self.device)

            root = self._build_tree(X_t[sub], G[sub], H[sub], torch.arange(len(sub), device=self.device), depth=0)
            self.trees_.append(root)

            F = F + self.learning_rate * self._predict_tree(root, X_t)

            val_str = ""
            if X_val_t is not None and y_val_t is not None:
                F_val = F_val + self.learning_rate * self._predict_tree(root, X_val_t)
                Fv_sh = F_val - F_val.max(1, keepdim=True).values
                P_val = Fv_sh.exp()
                P_val = P_val / P_val.sum(1, keepdim=True)
                
                val_loss = float(
                    -torch.log(P_val[torch.arange(len(y_val_t), device=self.device), y_val_t].clip(1e-8)).mean().item()
                )
                val_acc = float((P_val.argmax(1) == y_val_t).float().mean().item())
                val_str = f" | ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_trees = list(self.trees_)
                    no_improv = 0
                else:
                    no_improv += 1

            if self.verbose:
                ll = -torch.log(P[torch.arange(N, device=self.device), y_t].clip(1e-8)).mean().item()
                acc = (P.argmax(1) == y_t).float().mean().item()
                print(f"  Round {m+1:3d} | Loss={ll:.4f} | Acc={acc:.4f}{val_str}")

            if early_stopping_rounds is not None and no_improv >= early_stopping_rounds:
                self.trees_ = best_trees
                break

        # ─── Print Comprehensive Fidelity Report ────────────────────────────────
        print("\n" + "="*75)
        print("            OQBOOST PROXY FIDELITY MATHEMATICAL REPORT             ")
        print("="*75)
        print(" Depth | Meas. Nodes | Mean Ratio (ρ) | Median ρ |  Max ρ  | Mean Gain Delta")
        print("-"*75)
        
        all_ratios = []
        for d in range(self.max_depth):
            stats = self.fidelity_tracker_[d]
            ratios = stats['ratios']
            n_nodes = len(ratios)
            if n_nodes > 0:
                mean_r = np.mean(ratios)
                med_r = np.median(ratios)
                max_r = np.max(ratios)
                mean_delta = np.mean(np.array(stats['w_cov_gains']) - np.array(stats['w_tour_gains']))
                all_ratios.extend(ratios)
                print(f"  d={d}  |    {n_nodes:4d}    |     {mean_r:.4f}     |  {med_r:.4f}  | {max_r:.4f} |    {mean_delta:+.4f}")
            else:
                print(f"  d={d}  |       0    |       N/A      |    N/A   |   N/A   |       N/A")
        
        if all_ratios:
            print("-"*75)
            print(f"  GLOBAL ENSEMBLE PROXY FIDELITY (Mean ρ): {np.mean(all_ratios):.4f}")
        print("="*75 + "\n")

        return self

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        N = X_t.shape[0]

        F = self.F_init_.unsqueeze(0).expand(N, -1).clone()
        for tree in self.trees_:
            F = F + self.learning_rate * self._predict_tree(tree, X_t)

        Fs = F - F.max(1, keepdim=True).values
        P = Fs.exp()
        P = P / P.sum(1, keepdim=True)
        return P.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)