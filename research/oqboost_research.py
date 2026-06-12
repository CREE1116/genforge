"""
OQBoost Research Edition — Pure PyTorch implementation for interpretability analysis.

Re-implements the OQBoost algorithm in transparent Python/PyTorch so that every
intermediate quantity (split directions, gains, gradient alignments) can be
inspected and visualized.

Key differences from the C++ production version:
- Pure Python — no compiled extension, every step is inspectable
- WLS direction via torch.linalg.solve (closed-form, same fixed point as CD)
- All split vectors, gains, and routing masks stored for post-hoc analysis
- Analysis hooks: obliqueness profile, direction drift, gradient alignment, feature importance

Usage:
    from research.oqboost_research import OQBoostResearch, analyze

    clf = OQBoostResearch(n_estimators=50, max_depth=4, verbose=True)
    clf.fit(X_train, y_train)
    analyze(clf, X_train, y_train)
"""
from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SplitRecord:
    """Everything about one oblique split — for post-hoc analysis."""
    tree_idx: int
    depth: int
    n_samples: int
    w: torch.Tensor           # (D,) unit-norm split direction
    threshold: float
    gain: float
    top_features: list[int]   # SIS-selected feature indices
    parent_w: Optional[torch.Tensor]
    angle_from_parent: float  # degrees; NaN for root nodes
    angle_from_axis: float    # 0° = axis-aligned, 90° = fully oblique
    is_oblique_winner: bool   # True if WLS direction beat all axis-aligned candidates
    winner_type: str = 'axis' # which candidate won: axis | wls | inherit_A | inherit_B


@dataclass
class Node:
    is_leaf: bool
    leaf_value: Optional[torch.Tensor] = None  # (K,) for leaf nodes
    record: Optional[SplitRecord] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None


# ─── Low-level math ───────────────────────────────────────────────────────────

def _dominant_class(G: torch.Tensor) -> int:
    """Class with largest total absolute gradient mass."""
    return int(G.abs().sum(0).argmax().item())


def _sis_scores(X: torch.Tensor, G: torch.Tensor, H: torch.Tensor,
                k: int, reg_lambda: float) -> torch.Tensor:
    """
    Sure Independence Screening score per feature.

    s_d = |Σ_i x_id g_ik| / sqrt(Σ_i h_ik x_id² + λ)

    Marginal gradient-correlation normalized by hessian-weighted variance.
    High score = feature strongly correlates with gradient of class k.
    """
    g = G[:, k]
    h = H[:, k]
    cg = (X * g.unsqueeze(1)).sum(0)
    add = (h.unsqueeze(1) * X.pow(2)).sum(0)
    return cg.abs() / (add + reg_lambda + 1e-8).sqrt()


def _wls_direction(X: torch.Tensor, G: torch.Tensor, H: torch.Tensor,
                   k: int, reg_lambda: float) -> torch.Tensor:
    """
    Closed-form weighted-least-squares oblique split direction.

    Minimizes: Σ_i [ h_i (w'x_i)²/2 + g_i (w'x_i) ] + λ||w||²/2

    WHY THIS IS THE RIGHT DIRECTION:
      The second-order Taylor expansion of the boosting loss at the current
      predictions F is:  L(F + lr·leaf(x)) ≈ Σ_i [g_i Δ_i + h_i Δ_i²/2]
      A linear predictor Δ_i = w'x_i minimizes this exactly when:
        ∇_w = X'(H·Xw + g) + λw = 0
        → (X'HX + λI)w = -X'g
      This is the Newton step in feature space — the split direction is literally
      the gradient descent direction of the second-order loss approximation.

      CONSEQUENCE: oblique splits align with the geometry of the loss landscape.
      Axis-aligned splits are a restricted case (w = e_d); WLS finds the
      unconstrained optimum.
    """
    g = G[:, k]
    h = H[:, k]

    # Hessian-weighted Gram matrix (d × d)
    XH = X * h.unsqueeze(1)                                          # (N, d)
    A = XH.T @ X + reg_lambda * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)  # (d, d)
    b = -(X.T @ g)                                                   # (d,)

    try:
        w = torch.linalg.solve(A, b)
    except Exception:
        w = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
    return w


def _node_score(G_sum: torch.Tensor, H_sum: torch.Tensor, lam: float) -> float:
    """Per-node gain: Σ_k G_k² / (H_k + λ) / 2  (Newton reduction in loss)."""
    return float((G_sum.pow(2) / (H_sum + lam)).sum() / 2)


def _best_threshold(proj: torch.Tensor, G: torch.Tensor, H: torch.Tensor,
                    lam: float, n_bins: int = 64) -> tuple[float, float]:
    """Scan quantile-spaced thresholds, return (best_threshold, best_gain)."""
    N = len(proj)
    order = proj.argsort()
    proj_s = proj[order]
    G_s = G[order]
    H_s = H[order]

    G_tot = G.sum(0)
    H_tot = H.sum(0)
    root = _node_score(G_tot, H_tot, lam)

    G_left = torch.zeros_like(G_tot)
    H_left = torch.zeros_like(H_tot)

    best_gain = -1e18
    best_t = float(proj_s[0])

    step = max(1, N // n_bins)
    for i in range(step, N - step, step):
        G_left = G_left + G_s[i - step:i].sum(0)
        H_left = H_left + H_s[i - step:i].sum(0)
        G_right = G_tot - G_left
        H_right = H_tot - H_left
        if float(H_left.sum()) < 0.1 or float(H_right.sum()) < 0.1:
            continue
        gain = (_node_score(G_left, H_left, lam) +
                _node_score(G_right, H_right, lam) - root)
        if gain > best_gain:
            best_gain = gain
            best_t = float((proj_s[i - 1] + proj_s[i]) / 2)

    return best_t, best_gain


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    """Unsigned angle in degrees between two vectors."""
    cos = float(torch.clamp(
        (a * b).sum() / (a.norm() * b.norm() + 1e-8), -1.0, 1.0
    ))
    return float(np.degrees(np.arccos(abs(cos))))


def _obliqueness(w: torch.Tensor) -> float:
    """
    Angle in degrees from the nearest axis direction.

    0° = pure axis-aligned split (only one feature used).
    90° = maximally oblique (weight spread uniformly across features).

    Computed as arccos(max_d |w_d| / ||w||).
    """
    norm = float(w.norm())
    if norm < 1e-8:
        return 0.0
    cos_axis = float(w.abs().max()) / norm
    return float(np.degrees(np.arccos(min(cos_axis, 1.0))))


# ─── Single research tree ─────────────────────────────────────────────────────

class OQBoostResearchTree:
    """
    Single oblique boosting tree — pure Python, fully introspectable.

    Matches the algorithmic structure of the C++ engine:
    1. SIS feature screening
    2. WLS oblique direction on top features
    3. Axis-aligned candidates (top features individually)
    4. Parent direction inheritance + mutation (Strategy A and B)
    5. Best-gain winner picked from all candidates
    """

    def __init__(
        self,
        max_depth: int = 4,
        reg_lambda: float = 1.0,
        d_sub: int = 16,
        n_sis: int = 2048,
        inherited_rp_ratio: float = 1.0,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5,
        tree_idx: int = 0,
        use_wls: bool = True,
        inherit_mode: str = 'mutate',  # 'mutate' (A/B) | 'orth' | 'both'
        probe_cones: bool = False,
        orth_strategy: str = 'more',
        budget_strategy: str = 'uniform',
        noise_strategy: str = 'uniform',
        n_random: int = 0,    # sparse random candidates (diversity family)
        n_inherit: int = 4,   # inherited candidates in 'mutate' mode (continuity family)
        dir_cache: Optional[list] = None,
        winning_history: Optional[list] = None,
    ):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.d_sub = d_sub
        self.n_sis = n_sis
        self.inherited_rp_ratio = inherited_rp_ratio
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tree_idx = tree_idx
        self.use_wls = use_wls
        self.inherit_mode = inherit_mode
        self.probe_cones = probe_cones
        self.orth_strategy = orth_strategy
        self.budget_strategy = budget_strategy
        self.noise_strategy = noise_strategy
        self.n_random = n_random
        self.n_inherit = n_inherit
        self.dir_cache = dir_cache
        self.winning_history = winning_history

        self.root_: Optional[Node] = None
        self.split_records_: list[SplitRecord] = []
        # cone probe rows: (depth, gain_near, gain_orth, gain_rand)
        self.cone_log_: list[tuple[int, float, float, float]] = []
        self.D_: int = 0
        self.K_: int = 0

    def cache_direction(self, w: torch.Tensor, initial_score: float = 0.0):
        norm = float(w.norm())
        if norm < 1e-12:
            return
        wn = (w / norm).detach().cpu()

        if self.winning_history is not None:
            self.winning_history.append(wn)
            if len(self.winning_history) > 128:
                self.winning_history.pop(0)

        if self.dir_cache is None:
            return
        for item in self.dir_cache:
            dot = float(abs(torch.dot(item['w'], wn)))
            if dot > 0.95:
                return
        if len(self.dir_cache) < 32:
            self.dir_cache.append({'w': wn, 'score': initial_score})
        else:
            if self.inherit_mode == 'cache_evolutionary':
                min_item = min(self.dir_cache, key=lambda x: x['score'])
                if initial_score > min_item['score']:
                    min_item['w'] = wn
                    min_item['score'] = initial_score
            else:
                self.dir_cache.pop(0)
                self.dir_cache.append({'w': wn, 'score': 0.0})

    def fit_predict(
        self,
        X: torch.Tensor,  # (N, D) — full training set, preprocessed
        G: torch.Tensor,  # (N, K) gradients
        H: torch.Tensor,  # (N, K) hessians
        sub: Optional[torch.Tensor] = None,  # (Ns,) sample indices for building
        seed: int = 42,
    ) -> torch.Tensor:                       # (N, K) leaf predictions
        torch.manual_seed(seed)
        N, D = X.shape
        K = G.shape[1]
        self.D_ = D
        self.K_ = K
        self.split_records_ = []

        if sub is None:
            sub = torch.arange(N, device=X.device)

        self.n_root_ = len(sub)
        out = torch.zeros(N, K, dtype=X.dtype, device=X.device)
        self.root_ = self._build(X, G, H, sub, depth=0, lineage_w=[], out=out)
        return out

    def _build(
        self,
        X: torch.Tensor,
        G: torch.Tensor,
        H: torch.Tensor,
        idx: torch.Tensor,   # global indices of samples in this node
        depth: int,
        lineage_w: list[torch.Tensor],
        out: torch.Tensor,
    ) -> Node:
        n = len(idx)

        if depth >= self.max_depth or n < 4:
            val = self._leaf_value(G[idx], H[idx])
            out[idx] += val.unsqueeze(0)
            return Node(is_leaf=True, leaf_value=val)

        w, thresh, gain, record = self._find_split(
            X[idx], G[idx], H[idx], depth, lineage_w
        )

        if gain <= 0 or record is None:
            val = self._leaf_value(G[idx], H[idx])
            out[idx] += val.unsqueeze(0)
            return Node(is_leaf=True, leaf_value=val)

        self.split_records_.append(record)
        if record.winner_type.startswith('inherit_cache_'):
            parts = record.winner_type.split('_')
            if len(parts) > 2 and parts[2].isdigit():
                cache_idx = int(parts[2])
                if cache_idx < len(self.dir_cache):
                    self.dir_cache[cache_idx]['score'] += gain
        elif record.winner_type != 'axis':
            self.cache_direction(w, initial_score=gain)

        proj = X[idx] @ w
        left_mask = proj <= thresh
        right_mask = ~left_mask
        left_idx = idx[left_mask]
        right_idx = idx[right_mask]

        if len(left_idx) == 0 or len(right_idx) == 0:
            val = self._leaf_value(G[idx], H[idx])
            out[idx] += val.unsqueeze(0)
            return Node(is_leaf=True, leaf_value=val)

        left_child = self._build(X, G, H, left_idx, depth + 1, lineage_w + [w], out)
        right_child = self._build(X, G, H, right_idx, depth + 1, lineage_w + [w], out)
        return Node(is_leaf=False, record=record, left=left_child, right=right_child)

    def _find_split(
        self,
        X_node: torch.Tensor,
        G_node: torch.Tensor,
        H_node: torch.Tensor,
        depth: int,
        lineage_w: list[torch.Tensor],
    ) -> tuple[torch.Tensor, float, float, Optional[SplitRecord]]:
        parent_w = lineage_w[-1] if len(lineage_w) > 0 else None
        N, D = X_node.shape
        k_dom = _dominant_class(G_node)

        # SIS: score features by gradient-hessian correlation
        n_scr = min(N, self.n_sis)
        scr_perm = torch.randperm(N, device=X_node.device)[:n_scr]
        scores = _sis_scores(
            X_node[scr_perm], G_node[scr_perm], H_node[scr_perm],
            k_dom, self.reg_lambda
        )
        d_sub = min(self.d_sub, D)
        top_feat = scores.topk(d_sub).indices.tolist()

        candidates: list[tuple[torch.Tensor, str]] = []

        # 1. Axis-aligned: each top feature individually
        for f in top_feat:
            w = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
            w[f] = 1.0
            candidates.append((w, 'axis'))

        # 2. WLS oblique direction on top features
        if self.use_wls:
            X_sub = X_node[:, top_feat]
            w_wls = _wls_direction(X_sub, G_node, H_node, k_dom, self.reg_lambda)
            w_full = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
            w_full[torch.tensor(top_feat, dtype=torch.long, device=X_node.device)] = w_wls
            norm = float(w_full.norm())
            if norm > 1e-8:
                w_full = w_full / norm
            candidates.append((w_full, 'wls'))

        # 3. Inherited parent direction
        if parent_w is not None and self.inherited_rp_ratio > 0:
            wp = parent_w / (parent_w.norm() + 1e-12)
            support = parent_w.abs() > 1e-8

            # Budget-matched inherited slot: every mode contributes exactly
            # 4 candidates so accuracy comparisons are not confounded by
            # candidate count.
            def make_A():
                decay_a = self.mutation_rate / math.sqrt(1 + depth)
                w_a = parent_w.clone()
                if support.any():
                    w_a[support] = (w_a[support]
                                    + torch.randn(int(support.sum()), device=X_node.device) * decay_a)
                n = float(w_a.norm())
                return (w_a / n, 'inherit_A') if n > 1e-8 else None

            def make_B(rank: int):
                decay_b = self.mutation_strength / (1 + depth)
                w_b = parent_w.clone()
                if rank < len(top_feat):
                    w_b[top_feat[rank]] = w_b[top_feat[rank]] + decay_b
                n = float(w_b.norm())
                return (w_b / n, 'inherit_B') if n > 1e-8 else None

            if self.orth_strategy == 'full':
                beta_val = 1.0
            elif self.orth_strategy == 'more':
                beta_val = float(depth) / (float(depth) + 1.0)
            elif self.orth_strategy == 'less':
                beta_val = 1.0 / (float(depth) + 1.0)
            elif self.orth_strategy == 'random':
                beta_val = 0.0
            else:
                beta_val = 1.0

            def make_O(beta: float = 1.0):
                # Strategy O: search the subspace ORTHOGONAL to the parent
                # direction (within parent support ∪ top SIS features).
                # Rationale: the parent split consumed the gradient signal
                # along w_parent; the residual signal inside each child
                # concentrates in the orthogonal complement (E6 cone probe:
                # orth gain ≈ 2× near gain at matched budgets).
                sup_idx = torch.where(support)[0].tolist()
                sup_union = sorted(set(sup_idx) | set(top_feat[:4]))
                r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                ridx = torch.tensor(sup_union, dtype=torch.long, device=X_node.device)
                r[ridx] = torch.randn(len(sup_union), device=X_node.device)
                r = r - beta * (r @ wp) * wp
                n = float(r.norm())
                return (r / n, 'inherit_O') if n > 1e-8 else None

            if self.inherit_mode == 'mutate':
                makers = []
                for i in range(self.n_inherit):
                    if i % 2 == 0:
                        makers.append(make_A)
                    else:
                        makers.append(lambda r=(i // 2): make_B(r))
            elif self.inherit_mode == 'orth':
                if self.budget_strategy == 'uniform':
                    n_candidates = 8
                elif self.budget_strategy == 'less':
                    n_candidates = max(2, 20 - 6 * depth)
                elif self.budget_strategy == 'more':
                    n_candidates = max(2, 6 * depth - 4)
                else:
                    n_candidates = 8
                makers = [lambda: make_O(beta_val) for _ in range(n_candidates)]
            elif self.inherit_mode == 'noise_mutation':
                if self.noise_strategy == 'uniform':
                    noise_scale = 0.5
                elif self.noise_strategy == 'less':
                    noise_scale = max(0.1, 1.0 - 0.3 * depth)
                elif self.noise_strategy == 'more':
                    noise_scale = min(1.5, 0.1 + 0.3 * depth)
                else:
                    noise_scale = 0.5
                
                def make_noise_mutated(scale: float):
                    sup_idx = torch.where(support)[0].tolist()
                    sup_union = sorted(set(sup_idx) | set(top_feat[:4]))
                    r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                    ridx = torch.tensor(sup_union, dtype=torch.long, device=X_node.device)
                    r[ridx] = torch.randn(len(sup_union), device=X_node.device)
                    rn = r / (r.norm() + 1e-12)
                    w_rand = wp + scale * rn
                    n = float(w_rand.norm())
                    return (w_rand / n, 'inherit_noise') if n > 1e-8 else None
                
                makers = [lambda: make_noise_mutated(noise_scale) for _ in range(8)]
            elif self.inherit_mode == 'cosine_constrained':
                gamma = min(0.99, 0.3 + 0.2 * depth)
                
                def make_cosine_constrained(g: float):
                    cand_orth = make_O(1.0)
                    if cand_orth is None:
                        return None
                    r_orth, _ = cand_orth
                    w_cand = g * wp + math.sqrt(1.0 - g * g) * r_orth
                    n = float(w_cand.norm())
                    return (w_cand / n, 'inherit_cosine') if n > 1e-8 else None
                
                if self.budget_strategy == 'less':
                    n_candidates = max(2, 8 - 2 * depth)
                elif self.budget_strategy == 'more':
                    n_candidates = max(2, 2 * depth + 2)
                elif self.budget_strategy == 'density':
                    n_candidates = max(2, int(round(8 * math.sqrt(1.0 - gamma * gamma))))
                else:
                    n_candidates = 8
                makers = [lambda: make_cosine_constrained(gamma) for _ in range(n_candidates)]
            elif self.inherit_mode == 'cosine_adaptive_size':
                rho = float(N) / float(self.n_root_)
                gamma = 0.99 - 0.69 * rho
                
                def make_cosine_constrained(g: float):
                    cand_orth = make_O(1.0)
                    if cand_orth is None:
                        return None
                    r_orth, _ = cand_orth
                    w_cand = g * wp + math.sqrt(1.0 - g * g) * r_orth
                    n = float(w_cand.norm())
                    return (w_cand / n, 'inherit_cosine') if n > 1e-8 else None
                
                if self.budget_strategy == 'less':
                    n_candidates = max(2, 8 - 2 * depth)
                elif self.budget_strategy == 'more':
                    n_candidates = max(2, 2 * depth + 2)
                elif self.budget_strategy == 'density':
                    n_candidates = max(2, int(round(8 * math.sqrt(1.0 - gamma * gamma))))
                else:
                    n_candidates = 8
                makers = [lambda: make_cosine_constrained(gamma) for _ in range(n_candidates)]
            elif self.inherit_mode == 'cosine_adaptive_gradient':
                G_mean = G_node.mean(dim=0, keepdim=True)
                G_var = ((G_node - G_mean) ** 2).sum()
                G_energy = (G_node ** 2).sum()
                sigma2 = float(G_var / (G_energy + 1e-8))
                gamma = 0.99 - 0.69 * sigma2
                
                def make_cosine_constrained(g: float):
                    cand_orth = make_O(1.0)
                    if cand_orth is None:
                        return None
                    r_orth, _ = cand_orth
                    w_cand = g * wp + math.sqrt(1.0 - g * g) * r_orth
                    n = float(w_cand.norm())
                    return (w_cand / n, 'inherit_cosine') if n > 1e-8 else None
                
                if self.budget_strategy == 'less':
                    n_candidates = max(2, 8 - 2 * depth)
                elif self.budget_strategy == 'more':
                    n_candidates = max(2, 2 * depth + 2)
                elif self.budget_strategy == 'density':
                    n_candidates = max(2, int(round(8 * math.sqrt(1.0 - gamma * gamma))))
                else:
                    n_candidates = 8
                makers = [lambda: make_cosine_constrained(gamma) for _ in range(n_candidates)]
            elif self.inherit_mode in ['cache_blend', 'cache_orb', 'cache_evolutionary']:
                if self.dir_cache and len(self.dir_cache) > 0:
                    def make_cache_blend_or_orb():
                        idx_rnd = torch.randint(0, len(self.dir_cache), (1,)).item()
                        cached_item = self.dir_cache[idx_rnd]
                        cw = cached_item['w'].to(device=X_node.device, dtype=X_node.dtype)
                        alpha = float(torch.rand(1, device=X_node.device).item()) * 0.6 + 0.2
                        
                        if self.inherit_mode == 'cache_blend':
                            w_blend = alpha * wp + (1.0 - alpha) * cw
                            n = float(w_blend.norm())
                            return (w_blend / n, f'inherit_cache_{idx_rnd}') if n > 1e-8 else None
                        else:
                            # ORB: Orthogonal Random Blending
                            # 1. Project cached_w orthogonal to wp
                            cw_orth = cw - (cw @ wp) * wp
                            n_c = float(cw_orth.norm())
                            if n_c > 1e-8:
                                cw_orth = cw_orth / n_c
                            else:
                                cw_orth = cw
                            
                            # 2. Draw fresh random direction
                            sup_idx = torch.where(support)[0].tolist()
                            sup_union = sorted(set(sup_idx) | set(top_feat[:4]))
                            r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                            ridx = torch.tensor(sup_union, dtype=torch.long, device=X_node.device)
                            r[ridx] = torch.randn(len(sup_union), device=X_node.device)
                            rn = r / (r.norm() + 1e-12)
                            
                            # 3. Blend: alpha * random + (1 - alpha) * cached_orth
                            w_blend = alpha * rn + (1.0 - alpha) * cw_orth
                            
                            # 4. Project orthogonal to wp
                            w_final = w_blend - (w_blend @ wp) * wp
                            n = float(w_final.norm())
                            return (w_final / n, f'inherit_cache_{idx_rnd}') if n > 1e-8 else None
                else:
                    def make_cache_blend_or_orb():
                        sup_idx = torch.where(support)[0].tolist()
                        sup_union = sorted(set(sup_idx) | set(top_feat[:4]))
                        r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                        ridx = torch.tensor(sup_union, dtype=torch.long, device=X_node.device)
                        r[ridx] = torch.randn(len(sup_union), device=X_node.device)
                        n = float(r.norm())
                        return (r / n, 'inherit_cache_fallback') if n > 1e-8 else None

                if self.budget_strategy == 'less':
                    n_candidates = max(2, 8 - 2 * depth)
                elif self.budget_strategy == 'density':
                    gamma = 0.5
                    n_candidates = max(2, int(round(8 * math.sqrt(1.0 - gamma * gamma))))
                else:
                    n_candidates = 8
                makers = [make_cache_blend_or_orb for _ in range(n_candidates)]
            elif self.inherit_mode in ['cache_adaptive_blend_size', 'cache_adaptive_blend_gradient', 'cache_adaptive_blend_depth_prop', 'cache_adaptive_blend_depth_inv']:
                if self.inherit_mode == 'cache_adaptive_blend_size':
                    rho = float(N) / float(self.n_root_)
                    alpha = max(0.1, min(0.9, 1.0 - rho))
                elif self.inherit_mode == 'cache_adaptive_blend_gradient':
                    G_mean = G_node.mean(dim=0, keepdim=True)
                    G_var = ((G_node - G_mean) ** 2).sum()
                    G_energy = (G_node ** 2).sum()
                    sigma2 = float(G_var / (G_energy + 1e-8))
                    alpha = max(0.1, min(0.9, 1.0 - sigma2))
                elif self.inherit_mode == 'cache_adaptive_blend_depth_prop':
                    alpha = min(0.9, 0.2 + 0.17 * depth)
                elif self.inherit_mode == 'cache_adaptive_blend_depth_inv':
                    alpha = max(0.1, 0.8 - 0.17 * depth)
                else:
                    alpha = 0.5

                if self.dir_cache and len(self.dir_cache) > 0:
                    def make_cache_adaptive_blend():
                        idx_rnd = torch.randint(0, len(self.dir_cache), (1,)).item()
                        cached_item = self.dir_cache[idx_rnd]
                        cw = cached_item['w'].to(device=X_node.device, dtype=X_node.dtype)
                        w_blend = alpha * wp + (1.0 - alpha) * cw
                        n = float(w_blend.norm())
                        return (w_blend / n, f'inherit_cache_{idx_rnd}') if n > 1e-8 else None
                else:
                    def make_cache_adaptive_blend():
                        sup_idx = torch.where(support)[0].tolist()
                        sup_union = sorted(set(sup_idx) | set(top_feat[:4]))
                        r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                        ridx = torch.tensor(sup_union, dtype=torch.long, device=X_node.device)
                        r[ridx] = torch.randn(len(sup_union), device=X_node.device)
                        n = float(r.norm())
                        return (r / n, 'inherit_cache_fallback') if n > 1e-8 else None

                if self.budget_strategy == 'less':
                    n_candidates = max(2, 8 - 2 * depth)
                elif self.budget_strategy == 'density':
                    n_candidates = max(2, int(round(8 * math.sqrt(1.0 - alpha * alpha))))
                else:
                    n_candidates = 8
                makers = [make_cache_adaptive_blend for _ in range(n_candidates)]
            elif self.inherit_mode == 'random_only':
                def make_random():
                    p_nz = 1.0 / max(2.0, math.sqrt(D))
                    mask = torch.rand(D, device=X_node.device) < p_nz
                    if not mask.any():
                        mask[torch.randint(D, (1,), device=X_node.device)] = True
                    w_r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                    signs = torch.where(
                        torch.rand(int(mask.sum()), device=X_node.device) < 0.5,
                        -1.0, 1.0,
                    )
                    w_r[mask] = signs.to(X_node.dtype)
                    w_r = w_r / w_r.norm()
                    return (w_r, 'random_only')

                n_candidates = 8
                makers = [make_random for _ in range(n_candidates)]
            elif self.inherit_mode == 'cache_pca':
                def make_fallback():
                    p_nz = 1.0 / max(2.0, math.sqrt(D))
                    mask = torch.rand(D, device=X_node.device) < p_nz
                    if not mask.any():
                        mask[torch.randint(D, (1,), device=X_node.device)] = True
                    w_r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                    signs = torch.where(
                        torch.rand(int(mask.sum()), device=X_node.device) < 0.5,
                        -1.0, 1.0,
                    )
                    w_r[mask] = signs.to(X_node.dtype)
                    w_r = w_r / w_r.norm()
                    return (w_r, 'inherit_pca_fallback')

                if self.winning_history and len(self.winning_history) >= 8:
                    try:
                        # Compute SVD on CPU to be robust to device-specific SVD implementation limitations/bugs
                        W_hist = torch.stack(self.winning_history).to(device='cpu', dtype=torch.float32)
                        mean_W = W_hist.mean(dim=0, keepdim=True)
                        W_centered = W_hist - mean_W
                        U, S, Vh = torch.linalg.svd(W_centered, full_matrices=False)
                        
                        S_device = S.to(device=X_node.device, dtype=X_node.dtype)
                        Vh_device = Vh.to(device=X_node.device, dtype=X_node.dtype)
                        S_Vh = S_device.unsqueeze(1) * Vh_device
                        
                        def make_pca_cand():
                            z = torch.randn(len(S), device=X_node.device, dtype=X_node.dtype)
                            w_subspace = z @ S_Vh
                            n_sub = float(w_subspace.norm())
                            if n_sub < 1e-8:
                                w_subspace = torch.randn(D, device=X_node.device, dtype=X_node.dtype)
                                n_sub = float(w_subspace.norm())
                            w_subspace = w_subspace / n_sub
                            
                            p_nz = 1.0 / max(2.0, math.sqrt(D))
                            mask = torch.rand(D, device=X_node.device) < p_nz
                            if not mask.any():
                                mask[torch.randint(D, (1,), device=X_node.device)] = True
                            r_sparse = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                            signs = torch.where(
                                torch.rand(int(mask.sum()), device=X_node.device) < 0.5,
                                -1.0, 1.0,
                            )
                            r_sparse[mask] = signs.to(X_node.dtype)
                            r_sparse = r_sparse / (r_sparse.norm() + 1e-12)
                            
                            beta = 0.1
                            w_blend = (1.0 - beta) * w_subspace + beta * r_sparse
                            w_final = w_blend - (w_blend @ wp) * wp
                            n_final = float(w_final.norm())
                            return (w_final / n_final, 'inherit_pca') if n_final > 1e-8 else None
                        
                        makers = [make_pca_cand for _ in range(8)]
                    except Exception as e:
                        makers = [make_fallback for _ in range(8)]
                else:
                    makers = [make_fallback for _ in range(8)]
            elif self.inherit_mode == 'cache_lineage_orth':
                # Compute orthonormal basis Q of lineage_w on the current device
                Q = []
                for a in lineage_w:
                    q = a.to(device=X_node.device, dtype=X_node.dtype)
                    n_a = q.norm()
                    if n_a > 1e-12:
                        q = q / n_a
                    for prev_q in Q:
                        q = q - torch.dot(q, prev_q) * prev_q
                    n_q = q.norm()
                    if n_q > 1e-12:
                        Q.append(q / n_q)

                if self.dir_cache and len(self.dir_cache) > 0:
                    def make_cache_lineage_orth():
                        idx_rnd = torch.randint(0, len(self.dir_cache), (1,)).item()
                        cached_item = self.dir_cache[idx_rnd]
                        cw = cached_item['w'].to(device=X_node.device, dtype=X_node.dtype)
                        
                        # Project cw orthogonal to the lineage subspace Q
                        cw_orth = cw.clone()
                        for q in Q:
                            cw_orth = cw_orth - torch.dot(cw_orth, q) * q
                        n_c = float(cw_orth.norm())
                        if n_c > 1e-8:
                            cw_orth = cw_orth / n_c
                        else:
                            cw_orth = cw
                        
                        # Draw fresh random direction for blending/exploration
                        sup_idx = torch.where(support)[0].tolist()
                        sup_union = sorted(set(sup_idx) | set(top_feat[:4]))
                        r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                        ridx = torch.tensor(sup_union, dtype=torch.long, device=X_node.device)
                        r[ridx] = torch.randn(len(sup_union), device=X_node.device)
                        
                        # Project r orthogonal to the lineage subspace Q
                        r_orth = r.clone()
                        for q in Q:
                            r_orth = r_orth - torch.dot(r_orth, q) * q
                        n_r = float(r_orth.norm())
                        if n_r > 1e-8:
                            rn = r_orth / n_r
                        else:
                            rn = r / (r.norm() + 1e-12)
                            
                        # Blend: alpha * random + (1 - alpha) * cached_orth
                        alpha = float(torch.rand(1, device=X_node.device).item()) * 0.6 + 0.2
                        w_blend = alpha * rn + (1.0 - alpha) * cw_orth
                        
                        # Project w_blend orthogonal to Q once more to ensure numerical precision
                        w_final = w_blend.clone()
                        for q in Q:
                            w_final = w_final - torch.dot(w_final, q) * q
                        
                        n = float(w_final.norm())
                        return (w_final / n, f'inherit_cache_{idx_rnd}') if n > 1e-8 else None
                    
                    n_candidates = 8
                    makers = [make_cache_lineage_orth for _ in range(n_candidates)]
                else:
                    def make_fallback():
                        p_nz = 1.0 / max(2.0, math.sqrt(D))
                        mask = torch.rand(D, device=X_node.device) < p_nz
                        if not mask.any():
                            mask[torch.randint(D, (1,), device=X_node.device)] = True
                        w_r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                        signs = torch.where(
                            torch.rand(int(mask.sum()), device=X_node.device) < 0.5,
                            -1.0, 1.0,
                        )
                        w_r[mask] = signs.to(X_node.dtype)
                        w_r = w_r / w_r.norm()
                        return (w_r, 'inherit_cache_fallback')
                    
                    makers = [make_fallback for _ in range(8)]
            elif self.inherit_mode == 'pobs':
                sup_idx = torch.where(support)[0].tolist() if parent_w is not None else []
                sup_union = sorted(set(sup_idx) | set(top_feat[:4]))
                D_sup = len(sup_union)
                pobs_cands = []
                while len(pobs_cands) < 8:
                    block_size = min(D_sup, 8 - len(pobs_cands))
                    X_rnd = torch.randn(D_sup, block_size, device=X_node.device, dtype=X_node.dtype)
                    Q, R = torch.linalg.qr(X_rnd)
                    for i in range(block_size):
                        w = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                        ridx = torch.tensor(sup_union, dtype=torch.long, device=X_node.device)
                        w[ridx] = Q[:, i]
                        pobs_cands.append(w)
                makers = [lambda idx=i: (pobs_cands[idx], 'pobs') for i in range(8)]
            elif self.inherit_mode == 'qmc':
                sup_idx = torch.where(support)[0].tolist() if parent_w is not None else []
                sup_union = sorted(set(sup_idx) | set(top_feat[:4]))
                D_sup = len(sup_union)
                node_seed = int((depth * 10000 + len(X_node) + int(torch.randint(0, 100000, (1,)).item())) % (2**30))
                try:
                    engine = torch.quasirandom.SobolEngine(dimension=D_sup, scramble=True, seed=node_seed)
                    samples = engine.draw(8).to(device=X_node.device, dtype=X_node.dtype)
                    samples_clamped = torch.clamp(samples, 1e-6, 1.0 - 1e-6)
                    z = torch.erfinv(2 * samples_clamped - 1.0) * 1.41421356237
                    norms = z.norm(dim=1, keepdim=True) + 1e-12
                    z_normalized = z / norms
                    
                    qmc_cands = []
                    ridx = torch.tensor(sup_union, dtype=torch.long, device=X_node.device)
                    for i in range(8):
                        w = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                        w[ridx] = z_normalized[i]
                        qmc_cands.append(w)
                    makers = [lambda idx=i: (qmc_cands[idx], 'qmc') for i in range(8)]
                except Exception as e:
                    def make_fallback():
                        p_nz = 1.0 / max(2.0, math.sqrt(D))
                        mask = torch.rand(D, device=X_node.device) < p_nz
                        if not mask.any():
                            mask[torch.randint(D, (1,), device=X_node.device)] = True
                        w_r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                        signs = torch.where(
                            torch.rand(int(mask.sum()), device=X_node.device) < 0.5,
                            -1.0, 1.0,
                        )
                        w_r[mask] = signs.to(X_node.dtype)
                        w_r = w_r / w_r.norm()
                        return (w_r, 'qmc_fallback')
                    makers = [make_fallback for _ in range(8)]
            else:  # 'both'
                makers = [make_A, lambda: make_B(0), lambda: make_O(beta_val), lambda: make_O(beta_val)]
            for mk in makers:
                c = mk()
                if c is not None:
                    candidates.append(c)

        # 4. Sparse random directions (diversity family) — parent-independent,
        # mirrors the C++ engine's global random pool: ±1 entries with
        # P(nonzero) = 1/√D, unit-normalized.
        if self.n_random > 0:
            p_nz = 1.0 / max(2.0, math.sqrt(D))
            for _ in range(self.n_random):
                mask = torch.rand(D, device=X_node.device) < p_nz
                if not mask.any():
                    mask[torch.randint(D, (1,), device=X_node.device)] = True
                w_r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                signs = torch.where(
                    torch.rand(int(mask.sum()), device=X_node.device) < 0.5,
                    -1.0, 1.0,
                )
                w_r[mask] = signs.to(X_node.dtype)
                w_r = w_r / w_r.norm()
                candidates.append((w_r, 'random'))

        # Cone probe (mechanism study): with matched budgets, which cone
        # around the parent direction holds the best split gain?
        if self.probe_cones and parent_w is not None:
            wp = parent_w / (parent_w.norm() + 1e-12)
            sup_idx = torch.where(parent_w.abs() > 1e-8)[0].tolist()
            sup_union = sorted(set(sup_idx) | set(top_feat[:4]))
            ridx = torch.tensor(sup_union, dtype=torch.long, device=X_node.device)

            def best_gain_of(group: list[torch.Tensor]) -> float:
                g_best = 0.0
                for w in group:
                    proj = X_node @ w
                    _, g = _best_threshold(proj, G_node, H_node, self.reg_lambda)
                    g_best = max(g_best, g)
                return g_best

            near, orth, rand_ = [], [], []
            for _ in range(8):
                r = torch.zeros(D, dtype=X_node.dtype, device=X_node.device)
                r[ridx] = torch.randn(len(sup_union), device=X_node.device)
                rn = r / (r.norm() + 1e-12)
                w_near = wp + 0.2 * rn                  # ~11° cone
                near.append(w_near / w_near.norm())
                r_o = r - (r @ wp) * wp                 # 90° cone
                if float(r_o.norm()) > 1e-8:
                    orth.append(r_o / r_o.norm())
                rand_.append(rn)                        # unrestricted (in support)
            self.cone_log_.append(
                (depth, best_gain_of(near), best_gain_of(orth), best_gain_of(rand_))
            )

        # Score all candidates: find best threshold + gain for each direction
        best_gain = -1e18
        best_w = candidates[0][0]
        best_thresh = 0.0
        best_type = 'axis'

        # Stack candidate vectors to perform a single batched matrix projection on device
        W = torch.stack([w for w, _ in candidates])
        all_proj = X_node @ W.T

        # Transfer only the final projection results and node metrics to CPU in one go,
        # avoiding all fine-grained CPU-GPU round-trip synchronizations during threshold scanning
        all_proj_cpu = all_proj.cpu()
        G_node_cpu = G_node.cpu()
        H_node_cpu = H_node.cpu()

        for idx_cand, (w, ctype) in enumerate(candidates):
            proj = all_proj_cpu[:, idx_cand]
            thresh, gain = _best_threshold(proj, G_node_cpu, H_node_cpu, self.reg_lambda)
            if gain > best_gain:
                best_gain = gain
                best_w = w
                best_thresh = thresh
                best_type = ctype

        if best_gain <= 0:
            return best_w, best_thresh, best_gain, None

        record = SplitRecord(
            tree_idx=self.tree_idx,
            depth=depth,
            n_samples=N,
            w=best_w.clone(),
            threshold=best_thresh,
            gain=best_gain,
            top_features=top_feat,
            parent_w=parent_w.clone() if parent_w is not None else None,
            angle_from_parent=(
                _angle_deg(best_w, parent_w) if parent_w is not None else float('nan')
            ),
            angle_from_axis=_obliqueness(best_w),
            is_oblique_winner=(best_type == 'wls'),
            winner_type=best_type,
        )
        return best_w, best_thresh, best_gain, record

    def _leaf_value(self, G: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Newton step leaf value: -Σg_k / (Σh_k + λ) per class."""
        return -G.sum(0) / (H.sum(0) + self.reg_lambda)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        N = len(X)
        out = torch.zeros(N, self.K_, dtype=X.dtype, device=X.device)
        if self.root_ is not None:
            self._route(self.root_, X, torch.arange(N, device=X.device), out)
        return out

    def _route(self, node: Node, X: torch.Tensor,
               idx: torch.Tensor, out: torch.Tensor) -> None:
        if node.is_leaf:
            out[idx] += node.leaf_value.unsqueeze(0)
            return
        proj = X[idx] @ node.record.w
        left = idx[proj <= node.record.threshold]
        right = idx[proj > node.record.threshold]
        if len(left):
            self._route(node.left, X, left, out)
        if len(right):
            self._route(node.right, X, right, out)


# ─── Boosting loop ────────────────────────────────────────────────────────────

class OQBoostResearch:
    """
    Research-grade OQBoost: pure PyTorch, fully introspectable.

    Every split in every tree is recorded in `split_records_`.
    Use the analysis methods to understand WHY the model works.

    Parameters
    ----------
    n_estimators : int
        Boosting rounds.
    learning_rate : float
        Shrinkage for each tree's leaf values.
    max_depth : int
        Tree depth (leaf budget = 2^max_depth).
    reg_lambda : float
        L2 regularization on leaf Newton steps and WLS direction solve.
    subsample : float
        Row subsampling fraction per round.
    d_sub : int
        Max features entering one WLS solve (after SIS selection).
    inherited_rp_ratio : float
        Fraction of candidates using parent direction (0 = no inheritance).
    mutation_rate : float
        Strategy A noise scale for axis-maintaining mutation.
    mutation_strength : float
        Strategy B weight for new-axis borrowing.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        reg_lambda: float = 1.0,
        subsample: float = 0.8,
        d_sub: int = 16,
        inherited_rp_ratio: float = 1.0,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5,
        random_state: int = 42,
        verbose: bool = False,
        use_wls: bool = True,
        inherit_mode: str = 'mutate',
        probe_cones: bool = False,
        orth_strategy: str = 'more',
        device: str = 'auto',
        budget_strategy: str = 'uniform',
        noise_strategy: str = 'uniform',
        n_random: int = 0,
        n_inherit: int = 4,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.d_sub = d_sub
        self.inherited_rp_ratio = inherited_rp_ratio
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.random_state = random_state
        self.verbose = verbose
        self.use_wls = use_wls
        self.inherit_mode = inherit_mode
        self.probe_cones = probe_cones
        self.orth_strategy = orth_strategy
        self.budget_strategy = budget_strategy
        self.noise_strategy = noise_strategy
        self.n_random = n_random
        self.n_inherit = n_inherit

        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.trees_: list[OQBoostResearchTree] = []
        self.dir_cache_: list[torch.Tensor] = []
        self.winning_history_: list[torch.Tensor] = []
        self.F_init_: Optional[torch.Tensor] = None
        self.train_losses_: list[float] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OQBoostResearch':
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        y_t = torch.tensor(np.asarray(y, dtype=np.int64), device=self.device)
        N, _ = X_t.shape
        K = int(y_t.max().item()) + 1
        self.classes_ = np.unique(y)

        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.random_state)

        # Log-prior initialization (matches production code)
        cnt = torch.bincount(y_t, minlength=K).float()
        lp = torch.log(cnt / N + 1e-8)
        lp = lp - lp.mean()
        self.F_init_ = lp.clone()

        F = lp.unsqueeze(0).expand(N, -1).clone()
        oh = torch.zeros(N, K, dtype=torch.float32, device=self.device)
        oh[torch.arange(N, device=self.device), y_t] = 1.0

        self.trees_ = []
        self.train_losses_ = []

        for m in range(self.n_estimators):
            # Softmax probabilities and Newton gradients/hessians
            Fs = F - F.max(1, keepdim=True).values
            P = Fs.exp()
            P = P / P.sum(1, keepdim=True)
            G = P - oh            # (N, K)  first-order gradient
            H = P * (1 - P)       # (N, K)  diagonal Hessian approximation

            # Row subsampling
            if self.subsample < 1.0:
                n_sub = max(4, int(N * self.subsample))
                sub = torch.randperm(N, generator=rng, device=self.device)[:n_sub]
            else:
                sub = torch.arange(N, device=self.device)

            tree = OQBoostResearchTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                d_sub=self.d_sub,
                inherited_rp_ratio=self.inherited_rp_ratio,
                mutation_rate=self.mutation_rate,
                mutation_strength=self.mutation_strength,
                tree_idx=m,
                use_wls=self.use_wls,
                inherit_mode=self.inherit_mode,
                probe_cones=self.probe_cones,
                orth_strategy=self.orth_strategy,
                budget_strategy=self.budget_strategy,
                noise_strategy=self.noise_strategy,
                n_random=self.n_random,
                n_inherit=self.n_inherit,
                dir_cache=self.dir_cache_,
                winning_history=self.winning_history_,
            )

            # Build on subsample; predict on full X for F update
            tree.fit_predict(X_t[sub], G[sub], H[sub],
                             seed=m + self.random_state)
            F = F + self.learning_rate * tree.predict(X_t)
            self.trees_.append(tree)

            # Log-loss on training set
            Fs2 = F - F.max(1, keepdim=True).values
            P2 = Fs2.exp()
            P2 = P2 / P2.sum(1, keepdim=True)
            loss = float(-torch.log(P2[torch.arange(N, device=self.device), y_t].clamp(1e-8)).mean())
            self.train_losses_.append(loss)

            if self.verbose:
                acc = float((P2.argmax(1) == y_t).float().mean())
                n_splits = len(tree.split_records_)
                n_oblique = sum(1 for r in tree.split_records_ if r.is_oblique_winner)
                print(f"  [{m+1:3d}] loss={loss:.4f} acc={acc:.4f} "
                      f"splits={n_splits} oblique_wins={n_oblique}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        N = len(X_t)
        F = self.F_init_.to(self.device).unsqueeze(0).expand(N, -1).clone()
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X_t)
        Fs = F - F.max(1, keepdim=True).values
        P = Fs.exp()
        P = P / P.sum(1, keepdim=True)
        return P.detach().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)

    # ─── Analysis methods ──────────────────────────────────────────────────────

    def all_split_records(self) -> list[SplitRecord]:
        """All split records across every tree."""
        return [r for tree in self.trees_ for r in tree.split_records_]

    def all_cone_logs(self) -> list[tuple[int, float, float, float]]:
        """All cone-probe rows (depth, gain_near, gain_orth, gain_rand)."""
        return [row for tree in self.trees_ for row in tree.cone_log_]

    def obliqueness_profile(self) -> dict:
        """
        Distribution of split obliqueness across all trees.

        angle_from_axis: 0° = pure axis-aligned, 90° = maximally oblique.

        WHY THIS MATTERS: if most angles are > 10°, the model is genuinely
        using oblique directions that axis-aligned boosters cannot produce.
        High obliqueness = decision boundaries that cut diagonally through
        the feature space, exploiting correlations.
        """
        records = self.all_split_records()
        angles = np.array([r.angle_from_axis for r in records])
        oblique_wins = sum(r.is_oblique_winner for r in records)
        return {
            'mean_angle_deg': float(np.mean(angles)),
            'median_angle_deg': float(np.median(angles)),
            'pct_gt_10deg': float(np.mean(angles > 10)),
            'pct_gt_30deg': float(np.mean(angles > 30)),
            'wls_win_rate': oblique_wins / max(len(records), 1),
            'angles': angles.tolist(),
        }

    def direction_drift_profile(self) -> dict:
        """
        How much split directions change from parent to child node.

        WHY THIS MATTERS: low drift means the model reuses parent directions
        (exploiting the fact that correlated features remain correlated down
        the tree). This is the inheritance mechanism — it reduces within-tree
        variance by keeping nearby splits consistent.
        """
        records = self.all_split_records()
        angles = [r.angle_from_parent for r in records
                  if not math.isnan(r.angle_from_parent)]
        by_depth: dict[int, list[float]] = {}
        for r in records:
            if not math.isnan(r.angle_from_parent):
                by_depth.setdefault(r.depth, []).append(r.angle_from_parent)
        return {
            'mean_drift_deg': float(np.mean(angles)) if angles else float('nan'),
            'median_drift_deg': float(np.median(angles)) if angles else float('nan'),
            'by_depth': {d: float(np.mean(v)) for d, v in by_depth.items()},
            'angles': angles,
        }

    def feature_importance(
        self,
        feature_names: Optional[list[str]] = None,
        mode: str = 'gain_weighted',
    ) -> dict:
        """
        Feature importance from oblique split directions.

        mode='gain_weighted' : Σ_splits |w_d| * gain  (oblique-aware)
        mode='magnitude'     : Σ_splits |w_d|
        mode='frequency'     : count of splits where feature is non-zero

        WHY OBLIQUE IMPORTANCE DIFFERS FROM AXIS-ALIGNED:
          Standard importance counts only splits where a feature is the split
          variable. Oblique importance captures features that matter only in
          combination: a feature with small individual effect but high w_d in
          many oblique splits shows up here but not in standard importance.
        """
        records = self.all_split_records()
        if not records:
            D = self.trees_[0].D_ if self.trees_ else 0
            imp = np.zeros(D)
        else:
            D = records[0].w.shape[0]
            imp = np.zeros(D)
            for r in records:
                w_abs = r.w.abs().numpy()
                if mode == 'gain_weighted':
                    imp += w_abs * max(r.gain, 0.0)
                elif mode == 'magnitude':
                    imp += w_abs
                elif mode == 'frequency':
                    imp += (w_abs > 1e-8).astype(float)

        total = imp.sum()
        if total > 0:
            imp /= total

        names = feature_names or [f'f{i}' for i in range(D)]
        order = np.argsort(imp)[::-1]
        return {
            'importance': imp,
            'ranked_features': [(names[i], float(imp[i])) for i in order],
        }

    def gain_decomposition(self) -> dict:
        """
        Break down total gain by split type (axis-aligned vs WLS oblique).

        WHY THIS MATTERS: shows how much of the predictive gain comes from
        oblique splits vs what a standard axis-aligned booster would capture.
        If wls_gain_fraction is high, the data has structure that requires
        oblique boundaries.
        """
        records = self.all_split_records()
        axis_gain = sum(r.gain for r in records if not r.is_oblique_winner)
        wls_gain = sum(r.gain for r in records if r.is_oblique_winner)
        total = axis_gain + wls_gain + 1e-12
        return {
            'axis_gain': axis_gain,
            'wls_gain': wls_gain,
            'wls_gain_fraction': wls_gain / total,
            'n_axis_splits': sum(1 for r in records if not r.is_oblique_winner),
            'n_wls_splits': sum(1 for r in records if r.is_oblique_winner),
        }

    def depth_gain_profile(self) -> dict:
        """Average gain per split by tree depth."""
        records = self.all_split_records()
        by_depth: dict[int, list[float]] = {}
        for r in records:
            by_depth.setdefault(r.depth, []).append(r.gain)
        return {d: float(np.mean(v)) for d, v in sorted(by_depth.items())}


# ─── Ablation utilities ───────────────────────────────────────────────────────

def ablation_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 50,
    max_depth: int = 4,
) -> dict:
    """
    Compare accuracy of three configurations to isolate what helps:

    1. axis_only     — inherited_rp_ratio=0, d_sub=1  (standard GBDT)
    2. oblique_only  — inherited_rp_ratio=0           (oblique, no inheritance)
    3. full_oqboost  — full algorithm with inheritance
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    results = {}
    configs = {
        'axis_only': dict(inherited_rp_ratio=0.0, d_sub=1),
        'oblique_no_inherit': dict(inherited_rp_ratio=0.0, d_sub=16),
        'full_oqboost': dict(inherited_rp_ratio=1.0, d_sub=16),
    }
    for name, extra in configs.items():
        clf = OQBoostResearch(
            n_estimators=n_estimators,
            max_depth=max_depth,
            **extra,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)
        acc = float(accuracy_score(y_test, pred))
        K = proba.shape[1]
        if K == 2:
            auc = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            try:
                auc = float(roc_auc_score(y_test, proba, multi_class='ovr'))
            except Exception:
                auc = float('nan')
        results[name] = {'accuracy': acc, 'auc': auc}
        print(f"  {name:25s}  acc={acc:.4f}  auc={auc:.4f}")
    return results


# ─── Top-level analysis helper ────────────────────────────────────────────────

def analyze(
    clf: OQBoostResearch,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> None:
    """Print a full analysis report of a fitted OQBoostResearch model."""
    print("\n" + "=" * 60)
    print("  OQBoost Research Analysis")
    print("=" * 60)

    # Obliqueness
    op = clf.obliqueness_profile()
    print(f"\n[Obliqueness]")
    print(f"  mean angle from axis : {op['mean_angle_deg']:.1f}°")
    print(f"  median angle         : {op['median_angle_deg']:.1f}°")
    print(f"  splits > 10°         : {op['pct_gt_10deg']*100:.1f}%")
    print(f"  splits > 30°         : {op['pct_gt_30deg']*100:.1f}%")
    print(f"  WLS wins rate        : {op['wls_win_rate']*100:.1f}%")

    # Direction drift
    dd = clf.direction_drift_profile()
    print(f"\n[Direction Drift (parent→child)]")
    print(f"  mean drift  : {dd['mean_drift_deg']:.1f}°")
    print(f"  median drift: {dd['median_drift_deg']:.1f}°")
    if dd['by_depth']:
        print("  by depth    : " +
              " | ".join(f"d{d}={v:.1f}°" for d, v in dd['by_depth'].items()))

    # Gain decomposition
    gd = clf.gain_decomposition()
    print(f"\n[Gain Decomposition]")
    print(f"  axis-aligned splits : {gd['n_axis_splits']}  gain={gd['axis_gain']:.2f}")
    print(f"  WLS oblique splits  : {gd['n_wls_splits']}  gain={gd['wls_gain']:.2f}")
    print(f"  WLS gain fraction   : {gd['wls_gain_fraction']*100:.1f}%")

    # Depth gain profile
    dgp = clf.depth_gain_profile()
    print(f"\n[Mean Gain by Depth]")
    for d, g in dgp.items():
        print(f"  depth {d}: {g:.4f}")

    # Feature importance (top 10)
    fi = clf.feature_importance(feature_names=feature_names, mode='gain_weighted')
    print(f"\n[Top-10 Feature Importance (gain-weighted oblique)]")
    for name, imp in fi['ranked_features'][:10]:
        bar = '█' * int(imp * 50)
        print(f"  {name:20s} {imp:.4f} {bar}")

    # Accuracy
    pred = clf.predict(X)
    acc = float((pred == y).mean())
    print(f"\n[Training Accuracy] {acc:.4f}")
    print(f"[Total splits] {len(clf.all_split_records())}")
    print("=" * 60 + "\n")
