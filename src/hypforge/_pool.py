import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import numpy as np
import torch


# ── Hypothesis ────────────────────────────────────────────────────────────────

class Hypothesis:
    """
    A projection direction used as an oblique split candidate.
    hyp_type: 'linear' | 'square' | 'abs' | 'product'
    """

    __slots__ = (
        "hyp_type", "w", "h1", "h2",
        "fitness", "score", "use_count", "rounds_since_last_use", "is_base",
        "cache",
        "full_cache", "full_cache_cpu",
        "thresholds", "thresholds_cpu",
    )

    def __init__(self, hyp_type="linear", w=None, h1=None, h2=None):
        self.hyp_type = hyp_type
        self.w  = w
        self.h1 = h1
        self.h2 = h2

        self.fitness = 0.0
        self.score   = 0.0
        self.use_count = 0
        self.rounds_since_last_use = 0
        self.is_base = False

        self.cache          = None
        self.full_cache     = None
        self.full_cache_cpu = None
        self.thresholds     = None
        self.thresholds_cpu = None

    # ── complexity ────────────────────────────────────────────────────────────

    def complexity(self) -> int:
        if self.hyp_type == "product":
            return self.h1.complexity() + self.h2.complexity()
        return int((self.w.abs() > 1e-3).sum().item())

    # ── evaluation ───────────────────────────────────────────────────────────

    def eval(self, X: torch.Tensor) -> torch.Tensor:
        """Raw projection on X [N, D] → [N]."""
        if self.hyp_type == "linear":
            return X @ self.w
        elif self.hyp_type == "square":
            return (X @ self.w) ** 2
        elif self.hyp_type == "abs":
            return (X @ self.w).abs()
        elif self.hyp_type == "product":
            return self.h1.eval(X) * self.h2.eval(X)
        raise ValueError(f"Unknown hyp_type: {self.hyp_type}")

    # ── cache lifecycle ───────────────────────────────────────────────────────

    def initialize_cache(self, X_full: torch.Tensor):
        self.full_cache = self.eval(X_full)
        N = self.full_cache.shape[0]
        q = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            dtype=self.full_cache.dtype, device=self.full_cache.device,
        )
        sample = self.full_cache
        if N > 10000:
            idxs   = torch.randperm(N, device=self.full_cache.device)[:10000]
            sample = self.full_cache[idxs]
        self.thresholds     = torch.quantile(sample, q)
        # CPU copies: computed once at hypothesis init, eliminates per-round transfer
        self.full_cache_cpu = self.full_cache.cpu().numpy()
        self.thresholds_cpu = self.thresholds.cpu().numpy()

    def clear_runtime_caches(self):
        self.full_cache     = None
        self.full_cache_cpu = None
        self.thresholds     = None
        self.thresholds_cpu = None
        self.cache          = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_gain_fitness_impl(
    cs: torch.Tensor, G: torch.Tensor, H: torch.Tensor, reg_lambda: float
) -> torch.Tensor:
    P, ns    = cs.shape
    device   = cs.device
    G_total  = G.sum(dim=0)
    H_total  = H.sum(dim=0)

    q        = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], dtype=cs.dtype, device=device)
    theta    = torch.quantile(cs, q, dim=1)   # [9, P]

    cs_exp   = cs.unsqueeze(0)                # [1, P, ns]
    lm_all   = cs_exp < theta.unsqueeze(2)    # [9, P, ns]
    nl       = lm_all.sum(2)
    valid    = (nl >= 10) & (ns - nl >= 10)

    gains_list = []
    for qi in range(9):
        lm_q  = lm_all[qi].to(dtype=G.dtype)
        G_L   = lm_q @ G
        H_L   = lm_q @ H
        G_R   = G_total - G_L
        H_R   = H_total - H_L
        gnum  = (G_L**2 / (H_L + reg_lambda)).sum(1) + (G_R**2 / (H_R + reg_lambda)).sum(1) - (G_total**2 / (H_total + reg_lambda)).sum()
        gains_list.append(torch.where(valid[qi], 0.5 * gnum, torch.full_like(gnum, -1e9)))
    return torch.stack(gains_list).max(0).values.clamp(min=0.0)


try:
    _split_gain_fitness = torch.compile(_split_gain_fitness_impl, backend="aot_eager", fullgraph=False)
except Exception:
    _split_gain_fitness = _split_gain_fitness_impl


def _sparsify(w: torch.Tensor, D_num: int) -> torch.Tensor:
    dev = w.device
    D   = w.shape[0]

    w = w.clone()
    if D_num < D:
        w[D_num:] = 0.0

    k = max(2, int(np.sqrt(D_num)))
    if D_num > k:
        _, indices  = torch.topk(w[:D_num].abs(), k)
        mask        = torch.zeros(D_num, device=dev)
        mask[indices] = 1.0
        w[:D_num]  *= mask

    norm = torch.linalg.norm(w)
    if norm > 1e-5:
        return w / norm
    fallback = torch.zeros(D, device=dev)
    fallback[int(torch.randint(0, D_num, (1,)).item())] = 1.0
    return fallback


# ── HypForgePool ──────────────────────────────────────────────────────────────

class HypForgePool:
    def __init__(self, D: int, max_size: int = 500, dev: str = "cpu"):
        self.D        = D
        self.max_size = max_size
        self.dev      = torch.device(dev)

        self.pop: list[Hypothesis] = []
        for j in range(D):
            w = torch.zeros(D, device=self.dev)
            w[j] = 1.0
            h = Hypothesis(hyp_type="linear", w=w)
            h.is_base = True
            self.pop.append(h)

    def evolve(
        self,
        X_full: torch.Tensor,
        G_full: torch.Tensor,
        H_full: torch.Tensor,
        sub_indices: torch.Tensor,
        D_num: int,
        reg_lambda: float = 1.0,
        eta_penalty: float = 0.002,
    ):
        dev    = X_full.device
        P      = len(self.pop)
        if P == 0:
            return

        X_sub = X_full[sub_indices]
        G_s   = G_full[sub_indices]
        H_s   = H_full[sub_indices]

        for h in self.pop:
            if h.full_cache is None:
                h.initialize_cache(X_full)
            h.cache = h.full_cache[sub_indices]

        caches    = torch.stack([h.cache for h in self.pop])
        fitnesses = _split_gain_fitness(caches, G_s, H_s, reg_lambda)
        for i, h in enumerate(self.pop):
            h.fitness = float(fitnesses[i].item())

        for h in self.pop:
            h.score = h.fitness - eta_penalty * h.complexity()
        self.pop.sort(key=lambda h: h.score, reverse=True)

        raw_candidates: list[Hypothesis] = []

        # Gradient flow hypotheses
        V = X_sub.T @ G_s
        K = G_s.shape[1]
        for k in range(K):
            w_k = _sparsify(V[:, k], D_num)
            for ht in ("linear", "square", "abs"):
                raw_candidates.append(Hypothesis(hyp_type=ht, w=w_k))

        # SVD projections
        try:
            U, _, _ = torch.linalg.svd(V.cpu())
            for j in range(min(2, self.D)):
                w_j = _sparsify(U[:, j].to(dev), D_num)
                for ht in ("linear", "square", "abs"):
                    raw_candidates.append(Hypothesis(hyp_type=ht, w=w_j))
        except Exception:
            pass

        # Gradient ascent on existing pool members
        flow = [h for h in self.pop[:30] if h.hyp_type in ("linear", "square", "abs")]
        if flow:
            W_flow = torch.stack([h.w for h in flow], dim=1)
            P_flow = X_sub @ W_flow
            for idx, h in enumerate(flow):
                z = h.cache
                p = P_flow[:, idx]
                v = G_s.T @ z
                g = G_s @ v
                if   h.hyp_type == "linear": grad_w = X_sub.T @ (g - z)
                elif h.hyp_type == "square": grad_w = 2.0 * X_sub.T @ ((g - z) * p)
                elif h.hyp_type == "abs":    grad_w = X_sub.T @ ((g - z) * p.sign())
                else:                        continue
                raw_candidates.append(Hypothesis(hyp_type=h.hyp_type, w=_sparsify(h.w + 0.1 * grad_w, D_num)))

        # Product hypotheses (synergy)
        non_prod = [h for h in self.pop if h.hyp_type in ("linear", "square", "abs")]
        M = min(15, len(non_prod))
        for i in range(M):
            for j in range(i + 1, M):
                raw_candidates.append(Hypothesis(hyp_type="product", h1=non_prod[i], h2=non_prod[j]))

        # Blind-spot exploration (min-eigenvalue direction)
        vectors = [h.w for h in self.pop if h.hyp_type in ("linear", "square", "abs")]
        if vectors:
            W  = torch.stack(vectors, dim=1)
            C  = W @ W.T
            try:
                _, evecs = torch.linalg.eigh(C.cpu())
                w_blind  = _sparsify(evecs[:, 0].to(dev), D_num)
                if not any(
                    h.hyp_type in ("linear", "square", "abs") and torch.dot(h.w, w_blind).abs().item() > 0.98
                    for h in self.pop
                ):
                    raw_candidates.append(Hypothesis(hyp_type="linear",  w=w_blind))
                    raw_candidates.append(Hypothesis(hyp_type="square", w=w_blind))
            except Exception:
                pass

        if not raw_candidates:
            return

        vec_cands  = [c for c in raw_candidates if c.hyp_type in ("linear", "square", "abs")]
        prod_cands = [c for c in raw_candidates if c.hyp_type == "product"]

        if vec_cands:
            W_c    = torch.stack([c.w for c in vec_cands], dim=1)
            proj_c = X_sub @ W_c
            for idx, c in enumerate(vec_cands):
                p = proj_c[:, idx]
                if   c.hyp_type == "linear": c.cache = p
                elif c.hyp_type == "square": c.cache = p ** 2
                elif c.hyp_type == "abs":    c.cache = p.abs()
        for c in prod_cands:
            c.cache = c.h1.cache * c.h2.cache

        all_caches  = torch.stack([c.cache for c in raw_candidates])
        all_fitness = _split_gain_fitness(all_caches, G_s, H_s, reg_lambda)
        for idx, c in enumerate(raw_candidates):
            c.fitness = float(all_fitness[idx].item())
            c.score   = c.fitness - eta_penalty * c.complexity()
            if c.fitness > 1e-5:
                c.initialize_cache(X_full)
                self.pop.append(c)

        for h in self.pop:
            h.score = h.fitness + 0.05 * h.use_count - eta_penalty * h.complexity()
        self.pop.sort(key=lambda h: h.score, reverse=True)

        # Diversity pruning
        for h in self.pop:
            h.cache = h.full_cache[sub_indices]
        caches  = torch.stack([h.cache for h in self.pop])
        norms   = torch.linalg.norm(caches, dim=1, keepdim=True)
        normed  = (caches / (norms + 1e-8)).cpu().numpy()
        sim_mat = np.abs(normed @ normed.T)

        pruned, kept = [], []
        for i, h in enumerate(self.pop):
            if any(sim_mat[i, j] > 0.90 for j in kept):
                continue
            pruned.append(h)
            kept.append(i)
            if len(pruned) >= self.max_size:
                break
        self.pop = pruned

        self.pop = [h for h in self.pop if h.is_base or h.rounds_since_last_use <= 15]