# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import numpy as np
import torch
cimport numpy as cnp
from libc.math cimport sqrt as c_sqrt


# ── Hypothesis ────────────────────────────────────────────────────────────────

class Hypothesis:
    """Oblique split candidate: a projection direction + type."""

    def __init__(self, hyp_type="linear", w=None, h1=None, h2=None):
        self.hyp_type              = hyp_type
        self.w                     = w
        self.h1                    = h1
        self.h2                    = h2
        self.fitness               = 0.0
        self.score                 = 0.0
        self.use_count             = 0
        self.rounds_since_last_use = 0
        self.is_base               = False
        self.cache                 = None
        self.full_cache            = None
        self.full_cache_cpu        = None
        self.thresholds            = None
        self.thresholds_cpu        = None

    def complexity(self):
        if self.hyp_type == "product":
            return self.h1.complexity() + self.h2.complexity()
        return int((self.w.abs() > 1e-3).sum().item())

    def eval(self, X):
        if self.hyp_type == "linear":
            return X @ self.w
        elif self.hyp_type == "square":
            return (X @ self.w) ** 2
        elif self.hyp_type == "abs":
            return (X @ self.w).abs()
        elif self.hyp_type == "product":
            return self.h1.eval(X) * self.h2.eval(X)
        raise ValueError(f"Unknown hyp_type: {self.hyp_type}")

    def initialize_cache(self, X_full):
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
        self.full_cache_cpu = self.full_cache.cpu().numpy()
        self.thresholds_cpu = self.thresholds.cpu().numpy()

    def clear_runtime_caches(self):
        self.full_cache     = None
        self.full_cache_cpu = None
        self.thresholds     = None
        self.thresholds_cpu = None
        self.cache          = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_gain_fitness_impl(cs, G, H, double reg_lambda):
    cdef int P = cs.shape[0]
    cdef int ns = cs.shape[1]
    cdef int qi

    device  = cs.device
    G_total = G.sum(dim=0)
    H_total = H.sum(dim=0)

    q       = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], dtype=cs.dtype, device=device)
    theta   = torch.quantile(cs, q, dim=1)
    cs_exp  = cs.unsqueeze(0)
    lm_all  = cs_exp < theta.unsqueeze(2)
    nl      = lm_all.sum(2)
    valid   = (nl >= 10) & (ns - nl >= 10)

    gains_list = []
    for qi in range(9):
        lm_q  = lm_all[qi].to(dtype=G.dtype)
        G_L   = lm_q @ G
        H_L   = lm_q @ H
        G_R   = G_total - G_L
        H_R   = H_total - H_L
        gnum  = ((G_L**2 / (H_L + reg_lambda)).sum(1)
                 + (G_R**2 / (H_R + reg_lambda)).sum(1)
                 - (G_total**2 / (H_total + reg_lambda)).sum())
        gains_list.append(torch.where(valid[qi], 0.5 * gnum, torch.full_like(gnum, -1e9)))
    return torch.stack(gains_list).max(0).values.clamp(min=0.0)


try:
    _split_gain_fitness = torch.compile(_split_gain_fitness_impl, backend="aot_eager", fullgraph=False)
except Exception:
    _split_gain_fitness = _split_gain_fitness_impl


def _sparsify(w, int D_num):
    cdef int D   = w.shape[0]
    cdef int k   = max(2, <int>c_sqrt(<double>D_num))
    dev = w.device

    w = w.clone()
    if D_num < D:
        w[D_num:] = 0.0

    if D_num > k:
        _, indices    = torch.topk(w[:D_num].abs(), k)
        mask          = torch.zeros(D_num, device=dev)
        mask[indices] = 1.0
        w[:D_num]    *= mask

    norm = torch.linalg.norm(w)
    if norm > 1e-5:
        return w / norm
    fallback          = torch.zeros(D, device=dev)
    fallback[int(torch.randint(0, D_num, (1,)).item())] = 1.0
    return fallback


# ── diversity pruning (C-speed inner loop via typed memoryview) ───────────────

cdef list _diversity_prune(list pop, cnp.ndarray sim_mat_arr, int max_size):
    """O(n²) greedy redundancy removal — inner loop runs at C speed."""
    cdef double[:, :] S = sim_mat_arr.astype(np.float64)
    cdef int n = len(pop)
    cdef int i, j
    cdef bint redundant

    pruned = []
    kept   = []

    for i in range(n):
        redundant = False
        for j in kept:
            if S[i, j] > 0.90:
                redundant = True
                break
        if not redundant:
            pruned.append(pop[i])
            kept.append(i)
            if len(pruned) >= max_size:
                break
    return pruned


# ── HypForgePool ──────────────────────────────────────────────────────────────

class HypForgePool:
    def __init__(self, int D, int max_size=500, dev="cpu"):
        self.D        = D
        self.max_size = max_size
        self.dev      = torch.device(dev)
        self.pop      = []

        for j in range(D):
            w    = torch.zeros(D, device=self.dev)
            w[j] = 1.0
            h    = Hypothesis(hyp_type="linear", w=w)
            h.is_base = True
            self.pop.append(h)

    def evolve(self, X_full, G_full, H_full, sub_indices,
               int D_num, double reg_lambda=1.0, double eta_penalty=0.002):
        cdef int P = len(self.pop)
        cdef int i, K, M, ci, fi
        cdef double fit_val

        if P == 0:
            return

        dev   = X_full.device
        X_sub = X_full[sub_indices]
        G_s   = G_full[sub_indices]
        H_s   = H_full[sub_indices]

        for h in self.pop:
            if h.full_cache is None:
                h.initialize_cache(X_full)
            h.cache = h.full_cache[sub_indices]

        caches    = torch.stack([h.cache for h in self.pop])
        fitnesses = _split_gain_fitness(caches, G_s, H_s, reg_lambda)

        for i in range(P):
            fit_val = float(fitnesses[i].item())
            self.pop[i].fitness = fit_val
            self.pop[i].score   = fit_val - eta_penalty * self.pop[i].complexity()
        self.pop.sort(key=lambda h: h.score, reverse=True)

        raw_candidates = []

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
            for jj in range(min(2, self.D)):
                w_j = _sparsify(U[:, jj].to(dev), D_num)
                for ht in ("linear", "square", "abs"):
                    raw_candidates.append(Hypothesis(hyp_type=ht, w=w_j))
        except Exception:
            pass

        # Gradient ascent on top pool members
        flow = [h for h in self.pop[:30] if h.hyp_type in ("linear", "square", "abs")]
        if flow:
            W_flow = torch.stack([h.w for h in flow], dim=1)
            P_flow = X_sub @ W_flow
            for fi in range(len(flow)):
                h  = flow[fi]
                z  = h.cache
                p  = P_flow[:, fi]
                v  = G_s.T @ z
                g  = G_s @ v
                if   h.hyp_type == "linear": grad_w = X_sub.T @ (g - z)
                elif h.hyp_type == "square": grad_w = 2.0 * X_sub.T @ ((g - z) * p)
                elif h.hyp_type == "abs":    grad_w = X_sub.T @ ((g - z) * p.sign())
                else:                        continue
                raw_candidates.append(Hypothesis(hyp_type=h.hyp_type,
                                                  w=_sparsify(h.w + 0.1 * grad_w, D_num)))

        # Synergy (product) hypotheses
        non_prod = [h for h in self.pop if h.hyp_type in ("linear", "square", "abs")]
        M = min(15, len(non_prod))
        for i in range(M):
            for jj in range(i + 1, M):
                raw_candidates.append(Hypothesis(hyp_type="product",
                                                  h1=non_prod[i], h2=non_prod[jj]))

        # Blind-spot (min-eigenvalue) exploration
        vectors = [h.w for h in self.pop if h.hyp_type in ("linear", "square", "abs")]
        if vectors:
            W = torch.stack(vectors, dim=1)
            C = W @ W.T
            try:
                _, evecs = torch.linalg.eigh(C.cpu())
                w_blind  = _sparsify(evecs[:, 0].to(dev), D_num)
                if not any(
                    h.hyp_type in ("linear", "square", "abs")
                    and torch.dot(h.w, w_blind).abs().item() > 0.98
                    for h in self.pop
                ):
                    raw_candidates.append(Hypothesis(hyp_type="linear", w=w_blind))
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
            for ci in range(len(vec_cands)):
                c = vec_cands[ci]
                p = proj_c[:, ci]
                if   c.hyp_type == "linear": c.cache = p
                elif c.hyp_type == "square": c.cache = p ** 2
                elif c.hyp_type == "abs":    c.cache = p.abs()
        for c in prod_cands:
            c.cache = c.h1.cache * c.h2.cache

        all_caches  = torch.stack([c.cache for c in raw_candidates])
        all_fitness = _split_gain_fitness(all_caches, G_s, H_s, reg_lambda)
        for ci in range(len(raw_candidates)):
            c        = raw_candidates[ci]
            fit_val  = float(all_fitness[ci].item())
            c.fitness = fit_val
            c.score   = fit_val - eta_penalty * c.complexity()
            if fit_val > 1e-5:
                c.initialize_cache(X_full)
                self.pop.append(c)

        for h in self.pop:
            h.score = h.fitness + 0.05 * h.use_count - eta_penalty * h.complexity()
        self.pop.sort(key=lambda h: h.score, reverse=True)

        # Diversity pruning (typed memoryview inner loop)
        for h in self.pop:
            h.cache = h.full_cache[sub_indices]
        caches  = torch.stack([h.cache for h in self.pop])
        norms   = torch.linalg.norm(caches, dim=1, keepdim=True)
        normed  = (caches / (norms + 1e-8)).cpu().numpy().astype(np.float64)
        sim_mat = np.abs(normed @ normed.T)

        self.pop = _diversity_prune(self.pop, sim_mat, self.max_size)
        self.pop = [h for h in self.pop if h.is_base or h.rounds_since_last_use <= 15]
