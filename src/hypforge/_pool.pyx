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
from libc.math cimport sqrt as c_sqrt, log as c_log

# UCB exploration constants
DEF C_UCB   = 1.0   # controls width of confidence interval
DEF C_NOVEL = 0.5   # novelty bonus for newly-added hypotheses (decays as 1/sqrt(n_obs))


# ── Hypothesis ────────────────────────────────────────────────────────────────

class Hypothesis:
    """Oblique split candidate: a projection direction + type."""

    def __init__(self, hyp_type="linear", w=None, h1=None, h2=None):
        self.hyp_type = hyp_type
        self.w        = w
        self.h1       = h1
        self.h2       = h2

        # ── UCB bandit statistics (Welford online mean/variance) ────────────
        self.n_obs      = 0      # number of fitness observations
        self.mu_fitness = 0.0    # running mean
        self.M2_fitness = 0.0    # running sum of squared deviations (Welford)

        # ── usage stats ────────────────────────────────────────────────────
        self.use_count             = 0
        self.rounds_since_last_use = 0

        # ── meta ───────────────────────────────────────────────────────────
        self.fitness  = 0.0      # last observed fitness (for display/compat)
        self.score    = 0.0      # UCB score
        self.is_base  = False

        # ── projection caches ─────────────────────────────────────────────
        self.cache          = None
        self.full_cache     = None
        self.full_cache_cpu = None
        self.thresholds     = None
        self.thresholds_cpu = None
        self.val_cache_cpu  = None

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

    def ucb_score(self, double eta_penalty):
        """
        UCB1-style score with novelty bonus.

        score = μ
              + C_UCB   × σ / √n_obs    ← confidence interval half-width
              + C_NOVEL / √n_obs         ← novelty bonus (fades with observations)
              − η × complexity

        New hypotheses (n_obs=1, σ=0) start at μ + C_NOVEL (small boost).
        As evidence accumulates the score converges to μ − η×complexity.
        """
        cdef double sigma, n, bonus
        if self.n_obs == 0:
            return 0.0
        n     = float(self.n_obs)
        sigma = c_sqrt(self.M2_fitness / n) if self.n_obs > 1 else 0.0
        bonus = (C_UCB * sigma + C_NOVEL) / c_sqrt(n)
        return self.mu_fitness + bonus - eta_penalty * self.complexity()

    def observe_fitness(self, double fit_val):
        """Welford online update for mean and variance."""
        cdef double delta, delta2
        self.fitness  = fit_val
        self.n_obs   += 1
        delta         = fit_val - self.mu_fitness
        self.mu_fitness += delta / self.n_obs
        delta2        = fit_val - self.mu_fitness
        self.M2_fitness += delta * delta2

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
        self.val_cache_cpu  = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_gain_fitness_impl(cs, G, H, double reg_lambda):
    cdef int P = cs.shape[0]
    cdef int ns = cs.shape[1]
    cdef int qi

    device  = cs.device
    G_total = G.sum(dim=0)
    H_total = H.sum(dim=0)

    q      = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], dtype=cs.dtype, device=device)
    theta  = torch.quantile(cs, q, dim=1)
    cs_exp = cs.unsqueeze(0)
    lm_all = cs_exp < theta.unsqueeze(2)
    nl     = lm_all.sum(2)
    valid  = (nl >= 10) & (ns - nl >= 10)

    gains_list = []
    for qi in range(9):
        lm_q = lm_all[qi].to(dtype=G.dtype)
        G_L  = lm_q @ G
        H_L  = lm_q @ H
        G_R  = G_total - G_L
        H_R  = H_total - H_L
        gnum = ((G_L**2 / (H_L + reg_lambda)).sum(1)
                + (G_R**2 / (H_R + reg_lambda)).sum(1)
                - (G_total**2 / (H_total + reg_lambda)).sum())
        gains_list.append(torch.where(valid[qi], 0.5 * gnum, torch.full_like(gnum, -1e9)))
    return torch.stack(gains_list).max(0).values.clamp(min=0.0)


try:
    _split_gain_fitness = torch.compile(_split_gain_fitness_impl, backend="aot_eager", fullgraph=False)
except Exception:
    _split_gain_fitness = _split_gain_fitness_impl


def _sparsify(w, int D_num):
    cdef int D = w.shape[0]
    cdef int k = max(2, <int>c_sqrt(<double>D_num))
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


# ── diversity pruning ─────────────────────────────────────────────────────────

cdef list _diversity_prune(list pop, cnp.ndarray sim_mat_arr, int max_size):
    """
    Greedy diversity-aware selection.

    Same-type hypotheses: cosine > 0.90 → redundant
    Different-type:       cosine > 0.98 → redundant
    (linear vs square on the same direction are genuinely distinct: x vs x²)
    """
    cdef double[:, :] S = sim_mat_arr.astype(np.float64)
    cdef int n = len(pop)
    cdef int i, j
    cdef bint redundant
    cdef double threshold

    pruned = []
    kept   = []

    for i in range(n):
        redundant = False
        for j in kept:
            threshold = 0.90 if pop[i].hyp_type == pop[j].hyp_type else 0.98
            if S[i, j] > threshold:
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

        # ── refresh caches & observe fitness ─────────────────────────────
        for h in self.pop:
            if h.full_cache is None:
                h.initialize_cache(X_full)
        caches = torch.stack([h.full_cache for h in self.pop])[:, sub_indices]
        for i, h in enumerate(self.pop):
            h.cache = caches[i]
        fitnesses = _split_gain_fitness(caches, G_s, H_s, reg_lambda)

        for i in range(P):
            fit_val = float(fitnesses[i].item())
            self.pop[i].observe_fitness(fit_val)
            self.pop[i].score = self.pop[i].ucb_score(eta_penalty)

        self.pop.sort(key=lambda h: h.score, reverse=True)

        # ── generate candidates ───────────────────────────────────────────
        raw_candidates = []

        # ── gradient flow: one direction per class ───────────────────────
        V = X_sub.T @ G_s
        K = G_s.shape[1]
        for k in range(K):
            w_k = _sparsify(V[:, k], D_num)
            for ht in ("linear", "square", "abs"):
                raw_candidates.append(Hypothesis(hyp_type=ht, w=w_k))

        # ── gradient ascent: refine top pool members ─────────────────────
        flow = [h for h in self.pop[:20] if h.hyp_type in ("linear", "square", "abs")]
        if flow:
            # individual refinement
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


        if not raw_candidates:
            return

        # ── score candidates ──────────────────────────────────────────────
        W_c    = torch.stack([c.w for c in raw_candidates], dim=1)
        proj_c = X_sub @ W_c
        for ci in range(len(raw_candidates)):
            c = raw_candidates[ci]
            p = proj_c[:, ci]
            if   c.hyp_type == "linear": c.cache = p
            elif c.hyp_type == "square": c.cache = p ** 2
            elif c.hyp_type == "abs":    c.cache = p.abs()

        all_fitness = _split_gain_fitness(
            torch.stack([c.cache for c in raw_candidates]), G_s, H_s, reg_lambda
        )

        # ── admit & batch-initialize caches ──────────────────────────────
        admitted = []
        for ci in range(len(raw_candidates)):
            fit_val = float(all_fitness[ci].item())
            if fit_val > 1e-5:
                c = raw_candidates[ci]
                c.observe_fitness(fit_val)
                c.score = c.ucb_score(eta_penalty)
                admitted.append(c)

        if admitted:
            # One batched matmul replaces M serial X_full @ c.w calls
            W_new  = torch.stack([c.w for c in admitted], dim=1)   # D × M
            projs  = X_full @ W_new                                 # N × M
            N_full = X_full.shape[0]
            q_t    = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                   dtype=projs.dtype, device=dev)
            s_idx  = torch.randperm(N_full, device=dev)[:10000] if N_full > 10000 else None
            for i, c in enumerate(admitted):
                p = projs[:, i]
                if   c.hyp_type == "linear": c.full_cache = p
                elif c.hyp_type == "square": c.full_cache = p ** 2
                elif c.hyp_type == "abs":    c.full_cache = p.abs()
                samp             = c.full_cache[s_idx] if s_idx is not None else c.full_cache
                c.thresholds     = torch.quantile(samp, q_t)
                c.full_cache_cpu = c.full_cache.cpu().numpy()
                c.thresholds_cpu = c.thresholds.cpu().numpy()
            self.pop.extend(admitted)

        # ── re-score & sort ───────────────────────────────────────────────
        for h in self.pop:
            h.score = h.ucb_score(eta_penalty)
        self.pop.sort(key=lambda h: h.score, reverse=True)

        # ── diversity pruning (type-aware cosine threshold) ───────────────
        # sub-caches are stale for newly admitted hypotheses — refresh all at once
        full_stack = torch.stack([h.full_cache for h in self.pop])
        sub_stack  = full_stack[:, sub_indices]
        for i, h in enumerate(self.pop):
            h.cache = sub_stack[i]
        caches  = sub_stack
        norms   = torch.linalg.norm(caches, dim=1, keepdim=True)
        normed  = (caches / (norms + 1e-8)).cpu().numpy().astype(np.float64)
        sim_mat = np.abs(normed @ normed.T)

        self.pop = _diversity_prune(self.pop, sim_mat, self.max_size)

        # ── survival: base hypotheses are immortal, others survive by UCB score
        # No hard staleness threshold — low-UCB hypotheses are naturally displaced
        # by better candidates through the diversity prune cap above.
        # Only remove hypotheses that have accumulated enough evidence (n_obs >= 3)
        # and whose UCB upper-bound is near zero (provably useless).
        self.pop = [
            h for h in self.pop
            if h.is_base
            or h.n_obs < 3                        # too new to judge
            or h.mu_fitness + h.score > 1e-6      # UCB upper bound still meaningful
        ]