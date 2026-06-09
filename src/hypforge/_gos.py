"""
_gos.py — d-GOS (Differentiable Gradient Ordering Search) Pool

Implements GOSV2.md:

  max_W  Σ_b J(w^b)  -  γ · L_repel(W)
  J(w)  = H_soft(w) - λ||w||₁

  H_soft(w) = (1/K) Σ_k LogSumExp_m ( |G_m^k(w)| / √H_m^k(w) )

  G_m^k = Σ_{j≤m} ĝ_j^k,   ĝ_m^k = Σ_i A_im · g_{ik}
  A_im  = softmax_m( -|x_i^T w - c_m|² / τ )   (Soft Binning)

  L_repel = Σ_{i<j} log( 1 / (1 - cos²(w_i,w_j) + ε) )

Update rule (Proximal Gradient + L1 → Prune):
  w̃ = w + η · ∇_w H_soft  -  η·γ·∇_w L_repel
  w  = SoftThreshold(w̃, λη) / ‖SoftThreshold(w̃, λη)‖
"""

import ctypes
import numpy as np

from ._testforge import _get_lib


def _get_gos_lib():
    lib = _get_lib()
    if not hasattr(lib, "_gos_configured"):
        lib.gos_build_tree.restype  = ctypes.c_void_p
        lib.gos_build_tree.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Z (B×N)
            ctypes.c_int,                    # N
            ctypes.c_int,                    # B
            ctypes.c_int,                    # K
            ctypes.POINTER(ctypes.c_float),  # G_full
            ctypes.POINTER(ctypes.c_float),  # H_full
            ctypes.POINTER(ctypes.c_int),    # tree_sub
            ctypes.c_int,                    # Ns_tree
            ctypes.c_int,                    # max_depth
            ctypes.c_float,                  # reg_lambda
            ctypes.POINTER(ctypes.c_float),  # out_pred
        ]
        lib.gos_predict_tree.restype  = None
        lib.gos_predict_tree.argtypes = [
            ctypes.c_void_p,                 # tree_handle
            ctypes.POINTER(ctypes.c_float),  # Z (B×N)
            ctypes.c_int,                    # N
            ctypes.c_int,                    # B
            ctypes.c_int,                    # K
            ctypes.POINTER(ctypes.c_float),  # out_pred
        ]
        lib.bfs_tree_free.restype  = None
        lib.bfs_tree_free.argtypes = [ctypes.c_void_p]
        lib.gos_evolve.restype  = None
        lib.gos_evolve.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # X  (Ns, D)
            ctypes.c_int,                    # Ns
            ctypes.c_int,                    # D
            ctypes.POINTER(ctypes.c_float),  # G  (Ns, K)
            ctypes.POINTER(ctypes.c_float),  # H  (Ns, K)
            ctypes.c_int,                    # K
            ctypes.POINTER(ctypes.c_float),  # centers (M,)
            ctypes.c_int,                    # M
            ctypes.POINTER(ctypes.c_float),  # W  (B, D) — updated in place
            ctypes.c_int,                    # B
            ctypes.c_float,                  # tau
            ctypes.c_float,                  # eta
            ctypes.c_float,                  # lam
            ctypes.c_float,                  # gamma_rep
            ctypes.c_int,                    # n_steps
        ]
        lib._gos_configured = True
    return lib


class GOSPool:
    """
    d-GOS particle pool.

    Parameters
    ----------
    D        : feature dimension
    B        : number of particles (width of hypothesis pool)
    M        : number of soft bins  (resolution of differentiable H)
    tau      : bin softmax temperature  (lower = sharper, harder to diff)
    n_steps  : gradient ascent steps per evolve() call
    eta      : step size
    lam      : L1 sparsity strength  (Prune via SoftThreshold)
    gamma    : repulsion strength    (Split/Merge via particle divergence)
    seed     : RNG seed
    """

    def __init__(self, D: int, B: int = 32, M: int = 32, tau: float = 1.0,
                 n_steps: int = 10, eta: float = 0.05,
                 lam: float = 0.01, gamma: float = 0.1, seed: int = 42):
        self.D       = D
        self.B       = B
        self.M       = M
        self.tau     = tau
        self.n_steps = n_steps
        self.eta     = eta
        self.lam     = lam
        self.gamma   = gamma

        rng = np.random.default_rng(seed)
        W = np.zeros((B, D), dtype=np.float64)
        # Axis-aligned particles first (DT baselines, GOS.md §12)
        for j in range(min(B, D)):
            W[j, j] = 1.0
        if B > D:
            R = rng.standard_normal((B - D, D))
            R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
            W[D:] = R
        self.W = W          # B × D (float64 for numerical precision)
        self._round = 0
        self._trees: list  = []   # C BFSTree pointers (freed on del)

    # ── H functional & analytic gradient ─────────────────────────────────────

    @staticmethod
    def _soft_assign(z: np.ndarray, centers: np.ndarray, tau: float) -> np.ndarray:
        """A[i,m] = softmax_m(-|z_i - c_m|² / τ)   shape N×M"""
        d = -((z[:, None] - centers[None, :]) ** 2) / tau   # N×M
        d -= d.max(axis=1, keepdims=True)
        A  = np.exp(d)
        A /= A.sum(axis=1, keepdims=True) + 1e-10
        return A

    def _h_soft_and_grad(self, X: np.ndarray, g: np.ndarray, h: np.ndarray,
                          w: np.ndarray, centers: np.ndarray):
        """
        Returns (H_soft scalar, ∇_w H_soft  shape D).

        Analytic derivation (all O(ND)):
          z  = Xw
          A  = soft_assign(z, centers, τ)          N×M
          ĝ  = Aᵀg                                 M×K
          G  = cumsum(ĝ)                            M×K
          f  = |G|/√(H+ε)                          M×K
          H_soft = (1/K) Σ_k LogSumExp_m f_mk

        Chain rule back to w:
          α  = softmax_m(f)                        M×K   [LogSumExp weights]
          φ  = backward_cumsum(α · sign(G)/√H / K) M×K   [∂H_soft/∂ĝ_m]
          ψ  = backward_cumsum(α · -½|G|/H^{3/2}/K)M×K  [∂H_soft/∂ĥ_m]
          δ  = g·φᵀ + h·ψᵀ                        N×M   [∂H_soft/∂A_im]
          μ  = A·centers                           N     [expected bin center]
          ∂z = (2/τ)·Σ_m δ_im·A_im·(c_m - μ_i)   N     [∂H_soft/∂z_i]
          ∇w = Xᵀ∂z                               D
        """
        K = g.shape[1]
        M = len(centers)

        z = X @ w                                          # N
        A = self._soft_assign(z, centers, self.tau)        # N×M

        g_hat = A.T @ g                                    # M×K
        h_hat = A.T @ h                                    # M×K
        G = np.cumsum(g_hat, axis=0)                       # M×K
        H = np.cumsum(h_hat, axis=0)                       # M×K

        sqH  = np.sqrt(np.abs(H) + 1e-8)
        f    = np.abs(G) / sqH                             # M×K

        f_max = f.max(axis=0, keepdims=True)
        ef    = np.exp(f - f_max)
        lse   = np.log(ef.sum(axis=0)) + f_max[0]         # K
        alpha = ef / (ef.sum(axis=0, keepdims=True) + 1e-10)  # M×K

        h_val = lse.mean()

        dfdG = np.sign(G) / sqH                            # M×K
        dfdH = -0.5 * np.abs(G) / (np.abs(H) + 1e-8) ** 1.5

        dHdG = alpha * dfdG / K
        dHdH = alpha * dfdH / K

        # backward cumsum: φ_m = Σ_{m'≥m} dHdG_{m'}
        phi = np.flip(np.cumsum(np.flip(dHdG, 0), 0), 0)  # M×K
        psi = np.flip(np.cumsum(np.flip(dHdH, 0), 0), 0)

        delta  = g @ phi.T + h @ psi.T                    # N×M
        mu_c   = A @ centers                               # N
        c_mu   = centers[None, :] - mu_c[:, None]         # N×M
        grad_z = (2.0 / self.tau) * (delta * A * c_mu).sum(axis=1)  # N

        return h_val, X.T @ grad_z                         # scalar, D

    def _repulsion_and_grad(self, W: np.ndarray):
        """
        L_repel = Σ_{i<j} log(1/(1 - cos²(w_i,w_j) + ε))
        Gradient: ∂L_repel/∂w_b = Σ_{b'≠b} 2cos/(1-cos²+ε) · (ŵ_{b'} - cos·ŵ_b) / ‖w_b‖
        """
        norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
        Wn    = W / norms                                  # B×D (unit vectors)
        C     = Wn @ Wn.T                                  # B×B cosine matrix
        C2    = C ** 2
        np.fill_diagonal(C2, 0.0); np.fill_diagonal(C, 0.0)
        eps   = 1e-6
        denom = 1.0 - C2 + eps
        val   = np.log(1.0 / denom).sum() / 2.0

        coeff = 2.0 * C / denom                            # B×B (symmetric, zero diag)
        wsum  = coeff @ Wn                                 # B×D: Σ_{b'} coeff·ŵ_{b'}
        sw    = (coeff * C).sum(axis=1, keepdims=True)     # B×1: Σ_{b'} coeff·cos
        grad  = (wsum - sw * Wn) / norms                   # B×D

        return val, grad

    # ── Public interface ──────────────────────────────────────────────────────

    # ── GPU/accelerated evolve ────────────────────────────────────────────────

    @staticmethod
    def _pick_device():
        """Pick best available device for torch ops."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        except ImportError:
            return None

    def _evolve_torch(self, Xs, Gs, Hs, W_np, centers_np):
        """
        Full proximal gradient ascent loop on GPU/MPS using PyTorch autograd.

        All B particles updated in one batched forward+backward per step.
        Returns updated W as numpy (B, D).
        """
        import torch
        dev = self._pick_device()

        # Convert to torch (float32 on GPU — fast; autograd handles it)
        dtype  = torch.float32
        Xt  = torch.tensor(Xs,       dtype=dtype, device=dev)   # (Ns, D)
        Gt  = torch.tensor(Gs,       dtype=dtype, device=dev)   # (Ns, K)
        Ht  = torch.tensor(Hs,       dtype=dtype, device=dev)   # (Ns, K)
        ct  = torch.tensor(centers_np, dtype=dtype, device=dev) # (M,)
        W_t = torch.tensor(W_np,     dtype=dtype, device=dev)   # (B, D)

        tau   = float(self.tau)
        eta   = float(self.eta)
        lam   = float(self.lam)
        gamma = float(self.gamma)
        K     = Gs.shape[1]

        for _ in range(self.n_steps):
            W_t.requires_grad_(True)

            # Z[n, b] = Xs[n] · W[b]
            Z = Xt @ W_t.T                                       # (Ns, B)

            # Soft assignment A[n, b, m]
            d = -((Z.unsqueeze(2) - ct[None, None, :]) ** 2) / tau   # (Ns, B, M)
            A = torch.softmax(d, dim=2)                          # (Ns, B, M)

            # Bin-level aggregation: (B, M, Ns) @ (Ns, K) → (B, M, K)
            g_hat = A.permute(1, 2, 0) @ Gt                     # (B, M, K)
            h_hat = A.permute(1, 2, 0) @ Ht

            G_cum = torch.cumsum(g_hat, dim=1)                   # (B, M, K)
            H_cum = torch.cumsum(h_hat, dim=1)

            sqH = (H_cum.abs() + 1e-8).sqrt()
            f   = G_cum.abs() / sqH                              # (B, M, K)

            # H_soft = mean_k LogSumExp_m f[b, m, k], summed over B
            H_soft = torch.logsumexp(f, dim=1).mean(dim=1).sum()
            H_soft.backward()

            grads_H = W_t.grad.detach()                          # (B, D)

            with torch.no_grad():
                # Repulsion on CPU (B×B is tiny)
                W_np_cur = W_t.detach().cpu().numpy().astype(np.float64)
                _, grads_R_np = self._repulsion_and_grad(W_np_cur)
                grads_R = torch.tensor(grads_R_np, dtype=dtype, device=dev)

                W_new = W_t.detach() + eta * (grads_H - gamma * grads_R)

                # L1 SoftThreshold
                thr   = lam * eta
                W_new = W_new.sign() * (W_new.abs() - thr).clamp(min=0.0)

                # Project to unit sphere
                norms = W_new.norm(dim=1, keepdim=True)
                alive = norms.squeeze(1) > 1e-5
                W_new[alive]  = W_new[alive] / norms[alive]
                W_new[~alive] = W_t.detach()[~alive]

                W_t = W_new                                      # detached, ready for next iter

        return W_t.cpu().numpy().astype(np.float64)

    def _evolve_numpy(self, Xs, Gs, Hs, W, centers):
        """Vectorized numpy fallback (no torch)."""
        for _ in range(self.n_steps):
            Ns, K = Gs.shape
            Z  = Xs @ W.T                                        # (Ns, B)
            d  = -((Z[:, :, None] - centers[None, None, :]) ** 2) / self.tau
            d -= d.max(axis=2, keepdims=True)
            A  = np.exp(d); A /= A.sum(axis=2, keepdims=True) + 1e-10

            g_hat = A.transpose(1, 2, 0) @ Gs                   # (B, M, Ns)@(Ns,K)→(B,M,K)
            h_hat = A.transpose(1, 2, 0) @ Hs

            G_cum = np.cumsum(g_hat, axis=1)
            H_cum = np.cumsum(h_hat, axis=1)
            sqH   = np.sqrt(np.abs(H_cum) + 1e-8)
            f     = np.abs(G_cum) / sqH

            f_max = f.max(axis=1, keepdims=True)
            ef    = np.exp(f - f_max)
            alpha = ef / (ef.sum(axis=1, keepdims=True) + 1e-10)

            dHdG = alpha * np.sign(G_cum) / sqH / K
            dHdH = alpha * (-0.5 * np.abs(G_cum) / (np.abs(H_cum) + 1e-8) ** 1.5) / K

            phi  = np.flip(np.cumsum(np.flip(dHdG, 1), 1), 1)
            psi  = np.flip(np.cumsum(np.flip(dHdH, 1), 1), 1)

            # (B, M, Ns) @ (Ns, K) already done; now need (Ns,K)@(K,B*M)...
            # simpler: delta[n,b,m] = sum_k Gs[n,k]*phi[b,m,k] + Hs[n,k]*psi[b,m,k]
            delta  = (Gs @ phi.reshape(self.B, self.M, K).transpose(0, 2, 1)
                                .reshape(self.B * self.M, K).T
                     ).reshape(Ns, self.B, self.M)  # noqa — cleaner below
            # Recompute cleanly:
            delta  = np.einsum('nk,bmk->nbm', Gs, phi) + np.einsum('nk,bmk->nbm', Hs, psi)
            mu_c   = np.einsum('nbm,m->nb', A, centers)
            c_mu   = centers[None, None, :] - mu_c[:, :, None]
            grad_z = (2.0 / self.tau) * (delta * A * c_mu).sum(axis=2)  # (Ns, B)
            grads_H = (Xs.T @ grad_z).T                        # (B, D)

            _, grads_R = self._repulsion_and_grad(W)
            W_new = W + self.eta * (grads_H - self.gamma * grads_R)
            thr   = self.lam * self.eta
            W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - thr, 0.0)
            norms = np.linalg.norm(W_new, axis=1, keepdims=True)
            alive = norms.ravel() > 1e-5
            W_new[alive]  /= norms[alive]
            W_new[~alive]  = W[~alive]
            W = W_new
        return W

    def evolve(self, X: np.ndarray, G: np.ndarray, H: np.ndarray,
               sub: np.ndarray | None = None, round_num: int = 0) -> None:
        """
        One evolve step via C++ OpenMP backend (gos_evolve in testforge.cpp).
        Subsample capped at 4096; W updated in place via ctypes.
        """
        MAX_EVOLVE = 4096
        if sub is not None:
            es = sub[:MAX_EVOLVE] if len(sub) > MAX_EVOLVE else sub
        else:
            rng = np.random.default_rng(round_num)
            es  = rng.choice(len(X), min(MAX_EVOLVE, len(X)), replace=False)

        Xs = np.ascontiguousarray(X[es], dtype=np.float32)
        Gs_raw = G[es] if G.ndim > 1 else G[es, None]
        Hs_raw = H[es] if H.ndim > 1 else H[es, None]
        Gs = np.ascontiguousarray(Gs_raw, dtype=np.float32)
        Hs = np.ascontiguousarray(Hs_raw, dtype=np.float32)

        Ns, D = Xs.shape
        K     = Gs.shape[1]

        # Bin centers from current projections
        Z_s   = Xs @ self.W.T.astype(np.float32)
        z_min, z_max = float(Z_s.min()), float(Z_s.max())
        centers = np.linspace(z_min, z_max, self.M, dtype=np.float32)

        # W must be float32, C-contiguous for in-place C++ update
        W = np.ascontiguousarray(self.W, dtype=np.float32)

        lib = _get_gos_lib()
        lib.gos_evolve(
            Xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(Ns), ctypes.c_int(D),
            Gs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            Hs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(K),
            centers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(self.M),
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(self.B),
            ctypes.c_float(self.tau),
            ctypes.c_float(self.eta),
            ctypes.c_float(self.lam),
            ctypes.c_float(self.gamma),
            ctypes.c_int(self.n_steps),
        )
        self.W      = W.astype(np.float64)  # back to float64 for snapshot
        self._round = round_num

    def build_tree(self, X: np.ndarray, G: np.ndarray, H: np.ndarray,
                   tree_sub: np.ndarray, max_depth: int,
                   reg_lambda: float, out_pred: np.ndarray) -> ctypes.c_void_p:
        """Build BFSTree from current particles. Returns opaque C pointer."""
        lib = _get_gos_lib()
        N, K = len(X), (G.shape[1] if G.ndim > 1 else 1)

        # Z (B × N, row-major): row b = particle b's projection over all N samples
        Z = (X @ self.W.T).T.astype(np.float32)         # B × N, C-contiguous
        Z = np.ascontiguousarray(Z)

        G_c = np.ascontiguousarray(G.reshape(N, K).astype(np.float32))
        H_c = np.ascontiguousarray(H.reshape(N, K).astype(np.float32))
        sub_c = np.ascontiguousarray(tree_sub.astype(np.int32))
        out_c = np.ascontiguousarray(out_pred.reshape(N, K).astype(np.float32))

        tree_ptr = lib.gos_build_tree(
            Z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(N), ctypes.c_int(self.B), ctypes.c_int(K),
            G_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            H_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sub_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(len(sub_c)),
            ctypes.c_int(max_depth), ctypes.c_float(reg_lambda),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        out_pred[:] = out_c.reshape(out_pred.shape)
        return tree_ptr

    def predict_tree(self, tree_ptr: ctypes.c_void_p,
                     X: np.ndarray, out_pred: np.ndarray) -> None:
        """Apply stored tree to new X using snapshot W."""
        lib = _get_gos_lib()
        N = len(X)
        K = out_pred.shape[1] if out_pred.ndim > 1 else 1
        Z   = np.ascontiguousarray((X @ self.W.T).T.astype(np.float32))  # B×N
        out = np.zeros((N, K), dtype=np.float32)
        lib.gos_predict_tree(
            ctypes.c_void_p(tree_ptr),
            Z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(N), ctypes.c_int(self.B), ctypes.c_int(K),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        out_pred[:] = out.reshape(out_pred.shape)

    def snapshot(self) -> np.ndarray:
        """Return copy of current W (B × D) for inference."""
        return self.W.copy()

    def get_size(self) -> int:
        return self.B

    def __del__(self):
        # Tree handles owned by GOSClassifier; nothing to free here.
        pass
