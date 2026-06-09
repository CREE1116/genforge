from __future__ import annotations

import ctypes
import numpy as np
from ._tree import _get_lib as _get_tree_lib

# ── Standalone SALOT C bindings (pool-free) ───────────────────────────────────

_salot_configured = False


def _get_salot_lib():
    global _salot_configured
    lib = _get_tree_lib()
    if _salot_configured:
        return lib

    _pf = ctypes.POINTER(ctypes.c_float)
    _pi = ctypes.POINTER(ctypes.c_int)

    lib.salot_build.argtypes = [
        _pf,                   # X             [N, D]
        ctypes.c_int,          # N
        ctypes.c_int,          # D
        ctypes.c_int,          # D_num
        _pf,                   # G             [N, K]
        _pf,                   # H             [N, K]
        ctypes.c_int,          # K
        _pi,                   # sub           [Ns]
        ctypes.c_int,          # Ns
        ctypes.c_int,          # max_depth
        ctypes.c_float,        # reg_lambda
        ctypes.c_int,          # n_wls_max
        ctypes.c_int,          # d_sub_max
        ctypes.c_float,        # energy_frac
        ctypes.c_float,        # gbaor_alpha
        ctypes.c_uint,         # seed
        _pf,                   # out_pred      [N, K]  (may be NULL)
    ]
    lib.salot_build.restype = ctypes.c_void_p

    lib.salot_predict.argtypes = [
        ctypes.c_void_p,       # tree_handle
        _pf,                   # X     [N, D]
        ctypes.c_int,          # N
        ctypes.c_int,          # K
        _pf,                   # out_pred [N, K]
    ]
    lib.salot_predict.restype = None

    lib.salot_tree_free.argtypes = [ctypes.c_void_p]
    lib.salot_tree_free.restype = None

    _salot_configured = True
    return lib


class SALOTTree:
    """Single pool-free SALOT tree.

    Per-node block WLS with instance + feature subsampling and
    zero-cross-term feature bundling.  Fit once, predict multiple times.

    Parameters
    ----------
    max_depth      : int
    reg_lambda     : float   base L2 regularisation (λ₀)
    prune_strength : float   energy pruning: 0 = none, 1 = aggressive
    n_wls_max      : int     max instances used for WLS per node (default 512)
    d_sub_max      : int     max features for WLS block; 0 → ceil(sqrt(D))
    subsample      : float   fraction of training samples used
    gbaor_alpha    : float   Gershgorin bound ratio target; 0 = disabled
    random_state   : int | None
    """

    def __init__(
        self,
        max_depth:      int   = 4,
        reg_lambda:     float = 1.0,
        prune_strength: float = 0.1,
        n_wls_max:      int   = 512,
        d_sub_max:      int   = 32,
        subsample:      float = 1.0,
        gbaor_alpha:    float = 0.05,
        random_state:   int | None = None,
    ):
        self.max_depth      = max_depth
        self.reg_lambda     = reg_lambda
        self.prune_strength = prune_strength
        self.n_wls_max      = n_wls_max
        self.d_sub_max      = d_sub_max
        self.subsample      = subsample
        self.gbaor_alpha    = gbaor_alpha
        self.random_state   = random_state
        self._tree_handle   = None
        self._N             = None
        self._K             = None

    def fit_predict(
        self,
        X:      np.ndarray,
        G:      np.ndarray,
        H:      np.ndarray,
        D_num:  int | None = None,
        subset: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build SALOT tree and return leaf predictions for all N samples.

        Parameters
        ----------
        X      : float32 (N, D)
        G      : float32 (N, K)
        H      : float32 (N, K)
        D_num  : number of numerical features (default: D)
        subset : int32 (Ns,) indices used for tree building (default: all)
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        G = np.ascontiguousarray(G, dtype=np.float32)
        H = np.ascontiguousarray(H, dtype=np.float32)
        N, D = X.shape
        K    = G.shape[1]
        self._N, self._K = N, K

        if D_num is None:
            D_num = D

        if subset is None:
            if self.subsample < 1.0:
                rng    = np.random.default_rng(self.random_state)
                ns     = max(1, int(N * self.subsample))
                subset = rng.choice(N, ns, replace=False).astype(np.int32)
            else:
                subset = np.arange(N, dtype=np.int32)
        subset = np.ascontiguousarray(subset, dtype=np.int32)

        energy_frac = max(0.01, 1.0 - float(self.prune_strength))
        seed        = self.random_state if self.random_state is not None else 0

        lib  = _get_salot_lib()
        _pf  = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _pi  = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        out_pred = np.zeros((N, K), dtype=np.float32)

        self._tree_handle = lib.salot_build(
            _pf(X), N, D, D_num,
            _pf(G), _pf(H), K,
            _pi(subset), len(subset),
            self.max_depth,
            ctypes.c_float(self.reg_lambda),
            ctypes.c_int(self.n_wls_max),
            ctypes.c_int(self.d_sub_max),
            ctypes.c_float(energy_frac),
            ctypes.c_float(self.gbaor_alpha),
            ctypes.c_uint(seed),
            _pf(out_pred),
        )
        return out_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Route X through the fitted tree; returns float32 (N, K)."""
        if self._tree_handle is None:
            raise RuntimeError("Call fit_predict() before predict().")
        X   = np.ascontiguousarray(X, dtype=np.float32)
        N   = X.shape[0]
        out = np.zeros((N, self._K), dtype=np.float32)
        lib = _get_salot_lib()
        lib.salot_predict(
            self._tree_handle,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, self._K,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def __del__(self):
        if self._tree_handle is not None:
            try:
                _get_salot_lib().salot_tree_free(self._tree_handle)
            except Exception:
                pass
            self._tree_handle = None
