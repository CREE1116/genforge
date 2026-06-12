from __future__ import annotations

import ctypes
import os
import platform
import numpy as np

# ── OQBoost C bindings ───────────────────────────────────────────────────

_lib = None
_EXT_DIR = os.path.join(os.path.dirname(__file__), "_ext")

_pf = ctypes.POINTER(ctypes.c_float)
_pi = ctypes.POINTER(ctypes.c_int)
_pu8 = ctypes.POINTER(ctypes.c_uint8)


def _lib_filename() -> str:
    system = platform.system()
    if system == "Darwin":
        return "liboqboost.dylib"
    elif system == "Windows":
        return "oqboost.dll"
    return "liboqboost.so"


def _get_oqboost_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib_p = os.path.join(_EXT_DIR, _lib_filename())
    if not os.path.exists(lib_p):
        raise FileNotFoundError(
            f"[oqboost] Compiled C++ binary not found at {lib_p}.\n"
            "Please build/install the package first (e.g., using `pip install .` "
            "or `uv pip install -e .`)."
        )

    lib = ctypes.CDLL(lib_p)

    # ── tree structure (de)serialization ─────────────────────────────────
    lib.oqtree_get_K.argtypes = [ctypes.c_void_p]
    lib.oqtree_get_K.restype = ctypes.c_int
    lib.oqtree_get_max_depth.argtypes = [ctypes.c_void_p]
    lib.oqtree_get_max_depth.restype = ctypes.c_int
    lib.oqtree_get_total_nodes.argtypes = [ctypes.c_void_p]
    lib.oqtree_get_total_nodes.restype = ctypes.c_int
    lib.oqtree_get_D.argtypes = [ctypes.c_void_p]
    lib.oqtree_get_D.restype = ctypes.c_int

    lib.oqtree_export.argtypes = [ctypes.c_void_p, _pi, _pf, _pf, _pu8]
    lib.oqtree_export.restype = None

    lib.oqtree_from_arrays.argtypes = [
        _pi, _pf, _pf, _pu8,
        ctypes.c_int,  # total_nodes
        ctypes.c_int,  # K
        ctypes.c_int,  # max_depth
    ]
    lib.oqtree_from_arrays.restype = ctypes.c_void_p

    lib.oqtree_get_split_weights.argtypes = [ctypes.c_void_p, _pf]
    lib.oqtree_get_split_weights.restype = None
    lib.oqtree_set_split_weights.argtypes = [ctypes.c_void_p, ctypes.c_int, _pf]
    lib.oqtree_set_split_weights.restype = None

    lib.gf_ctx_create.argtypes = [
        _pf,           # X     [N, D]   (copied; need not stay alive)
        ctypes.c_int,  # N
        ctypes.c_int,  # D
        ctypes.c_int,  # D_num
        _pi,           # sub   [Ns]     (stats subsample)
        ctypes.c_int,  # Ns
    ]
    lib.gf_ctx_create.restype = ctypes.c_void_p

    lib.gf_ctx_free.argtypes = [ctypes.c_void_p]
    lib.gf_ctx_free.restype = None

    lib.gf_build.argtypes = [
        ctypes.c_void_p,  # ctx
        _pf,              # X (unused in v9; context copy is authoritative)
        _pf,              # G        [N, K]
        _pf,              # H        [N, K]
        ctypes.c_int,     # K
        _pi,              # sub      [Ns]
        ctypes.c_int,     # Ns
        ctypes.c_int,     # max_depth
        ctypes.c_float,   # reg_lambda
        ctypes.c_float,   # inherited_rp_ratio
        ctypes.c_float,   # mutation_rate
        ctypes.c_float,   # mutation_strength
        ctypes.c_int,     # seed
        ctypes.c_int,     # pobs (1 = per-node pobs carve, 0 = all A/B/C)
        _pf,              # out_pred [N, K]  (may be NULL)
    ]

    lib.gf_build.restype = ctypes.c_void_p

    lib.gf_predict.argtypes = [
        ctypes.c_void_p,  # tree_handle
        _pf,              # X     [N, D]  raw (NaN / cat values allowed)
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        _pf,              # out_pred [N, K]
    ]
    lib.gf_predict.restype = None

    lib.gf_predict_ensemble.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # tree handles [n_trees]
        ctypes.c_int,                     # n_trees
        _pf,                              # X        [N, D] raw
        ctypes.c_int,                     # N
        ctypes.c_int,                     # K
        ctypes.c_float,                   # lr (leaf value scale)
        _pf,                              # out_pred [N, K] pre-filled F_init
    ]
    lib.gf_predict_ensemble.restype = None

    lib.gf_tree_free.argtypes = [ctypes.c_void_p]
    lib.gf_tree_free.restype = None

    lib.gf_tree_meta_sizes.argtypes = [ctypes.c_void_p, _pi]
    lib.gf_tree_meta_sizes.restype = None

    lib.gf_tree_export_meta.argtypes = [ctypes.c_void_p, _pf, _pi, _pi, _pf]
    lib.gf_tree_export_meta.restype = None

    lib.gf_tree_import_meta.argtypes = [
        ctypes.c_void_p, ctypes.c_int, _pf, ctypes.c_int,
        _pi, ctypes.c_int, _pi, _pf,
    ]
    lib.gf_tree_import_meta.restype = None

    lib.gf_update_gradients.argtypes = [
        _pf,           # F
        _pf,           # oh
        ctypes.c_int,  # N
        ctypes.c_int,  # K
        _pf,           # G
        _pf,           # H
    ]
    lib.gf_update_gradients.restype = None

    _lib = lib
    return lib


def _fptr(a: np.ndarray):
    return a.ctypes.data_as(_pf)


def _iptr(a: np.ndarray):
    return a.ctypes.data_as(_pi)


def update_gradients(F: np.ndarray, oh: np.ndarray, G: np.ndarray, H: np.ndarray) -> None:
    N, K = F.shape
    lib = _get_oqboost_lib()
    lib.gf_update_gradients(
        _fptr(F),
        _fptr(oh),
        N,
        K,
        _fptr(G),
        _fptr(H),
    )


def predict_ensemble(trees: list, X: np.ndarray, K: int, lr: float,
                     F_init: np.ndarray) -> np.ndarray:
    """Accumulate lr-scaled predictions of fitted trees in one C call.

    The numeric NaN transform is computed once and shared across trees;
    only categorical re-encoding is per-tree.  ~n_trees× fewer FFI
    crossings and transform passes than looping ``tree.predict``.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    N = X.shape[0]
    out = np.tile(np.asarray(F_init, dtype=np.float32), (N, 1))
    handles = [t._tree_handle for t in trees if t._tree_handle is not None]
    if not handles:
        return out
    arr = (ctypes.c_void_p * len(handles))(*handles)
    lib = _get_oqboost_lib()
    lib.gf_predict_ensemble(arr, len(handles), _fptr(X), N, K,
                            ctypes.c_float(lr), _fptr(out))
    return out


class OQBoostTree:
    """Single pool-free OQBoost tree.

    Deterministic two-hyperparameter oblique booster round: global 256-bin
    histogram axis tournament + per-node CD oblique directions, with native
    NaN (mean imputation) and categorical (per-round gradient-rank target
    encoding) handling.

    Parameters
    ----------
    max_depth  : int     leaf budget 2^max_depth, best-first allocation
    reg_lambda : float   L2 regularisation; also scales leaf path smoothing
    subsample  : float   fraction of training samples used (fit_predict only)
    random_state : int | None   used only for the subsample draw
    """

    def __init__(
        self,
        max_depth:    int   = 4,
        reg_lambda:   float = 1.0,
        subsample:    float = 1.0,
        random_state: int | None = None,
    ):
        self.max_depth    = max_depth
        self.reg_lambda   = reg_lambda
        self.subsample    = subsample
        self.random_state = random_state
        self._tree_handle = None
        self._K           = None

    @classmethod
    def _from_handle(cls, handle, K: int, max_depth: int, reg_lambda: float):
        t = cls(max_depth=max_depth, reg_lambda=reg_lambda)
        t._tree_handle = handle
        t._K = K
        return t

    def fit_predict(
        self,
        X:      np.ndarray,
        G:      np.ndarray,
        H:      np.ndarray,
        D_num:  int | None = None,
        subset: np.ndarray | None = None,
        inherited_rp_ratio: float = 1.0,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5,
        pobs: bool = True,
    ) -> np.ndarray:
        """One-shot build (creates and frees a binning context internally).

        For boosting loops use :class:`OQBoostContext` instead — it bins X once
        and reuses the codes across every round.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        G = np.ascontiguousarray(G, dtype=np.float32)
        H = np.ascontiguousarray(H, dtype=np.float32)
        N, _ = X.shape
        K = G.shape[1]

        if subset is None:
            if self.subsample < 1.0:
                rng    = np.random.default_rng(self.random_state)
                ns     = max(1, int(N * self.subsample))
                subset = rng.choice(N, ns, replace=False).astype(np.int32)
            else:
                subset = np.arange(N, dtype=np.int32)

        ctx = OQBoostContext(X, D_num=D_num)
        try:
            tree, out_pred = ctx.build(
                G, H, subset, self.max_depth, self.reg_lambda,
                inherited_rp_ratio=inherited_rp_ratio,
                mutation_rate=mutation_rate,
                mutation_strength=mutation_strength,
                pobs=pobs,
            )
        finally:
            ctx.close()
        self._tree_handle = tree._tree_handle
        tree._tree_handle = None  # ownership transferred to self
        self._K = K
        return out_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Route raw X (NaN / categorical values allowed) → float32 (N, K)."""
        if self._tree_handle is None:
            raise RuntimeError("Tree is not fitted.")
        X   = np.ascontiguousarray(X, dtype=np.float32)
        N   = X.shape[0]
        out = np.zeros((N, self._K), dtype=np.float32)
        lib = _get_oqboost_lib()
        lib.gf_predict(self._tree_handle, _fptr(X), N, self._K, _fptr(out))
        return out

    # ── pickling: structure via oqtree arrays + v9 meta ──────────────────────

    def __getstate__(self):
        base = {
            "max_depth":    self.max_depth,
            "reg_lambda":   self.reg_lambda,
            "subsample":    self.subsample,
            "random_state": self.random_state,
            "K":            self._K,
            "handle":       None,
        }
        if self._tree_handle is None:
            return base
        lib     = _get_oqboost_lib()
        h       = self._tree_handle
        n_nodes = lib.oqtree_get_total_nodes(h)
        K       = self._K
        D       = lib.oqtree_get_D(h)

        hyp_idx   = np.empty(n_nodes,     dtype=np.int32)
        threshold = np.empty(n_nodes,     dtype=np.float32)
        leaf_vals = np.empty(n_nodes * K, dtype=np.float32)
        is_leaf   = np.empty(n_nodes,     dtype=np.uint8)
        lib.oqtree_export(
            h, _iptr(hyp_idx), _fptr(threshold), _fptr(leaf_vals),
            is_leaf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        split_weights = np.empty(n_nodes * D, dtype=np.float32)
        lib.oqtree_get_split_weights(h, _fptr(split_weights))

        sizes = np.zeros(4, dtype=np.int32)
        lib.gf_tree_meta_sizes(h, _iptr(sizes))
        D_num, D_cat, n_entries, na_len = (int(v) for v in sizes)
        na_means  = np.zeros(max(na_len, 1),    dtype=np.float32)
        cat_sizes = np.zeros(max(D_cat, 1),     dtype=np.int32)
        cat_keys  = np.zeros(max(n_entries, 1), dtype=np.int32)
        cat_vals  = np.zeros(max(n_entries, 1), dtype=np.float32)
        lib.gf_tree_export_meta(
            h, _fptr(na_means), _iptr(cat_sizes), _iptr(cat_keys),
            _fptr(cat_vals),
        )

        base.update({
            "handle":        "serialized",
            "tree_max_depth": lib.oqtree_get_max_depth(h),
            "n_nodes":       n_nodes,
            "D":             D,
            "hyp_idx":       hyp_idx,
            "threshold":     threshold,
            "leaf_vals":     leaf_vals,
            "is_leaf":       is_leaf,
            "split_weights": split_weights,
            "D_num":         D_num,
            "D_cat":         D_cat,
            "na_len":        na_len,
            "na_means":      na_means[:na_len],
            "cat_sizes":     cat_sizes[:D_cat],
            "cat_keys":      cat_keys[:n_entries],
            "cat_vals":      cat_vals[:n_entries],
        })
        return base

    def __setstate__(self, state):
        self.max_depth    = state["max_depth"]
        self.reg_lambda   = state["reg_lambda"]
        self.subsample    = state.get("subsample", 1.0)
        self.random_state = state.get("random_state")
        self._K           = state["K"]
        self._tree_handle = None
        if state["handle"] is None:
            return
        lib = _get_oqboost_lib()
        s   = state
        handle = lib.oqtree_from_arrays(
            _iptr(s["hyp_idx"]), _fptr(s["threshold"]), _fptr(s["leaf_vals"]),
            s["is_leaf"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            s["n_nodes"], s["K"], s["tree_max_depth"],
        )
        lib.oqtree_set_split_weights(handle, s["D"], _fptr(s["split_weights"]))
        na_means  = np.ascontiguousarray(s["na_means"],  dtype=np.float32)
        cat_sizes = np.ascontiguousarray(s["cat_sizes"], dtype=np.int32)
        cat_keys  = np.ascontiguousarray(s["cat_keys"],  dtype=np.int32)
        cat_vals  = np.ascontiguousarray(s["cat_vals"],  dtype=np.float32)
        lib.gf_tree_import_meta(
            handle, s["D_num"], _fptr(na_means), s["na_len"],
            _iptr(cat_sizes), s["D_cat"], _iptr(cat_keys), _fptr(cat_vals),
        )
        self._tree_handle = handle

    def __del__(self):
        if self._tree_handle is not None:
            try:
                _get_oqboost_lib().gf_tree_free(self._tree_handle)
            except Exception:
                pass
            self._tree_handle = None


class OQBoostContext:
    """Reusable binning context for a boosting run.

    Bins X once (numeric: NaN-mean-imputed 256-bin codes; categorical:
    value dictionaries) and reuses it for every round — only the
    categorical gradient-rank re-encoding is recomputed per tree.
    """

    def __init__(self, X: np.ndarray, D_num: int | None = None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.N, self.D = X.shape
        self.D_num = self.D if D_num is None else int(D_num)
        sub = np.arange(self.N, dtype=np.int32)
        lib = _get_oqboost_lib()
        self._handle = lib.gf_ctx_create(
            _fptr(X), self.N, self.D, self.D_num, _iptr(sub), self.N
        )

    def build(
        self,
        G: np.ndarray,
        H: np.ndarray,
        sub: np.ndarray,
        max_depth: int,
        reg_lambda: float,
        inherited_rp_ratio: float = 1.0,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5,
        seed: int = 42,
        pobs: bool = True,
    ) -> tuple[OQBoostTree, np.ndarray]:
        """One boosting round → (fitted tree, predictions for all N rows)."""
        if self._handle is None:
            raise RuntimeError("Context is closed.")
        G = np.ascontiguousarray(G, dtype=np.float32)
        H = np.ascontiguousarray(H, dtype=np.float32)
        sub = np.ascontiguousarray(sub, dtype=np.int32)
        K = G.shape[1]
        out_pred = np.zeros((self.N, K), dtype=np.float32)
        lib = _get_oqboost_lib()
        handle = lib.gf_build(
            self._handle, None, _fptr(G), _fptr(H), K,
            _iptr(sub), len(sub), max_depth,
            ctypes.c_float(reg_lambda),
            ctypes.c_float(inherited_rp_ratio),
            ctypes.c_float(mutation_rate),
            ctypes.c_float(mutation_strength),
            ctypes.c_int(seed),
            ctypes.c_int(1 if pobs else 0),
            _fptr(out_pred),
        )

        tree = OQBoostTree._from_handle(handle, K, max_depth, reg_lambda)
        return tree, out_pred

    def close(self):
        if self._handle is not None:
            try:
                _get_oqboost_lib().gf_ctx_free(self._handle)
            except Exception:
                pass
            self._handle = None

    def __del__(self):
        self.close()