"""Python bindings for TestForgePool (tpool_* C API from testforge.cpp)."""

import ctypes
import numpy as np
from ._pool import Hypothesis
from ._tree import _get_lib

_tpool_configured = False


def _get_tpool_lib():
    global _tpool_configured
    lib = _get_lib()
    if _tpool_configured:
        return lib

    lib.tpool_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.tpool_create.restype = ctypes.c_void_p

    lib.tpool_set_options.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ]
    lib.tpool_set_options.restype = None

    lib.tpool_free.argtypes = [ctypes.c_void_p]
    lib.tpool_free.restype = None

    lib.tpool_evolve.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int,
    ]
    lib.tpool_evolve.restype = None

    lib.tpool_eval.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ]
    lib.tpool_eval.restype = None

    lib.tpool_get_size.argtypes = [ctypes.c_void_p]
    lib.tpool_get_size.restype = ctypes.c_int

    lib.tpool_get_history_size.argtypes = [ctypes.c_void_p]
    lib.tpool_get_history_size.restype = ctypes.c_int

    lib.tpool_get_active_indices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.tpool_get_active_indices.restype = None

    lib.tpool_get_caches_and_thresholds.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tpool_get_caches_and_thresholds.restype = None

    lib.tpool_update_use_counts.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ]
    lib.tpool_update_use_counts.restype = None

    _tree_argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.tpool_build_tree.argtypes       = _tree_argtypes
    lib.tpool_build_tree.restype        = ctypes.c_void_p
    lib.tpool_build_salot_tree.argtypes = _tree_argtypes
    lib.tpool_build_salot_tree.restype  = ctypes.c_void_p

    lib.tpool_export.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.tpool_export.restype = None

    lib.tpool_import.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.tpool_import.restype = ctypes.c_void_p

    lib.tpool_get_transition_matrix.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tpool_get_transition_matrix.restype = None

    _tpool_configured = True
    return lib


class SALOTPool:
    """RBCD hypothesis pool for SALOT tree building.

    Backed by testforge.cpp (MVP optimizations):
      [1] DAG-based one-shot cache refresh
      [2] Alternating candidate generation (projection / crossover / product)
      [3] Top-128 hypothesis limit in tree building

    Interface is identical to HypForgePool so classifiers can swap freely.
    """

    OP_MODES   = {"all": 0, "linear_only": 1, "lrelu_only": 2}
    TYPE_NAMES = {0: "linear", 1: "leaky_relu", 2: "product"}

    def __init__(self, D, max_size=500, dev="cpu", op_mode="all",
                 crossover_top_k=6, prune_energy=0.75):
        self.D = D
        self.max_size = max_size
        lib = _get_tpool_lib()

        def _resolve(val, mapping):
            return mapping.get(val, val) if isinstance(val, str) else int(val)

        self._op_mode         = _resolve(op_mode, self.OP_MODES)
        self._crossover_top_k = int(crossover_top_k)
        self._prune_energy    = float(prune_energy)
        self._handle          = lib.tpool_create(D, max_size, 1)
        self._pop_cache       = None
        self._apply_options(lib)

    def _apply_options(self, lib=None):
        if lib is None:
            lib = _get_tpool_lib()
        lib.tpool_set_options(
            self._handle, self._op_mode, self._crossover_top_k,
            0, 0, 0, 0, 0,
            ctypes.c_float(self._prune_energy),  # energy_frac: 1 - prune_strength
            ctypes.c_float(0.0),
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _pf(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    @staticmethod
    def _pi(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    @staticmethod
    def _pu8(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    # ── public API ────────────────────────────────────────────────────────────

    def evolve(self, X_full, G_full, H_full, sub_indices, D_num,
               reg_lambda=1.0, eta_penalty=0.002, current_round=0):
        lib = _get_tpool_lib()
        X   = np.ascontiguousarray(X_full,     dtype=np.float32)
        G   = np.ascontiguousarray(G_full,     dtype=np.float32)
        H   = np.ascontiguousarray(H_full,     dtype=np.float32)
        sub = np.ascontiguousarray(sub_indices, dtype=np.int32)
        self._pop_cache = None
        lib.tpool_evolve(
            self._handle,
            self._pf(X), self._pf(G), self._pf(H),
            self._pi(sub),
            X.shape[0], len(sub), G.shape[1],
            D_num, reg_lambda, eta_penalty, int(current_round),
        )

    def eval(self, X):
        lib = _get_tpool_lib()
        X = np.ascontiguousarray(X, dtype=np.float32)
        P = lib.tpool_get_size(self._handle)
        N = X.shape[0]
        out_Z = np.empty((P, N), dtype=np.float32)
        lib.tpool_eval(self._handle, self._pf(X), N, self._pf(out_Z))
        return out_Z

    def get_caches_and_thresholds(self, N):
        lib = _get_tpool_lib()
        P          = lib.tpool_get_size(self._handle)
        Z_full     = np.empty((P, N), dtype=np.float32)
        thresholds = np.empty((9, P), dtype=np.float32)
        lib.tpool_get_caches_and_thresholds(
            self._handle, self._pf(Z_full), self._pf(thresholds))
        return Z_full, thresholds

    def update_use_counts(self, split_indices):
        lib  = _get_tpool_lib()
        idxs = np.ascontiguousarray(split_indices, dtype=np.int32)
        lib.tpool_update_use_counts(self._handle, self._pi(idxs), len(idxs))

    def build_tree(self, X_full, G_full, H_full,
                   evolve_sub, tree_sub,
                   D_num, max_depth,
                   reg_lambda=1.0, eta_penalty=0.002,
                   do_evolve=True, current_round=0):
        from ._tree import BFSTree, _get_lib as _get_tree_lib
        lib = _get_tpool_lib()

        X      = np.ascontiguousarray(X_full,    dtype=np.float32)
        G      = np.ascontiguousarray(G_full,    dtype=np.float32)
        H      = np.ascontiguousarray(H_full,    dtype=np.float32)
        ev_sub = np.ascontiguousarray(evolve_sub, dtype=np.int32)
        tr_sub = np.ascontiguousarray(tree_sub,   dtype=np.int32)

        N        = X.shape[0]
        K        = G.shape[1]
        out_pred = np.zeros((N, K), dtype=np.float32)

        # Use SALOT (shared-axis leaf-wise oblique tree) by default.
        # tpool_build_salot_tree has identical C-API signature to tpool_build_tree.
        tree_handle = lib.tpool_build_salot_tree(
            self._handle,
            self._pf(X), self._pf(G), self._pf(H),
            N, K,
            self._pi(ev_sub), len(ev_sub),
            self._pi(tr_sub), len(tr_sub),
            D_num, max_depth,
            ctypes.c_float(reg_lambda),
            ctypes.c_float(eta_penalty),
            ctypes.c_int(1 if do_evolve else 0),
            self._pf(out_pred),
            ctypes.c_int(current_round),
        )

        self._pop_cache  = None
        tlib             = _get_tree_lib()
        tree             = BFSTree.__new__(BFSTree)
        tree._handle     = tree_handle
        tree.K           = tlib.bfstree_get_K(tree_handle)
        tree.max_depth   = tlib.bfstree_get_max_depth(tree_handle)
        return tree, out_pred

    def get_transition_matrix(self):
        lib       = _get_tpool_lib()
        births    = np.zeros((3, 3), dtype=np.float32)
        survivors = np.zeros((3, 3), dtype=np.float32)
        lib.tpool_get_transition_matrix(
            self._handle, self._pf(births), self._pf(survivors))
        return births, survivors

    # ── export / import (mirrors HypForgePool interface) ─────────────────────

    def export_pop(self):
        lib = _get_tpool_lib()
        P   = lib.tpool_get_size(self._handle)
        H   = lib.tpool_get_history_size(self._handle)

        types                 = np.zeros(H, dtype=np.int32)
        weights               = np.zeros(H * self.D, dtype=np.float32)
        h1_indices            = np.zeros(H, dtype=np.int32)
        h2_indices            = np.zeros(H, dtype=np.int32)
        use_counts            = np.zeros(H, dtype=np.int32)
        rounds_since_last_use = np.zeros(H, dtype=np.int32)
        credits               = np.zeros(H, dtype=np.float32)
        is_base               = np.zeros(H, dtype=np.uint8)
        _stub                 = np.zeros(H, dtype=np.int32)

        lib.tpool_export(
            self._handle,
            self._pi(types), self._pf(weights),
            self._pi(h1_indices), self._pi(h2_indices),
            self._pi(use_counts), self._pi(rounds_since_last_use),
            self._pf(credits), self._pu8(is_base),
            self._pi(_stub), self._pi(_stub), self._pi(_stub), self._pi(_stub),
        )

        active_indices = np.zeros(P, dtype=np.int32)
        lib.tpool_get_active_indices(self._handle, self._pi(active_indices))

        TYPE_NAMES = {0: "linear", 1: "leaky_relu", 2: "product"}
        py_history: list[Hypothesis] = []
        for p in range(H):
            t_name = TYPE_NAMES.get(int(types[p]), f"type_{types[p]}")
            if types[p] != 2:
                w  = weights[p * self.D : (p + 1) * self.D].copy()
                h1 = h2 = None
            else:
                w     = None
                h1_id = int(h1_indices[p])
                h2_id = int(h2_indices[p])
                h1 = py_history[h1_id] if 0 <= h1_id < len(py_history) else None
                h2 = py_history[h2_id] if 0 <= h2_id < len(py_history) else None

            h                       = Hypothesis(hyp_type=t_name, w=w, h1=h1, h2=h2)
            h.use_count             = int(use_counts[p])
            h.rounds_since_last_use = int(rounds_since_last_use[p])
            h.credit                = float(credits[p])
            h.is_base               = bool(is_base[p])
            py_history.append(h)

        return [py_history[idx] for idx in active_indices if 0 <= idx < H]

    @property
    def pop(self):
        if self._pop_cache is not None:
            return self._pop_cache
        return self.export_pop()

    @pop.setter
    def pop(self, value):
        # Called by the classifier's early-stopping restore path; cache the list.
        self._pop_cache = value

    def __del__(self):
        if hasattr(self, "_handle") and self._handle is not None:
            try:
                _get_tpool_lib().tpool_free(self._handle)
            except Exception:
                pass
            self._handle = None


# Backwards-compat alias
TestForgePool = SALOTPool
