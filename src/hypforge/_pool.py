import os
import ctypes
import numpy as np
from ._tree import _get_lib

_pool_configured = False

def _get_pool_lib():
    global _pool_configured
    lib = _get_lib()
    if _pool_configured:
        return lib
        
    lib.pool_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.pool_create.restype = ctypes.c_void_p

    lib.pool_set_options.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float
    ]
    lib.pool_set_options.restype = None

    lib.pool_free.argtypes = [ctypes.c_void_p]
    lib.pool_free.restype = None
    
    lib.pool_evolve.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),  # X
        ctypes.POINTER(ctypes.c_float),  # G
        ctypes.POINTER(ctypes.c_float),  # H
        ctypes.POINTER(ctypes.c_int),    # sub_indices
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # N, Ns, K
        ctypes.c_int,                    # D_num
        ctypes.c_float, ctypes.c_float,  # reg_lambda, eta_penalty
        ctypes.c_int                     # current_round
    ]
    lib.pool_evolve.restype = None
    
    lib.pool_eval.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.pool_eval.restype = None
    
    lib.pool_get_size.argtypes = [ctypes.c_void_p]
    lib.pool_get_size.restype = ctypes.c_int

    lib.pool_get_history_size.argtypes = [ctypes.c_void_p]
    lib.pool_get_history_size.restype = ctypes.c_int

    lib.pool_get_active_indices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.pool_get_active_indices.restype = None
    
    lib.pool_get_caches_and_thresholds.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.pool_get_caches_and_thresholds.restype = None
    
    lib.pool_update_use_counts.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    lib.pool_update_use_counts.restype = None

    lib.pool_build_tree.argtypes = [
        ctypes.c_void_p,                              # pool_handle
        ctypes.POINTER(ctypes.c_float),               # X [N, D]
        ctypes.POINTER(ctypes.c_float),               # G [N, K]
        ctypes.POINTER(ctypes.c_float),               # H [N, K]
        ctypes.c_int, ctypes.c_int,                   # N, K
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,   # evolve_sub, Ns_evolve
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,   # tree_sub, Ns_tree
        ctypes.c_int, ctypes.c_int,                   # D_num, max_depth
        ctypes.c_float, ctypes.c_float,               # reg_lambda, eta_penalty
        ctypes.c_int,                                 # do_evolve
        ctypes.POINTER(ctypes.c_float),               # out_pred [N, K]
        ctypes.c_int                                  # current_round
    ]
    lib.pool_build_tree.restype = ctypes.c_void_p

    lib.pool_export.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),      # types
        ctypes.POINTER(ctypes.c_float),    # weights
        ctypes.POINTER(ctypes.c_int),      # h1_indices
        ctypes.POINTER(ctypes.c_int),      # h2_indices
        ctypes.POINTER(ctypes.c_int),      # use_counts
        ctypes.POINTER(ctypes.c_int),      # rounds_since_last_use
        ctypes.POINTER(ctypes.c_float),    # credits
        ctypes.POINTER(ctypes.c_uint8),    # is_base
        ctypes.POINTER(ctypes.c_int),      # parent1
        ctypes.POINTER(ctypes.c_int),      # parent2
        ctypes.POINTER(ctypes.c_int),      # birth_round
        ctypes.POINTER(ctypes.c_int),      # family_id
        ctypes.POINTER(ctypes.c_float),    # family_fitness
        ctypes.POINTER(ctypes.c_float),    # breeding_value
        ctypes.POINTER(ctypes.c_float),    # ancestor_credit
    ]
    lib.pool_export.restype = None

    lib.pool_import.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),      # types
        ctypes.POINTER(ctypes.c_float),    # weights
        ctypes.POINTER(ctypes.c_int),      # h1_indices
        ctypes.POINTER(ctypes.c_int),      # h2_indices
        ctypes.POINTER(ctypes.c_int),      # use_counts
        ctypes.POINTER(ctypes.c_int),      # rounds_since_last_use
        ctypes.POINTER(ctypes.c_float),    # credits
        ctypes.POINTER(ctypes.c_uint8),    # is_base
        ctypes.POINTER(ctypes.c_int),      # parent1
        ctypes.POINTER(ctypes.c_int),      # parent2
        ctypes.POINTER(ctypes.c_int),      # birth_round
        ctypes.POINTER(ctypes.c_int),      # family_id
        ctypes.POINTER(ctypes.c_float),    # family_fitness
        ctypes.POINTER(ctypes.c_float),    # breeding_value
        ctypes.POINTER(ctypes.c_float),    # ancestor_credit
        ctypes.c_int,                      # P
        ctypes.POINTER(ctypes.c_int),      # active_indices
    ]
    lib.pool_import.restype = ctypes.c_void_p

    lib.pool_get_transition_matrix.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),    # births
        ctypes.POINTER(ctypes.c_float)     # survivors
    ]
    lib.pool_get_transition_matrix.restype = None
    
    _pool_configured = True
    return lib


class Hypothesis:
    """Oblique split candidate: a projection direction + type."""

    def __init__(self, hyp_type="linear", w=None, h1=None, h2=None):
        self.hyp_type = hyp_type
        self.w        = w
        self.h1       = h1
        self.h2       = h2

        self.use_count             = 0
        self.rounds_since_last_use = 0
        self.credit                = 0.0
        self.is_base               = False

        self.parent1         = -1
        self.parent2         = -1
        self.birth_round     = 0
        self.family_id       = -1
        self.family_fitness  = 0.0
        self.breeding_value  = 0.0
        self.ancestor_credit = 0.0

        # projection cache (cleared before pickling)
        self.full_cache = None
        self.thresholds = None

    @property
    def fitness(self) -> float:
        return self.credit

    @fitness.setter
    def fitness(self, val: float):
        self.credit = val

    @property
    def score(self) -> float:
        return self.credit

    @score.setter
    def score(self, val: float):
        self.credit = val

    def complexity(self):
        if self.hyp_type == "product":
            return self.h1.complexity() + self.h2.complexity()
        return int((np.abs(self.w) > 1e-3).sum())

    def eval(self, X):
        import numpy as np
        w = self.w
        try:
            import torch
            if isinstance(w, torch.Tensor):
                w = w.detach().cpu().numpy()
        except ImportError:
            pass

        is_torch = False
        try:
            import torch
            if isinstance(X, torch.Tensor):
                is_torch = True
        except ImportError:
            pass

        if self.hyp_type == "product" and self.h1 is not None and self.h2 is not None:
            return self.h1.eval(X) * self.h2.eval(X)

        if is_torch:
            import torch
            w_t = torch.as_tensor(w, dtype=X.dtype, device=X.device)
            z = X @ w_t
            if self.hyp_type == "leaky_relu":
                return torch.where(z > 0, z, 0.01 * z)
            return z
        else:
            z = X @ w
            if self.hyp_type == "leaky_relu":
                return np.where(z > 0, z, 0.01 * z)
            return z

    def clear_runtime_caches(self):
        self.full_cache = None
        self.thresholds = None


class HypForgePool:
    OP_MODES   = {"all": 0, "linear_only": 1, "lrelu_only": 2}
    TYPE_NAMES = {0: "linear", 1: "leaky_relu", 2: "product"}

    def __init__(self, D, max_size=500, dev="cpu",
                 op_mode="all", crossover_top_k=6):
        self.D = D
        self.max_size = max_size
        lib = _get_pool_lib()

        def _resolve(val, mapping):
            return mapping.get(val, val) if isinstance(val, str) else int(val)

        self._op_mode         = _resolve(op_mode, self.OP_MODES)
        self._crossover_top_k = int(crossover_top_k)

        self._handle = lib.pool_create(D, max_size, 1)
        self._apply_options(lib)

    def _apply_options(self, lib=None):
        if lib is None:
            lib = _get_pool_lib()
        # C++ pool_set_options uses only op_mode, crossover_top_k, elitism_k;
        # the remaining 6 args are stub slots kept for ABI compatibility.
        lib.pool_set_options(
            self._handle,
            self._op_mode,
            self._crossover_top_k,
            0,    # elitism_k (hardcoded — no Python-side knob needed)
            0, 0, 0, 0, 0.0, 0.0,  # stub slots
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pf(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    @staticmethod
    def _pi(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    @staticmethod
    def _pu8(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    # ── public API ────────────────────────────────────────────────────────────

    def evolve(self, X_full, G_full, H_full, sub_indices, D_num,
               reg_lambda=1.0, eta_penalty=0.002, current_round=0):
        lib = _get_pool_lib()
        X   = np.ascontiguousarray(X_full,      dtype=np.float32)
        G   = np.ascontiguousarray(G_full,      dtype=np.float32)
        H   = np.ascontiguousarray(H_full,      dtype=np.float32)
        sub = np.ascontiguousarray(sub_indices,  dtype=np.int32)
        lib.pool_evolve(
            self._handle,
            self._pf(X), self._pf(G), self._pf(H),
            self._pi(sub),
            X.shape[0], len(sub), G.shape[1],
            D_num, reg_lambda, eta_penalty, int(current_round),
        )

    def eval(self, X):
        lib = _get_pool_lib()
        X = np.ascontiguousarray(X, dtype=np.float32)
        P = lib.pool_get_size(self._handle)
        N = X.shape[0]
        out_Z = np.empty((P, N), dtype=np.float32)
        lib.pool_eval(self._handle, self._pf(X), N, self._pf(out_Z))
        return out_Z

    def get_caches_and_thresholds(self, N):
        lib = _get_pool_lib()
        P          = lib.pool_get_size(self._handle)
        Z_full     = np.empty((P, N),    dtype=np.float32)
        thresholds = np.empty((9, P),    dtype=np.float32)
        lib.pool_get_caches_and_thresholds(
            self._handle, self._pf(Z_full), self._pf(thresholds))
        return Z_full, thresholds

    def update_use_counts(self, split_indices):
        lib  = _get_pool_lib()
        idxs = np.ascontiguousarray(split_indices, dtype=np.int32)
        lib.pool_update_use_counts(self._handle, self._pi(idxs), len(idxs))

    def build_tree(self, X_full, G_full, H_full,
                   evolve_sub, tree_sub,
                   D_num, max_depth,
                   reg_lambda=1.0, eta_penalty=0.002,
                   do_evolve=True, current_round=0):
        """Evolve pool + build BFS tree + predict on all N.

        Returns (BFSTree, pred_np) where pred_np is float32 [N, K].
        """
        from ._tree import BFSTree, _get_lib as _get_tree_lib

        lib = _get_pool_lib()

        X      = np.ascontiguousarray(X_full,     dtype=np.float32)
        G      = np.ascontiguousarray(G_full,     dtype=np.float32)
        H      = np.ascontiguousarray(H_full,     dtype=np.float32)
        ev_sub = np.ascontiguousarray(evolve_sub,  dtype=np.int32)
        tr_sub = np.ascontiguousarray(tree_sub,    dtype=np.int32)

        N        = X.shape[0]
        K        = G.shape[1]
        out_pred = np.zeros((N, K), dtype=np.float32)

        tree_handle = lib.pool_build_tree(
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

        tlib = _get_tree_lib()
        tree           = BFSTree.__new__(BFSTree)
        tree._handle   = tree_handle
        tree.K         = tlib.bfstree_get_K(tree_handle)
        tree.max_depth = tlib.bfstree_get_max_depth(tree_handle)
        return tree, out_pred

    # ── export / import ───────────────────────────────────────────────────────

    def export_pop(self):
        lib = _get_pool_lib()
        P   = lib.pool_get_size(self._handle)
        H   = lib.pool_get_history_size(self._handle)

        types                 = np.zeros(H, dtype=np.int32)
        weights               = np.zeros(H * self.D, dtype=np.float32)
        h1_indices            = np.zeros(H, dtype=np.int32)
        h2_indices            = np.zeros(H, dtype=np.int32)
        use_counts            = np.zeros(H, dtype=np.int32)
        rounds_since_last_use = np.zeros(H, dtype=np.int32)
        credits               = np.zeros(H, dtype=np.float32)
        is_base               = np.zeros(H, dtype=np.uint8)
        parent1               = np.zeros(H, dtype=np.int32)
        parent2               = np.zeros(H, dtype=np.int32)
        birth_round           = np.zeros(H, dtype=np.int32)
        family_id             = np.zeros(H, dtype=np.int32)
        family_fitness        = np.zeros(H, dtype=np.float32)
        breeding_value        = np.zeros(H, dtype=np.float32)
        ancestor_credit       = np.zeros(H, dtype=np.float32)

        lib.pool_export(
            self._handle,
            self._pi(types),
            self._pf(weights),
            self._pi(h1_indices),
            self._pi(h2_indices),
            self._pi(use_counts),
            self._pi(rounds_since_last_use),
            self._pf(credits),
            self._pu8(is_base),
            self._pi(parent1),
            self._pi(parent2),
            self._pi(birth_round),
            self._pi(family_id),
            self._pf(family_fitness),
            self._pf(breeding_value),
            self._pf(ancestor_credit),
        )

        active_indices = np.zeros(P, dtype=np.int32)
        lib.pool_get_active_indices(self._handle, self._pi(active_indices))

        py_history: list["Hypothesis"] = []
        for p in range(H):
            t_name = self.TYPE_NAMES.get(int(types[p]), f"type_{types[p]}")

            if types[p] != 2:
                w  = weights[p * self.D : (p + 1) * self.D].copy()
                h1 = None
                h2 = None
            else:
                w      = None
                h1_id  = int(h1_indices[p])
                h2_id  = int(h2_indices[p])
                h1 = py_history[h1_id] if 0 <= h1_id < len(py_history) else None
                h2 = py_history[h2_id] if 0 <= h2_id < len(py_history) else None

            h                       = Hypothesis(hyp_type=t_name, w=w, h1=h1, h2=h2)
            h.use_count             = int(use_counts[p])
            h.rounds_since_last_use = int(rounds_since_last_use[p])
            h.credit                = float(credits[p])
            h.is_base               = bool(is_base[p])
            h.parent1               = int(parent1[p])
            h.parent2               = int(parent2[p])
            h.birth_round           = int(birth_round[p])
            h.family_id             = int(family_id[p])
            h.family_fitness        = float(family_fitness[p])
            h.breeding_value        = float(breeding_value[p])
            h.ancestor_credit       = float(ancestor_credit[p])
            py_history.append(h)

        return [py_history[idx] for idx in active_indices if 0 <= idx < H]

    def import_pop(self, py_pop):
        type_map = {"linear": 0, "leaky_relu": 1, "product": 2}

        unique_hyps: list["Hypothesis"] = []
        seen: set = set()

        def _collect(h):
            if h is None or id(h) in seen:
                return
            if h.hyp_type == "product":
                _collect(h.h1)
                _collect(h.h2)
            seen.add(id(h))
            unique_hyps.append(h)

        for h in py_pop:
            _collect(h)

        id_to_idx = {id(h): i for i, h in enumerate(unique_hyps)}

        U                     = len(unique_hyps)
        types                 = np.zeros(U, dtype=np.int32)
        weights               = np.zeros(U * self.D, dtype=np.float32)
        h1_indices            = np.full(U, -1, dtype=np.int32)
        h2_indices            = np.full(U, -1, dtype=np.int32)
        use_counts            = np.zeros(U, dtype=np.int32)
        rounds_since_last_use = np.zeros(U, dtype=np.int32)
        credits               = np.zeros(U, dtype=np.float32)
        is_base               = np.zeros(U, dtype=np.uint8)
        parent1               = np.full(U, -1, dtype=np.int32)
        parent2               = np.full(U, -1, dtype=np.int32)
        birth_round           = np.zeros(U, dtype=np.int32)
        family_id             = np.full(U, -1, dtype=np.int32)
        family_fitness        = np.zeros(U, dtype=np.float32)
        breeding_value        = np.zeros(U, dtype=np.float32)
        ancestor_credit       = np.zeros(U, dtype=np.float32)

        for u, h in enumerate(unique_hyps):
            types[u] = type_map[h.hyp_type]
            if h.hyp_type != "product":
                weights[u * self.D : (u + 1) * self.D] = h.w
            else:
                if h.h1 is not None and id(h.h1) in id_to_idx:
                    h1_indices[u] = id_to_idx[id(h.h1)]
                if h.h2 is not None and id(h.h2) in id_to_idx:
                    h2_indices[u] = id_to_idx[id(h.h2)]

            use_counts[u]            = h.use_count
            rounds_since_last_use[u] = h.rounds_since_last_use
            credits[u]               = h.credit
            is_base[u]               = 1 if h.is_base else 0
            parent1[u]               = h.parent1
            parent2[u]               = h.parent2
            birth_round[u]           = h.birth_round
            family_id[u]             = h.family_id
            family_fitness[u]        = getattr(h, "family_fitness", 0.0)
            breeding_value[u]        = getattr(h, "breeding_value", 0.0)
            ancestor_credit[u]       = getattr(h, "ancestor_credit", 0.0)

        P              = len(py_pop)
        active_indices = np.array([id_to_idx[id(h)] for h in py_pop], dtype=np.int32)

        lib = _get_pool_lib()
        if self._handle is not None:
            lib.pool_free(self._handle)

        self._handle = lib.pool_import(
            self.D, self.max_size, U,
            self._pi(types),
            self._pf(weights),
            self._pi(h1_indices),
            self._pi(h2_indices),
            self._pi(use_counts),
            self._pi(rounds_since_last_use),
            self._pf(credits),
            self._pu8(is_base),
            self._pi(parent1),
            self._pi(parent2),
            self._pi(birth_round),
            self._pi(family_id),
            self._pf(family_fitness),
            self._pf(breeding_value),
            self._pf(ancestor_credit),
            P,
            self._pi(active_indices),
        )
        self._apply_options()

    def get_transition_matrix(self):
        lib       = _get_pool_lib()
        births    = np.zeros((3, 3), dtype=np.float32)
        survivors = np.zeros((3, 3), dtype=np.float32)
        lib.pool_get_transition_matrix(
            self._handle,
            self._pf(births),
            self._pf(survivors),
        )
        return births, survivors

    @property
    def pop(self):
        return self.export_pop()

    @pop.setter
    def pop(self, py_pop):
        self.import_pop(py_pop)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle is not None:
            try:
                _get_pool_lib().pool_free(self._handle)
            except Exception:
                pass
            self._handle = None