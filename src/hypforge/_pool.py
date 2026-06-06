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
        
    lib.pool_create.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.pool_create.restype = ctypes.c_void_p
    
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
        ctypes.c_float, ctypes.c_float   # reg_lambda, eta_penalty
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
    
    lib.pool_export.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),      # types
        ctypes.POINTER(ctypes.c_float),    # weights
        ctypes.POINTER(ctypes.c_int),      # h1_indices
        ctypes.POINTER(ctypes.c_int),      # h2_indices
        ctypes.POINTER(ctypes.c_int),      # n_obs
        ctypes.POINTER(ctypes.c_double),   # mu_fitness
        ctypes.POINTER(ctypes.c_double),   # M2_fitness
        ctypes.POINTER(ctypes.c_int),      # use_counts
        ctypes.POINTER(ctypes.c_int),      # rounds_since_last_use
        ctypes.POINTER(ctypes.c_float),    # fitnesses
        ctypes.POINTER(ctypes.c_float),    # scores
        ctypes.POINTER(ctypes.c_uint8)     # is_base
    ]
    lib.pool_export.restype = None
    
    lib.pool_import.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),      # types
        ctypes.POINTER(ctypes.c_float),    # weights
        ctypes.POINTER(ctypes.c_int),      # h1_indices
        ctypes.POINTER(ctypes.c_int),      # h2_indices
        ctypes.POINTER(ctypes.c_int),      # n_obs
        ctypes.POINTER(ctypes.c_double),   # mu_fitness
        ctypes.POINTER(ctypes.c_double),   # M2_fitness
        ctypes.POINTER(ctypes.c_int),      # use_counts
        ctypes.POINTER(ctypes.c_int),      # rounds_since_last_use
        ctypes.POINTER(ctypes.c_float),    # fitnesses
        ctypes.POINTER(ctypes.c_float),    # scores
        ctypes.POINTER(ctypes.c_uint8)     # is_base
    ]
    lib.pool_import.restype = ctypes.c_void_p
    
    _pool_configured = True
    return lib


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
        return int((np.abs(self.w) > 1e-3).sum())

    def eval(self, X):
        import numpy as np
        is_torch = False
        try:
            import torch
            if isinstance(X, torch.Tensor):
                is_torch = True
        except ImportError:
            pass

        if is_torch:
            w_t = torch.as_tensor(self.w, dtype=X.dtype, device=X.device)
            if self.hyp_type == "linear":
                return X @ w_t
            elif self.hyp_type == "square":
                return (X @ w_t) ** 2
            elif self.hyp_type == "abs":
                return (X @ w_t).abs()
            elif self.hyp_type == "product":
                return self.h1.eval(X) * self.h2.eval(X)
        else:
            if self.hyp_type == "linear":
                return X @ self.w
            elif self.hyp_type == "square":
                return (X @ self.w) ** 2
            elif self.hyp_type == "abs":
                return np.abs(X @ self.w)
            elif self.hyp_type == "product":
                return self.h1.eval(X) * self.h2.eval(X)
        raise ValueError(f"Unknown hyp_type: {self.hyp_type}")

    def clear_runtime_caches(self):
        self.full_cache     = None
        self.full_cache_cpu = None
        self.thresholds     = None
        self.thresholds_cpu = None
        self.cache          = None
        self.val_cache_cpu  = None


class HypForgePool:
    def __init__(self, D, max_size=500, dev="cpu"):
        self.D = D
        self.max_size = max_size
        lib = _get_pool_lib()
        self._handle = lib.pool_create(D, max_size)

    def evolve(self, X_full, G_full, H_full, sub_indices, D_num, reg_lambda=1.0, eta_penalty=0.002):
        import ctypes
        lib = _get_pool_lib()
        
        X = np.ascontiguousarray(X_full, dtype=np.float32)
        G = np.ascontiguousarray(G_full, dtype=np.float32)
        H = np.ascontiguousarray(H_full, dtype=np.float32)
        sub_indices = np.ascontiguousarray(sub_indices, dtype=np.int32)
        
        def _ptr_f(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        def _ptr_i(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        lib.pool_evolve(
            self._handle,
            _ptr_f(X), _ptr_f(G), _ptr_f(H),
            _ptr_i(sub_indices),
            X.shape[0], len(sub_indices), G.shape[1],
            D_num, reg_lambda, eta_penalty
        )

    def eval(self, X):
        import ctypes
        lib = _get_pool_lib()
        X = np.ascontiguousarray(X, dtype=np.float32)
        P = lib.pool_get_size(self._handle)
        N = X.shape[0]
        out_Z = np.empty((P, N), dtype=np.float32)
        lib.pool_eval(
            self._handle,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N,
            out_Z.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return out_Z

    def get_caches_and_thresholds(self, N):
        import ctypes
        lib = _get_pool_lib()
        P = lib.pool_get_size(self._handle)
        Z_full = np.empty((P, N), dtype=np.float32)
        thresholds = np.empty((9, P), dtype=np.float32)
        
        lib.pool_get_caches_and_thresholds(
            self._handle,
            Z_full.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            thresholds.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return Z_full, thresholds

    def update_use_counts(self, split_indices):
        import ctypes
        lib = _get_pool_lib()
        split_indices = np.ascontiguousarray(split_indices, dtype=np.int32)
        lib.pool_update_use_counts(
            self._handle,
            split_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(split_indices)
        )

    def export_pop(self):
        import ctypes
        lib = _get_pool_lib()
        P = lib.pool_get_size(self._handle)
        
        types = np.zeros(P, dtype=np.int32)
        weights = np.zeros(P * self.D, dtype=np.float32)
        h1_indices = np.zeros(P, dtype=np.int32)
        h2_indices = np.zeros(P, dtype=np.int32)
        n_obs = np.zeros(P, dtype=np.int32)
        mu_fitness = np.zeros(P, dtype=np.float64)
        M2_fitness = np.zeros(P, dtype=np.float64)
        use_counts = np.zeros(P, dtype=np.int32)
        rounds_since_last_use = np.zeros(P, dtype=np.int32)
        fitnesses = np.zeros(P, dtype=np.float32)
        scores = np.zeros(P, dtype=np.float32)
        is_base = np.zeros(P, dtype=np.uint8)
        
        def _ptr_i(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        def _ptr_f(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        def _ptr_d(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        def _ptr_u8(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        
        lib.pool_export(
            self._handle,
            _ptr_i(types),
            _ptr_f(weights),
            _ptr_i(h1_indices),
            _ptr_i(h2_indices),
            _ptr_i(n_obs),
            _ptr_d(mu_fitness),
            _ptr_d(M2_fitness),
            _ptr_i(use_counts),
            _ptr_i(rounds_since_last_use),
            _ptr_f(fitnesses),
            _ptr_f(scores),
            _ptr_u8(is_base)
        )
        
        py_pop = []
        type_names = ["linear", "square", "abs", "product"]
        for p in range(P):
            t_name = type_names[types[p]]
            w = None
            h1 = None
            h2 = None
            if types[p] < 3:
                w = weights[p * self.D : (p + 1) * self.D].copy()
            else:
                h1 = py_pop[h1_indices[p]]
                h2 = py_pop[h2_indices[p]]
            
            h = Hypothesis(hyp_type=t_name, w=w, h1=h1, h2=h2)
            h.n_obs = int(n_obs[p])
            h.mu_fitness = float(mu_fitness[p])
            h.M2_fitness = float(M2_fitness[p])
            h.use_count = int(use_counts[p])
            h.rounds_since_last_use = int(rounds_since_last_use[p])
            h.fitness = float(fitnesses[p])
            h.score = float(scores[p])
            h.is_base = bool(is_base[p])
            py_pop.append(h)
            
        return py_pop

    def import_pop(self, py_pop):
        import ctypes
        P = len(py_pop)
        type_map = {"linear": 0, "square": 1, "abs": 2, "product": 3}
        
        types = np.zeros(P, dtype=np.int32)
        weights = np.zeros(P * self.D, dtype=np.float32)
        h1_indices = np.zeros(P, dtype=np.int32)
        h2_indices = np.zeros(P, dtype=np.int32)
        n_obs = np.zeros(P, dtype=np.int32)
        mu_fitness = np.zeros(P, dtype=np.float64)
        M2_fitness = np.zeros(P, dtype=np.float64)
        use_counts = np.zeros(P, dtype=np.int32)
        rounds_since_last_use = np.zeros(P, dtype=np.int32)
        fitnesses = np.zeros(P, dtype=np.float32)
        scores = np.zeros(P, dtype=np.float32)
        is_base = np.zeros(P, dtype=np.uint8)
        
        id_to_idx = {id(h): idx for idx, h in enumerate(py_pop)}
        
        for p, h in enumerate(py_pop):
            types[p] = type_map[h.hyp_type]
            if h.hyp_type != "product":
                weights[p * self.D : (p + 1) * self.D] = h.w
                h1_indices[p] = -1
                h2_indices[p] = -1
            else:
                h1_indices[p] = id_to_idx[id(h.h1)]
                h2_indices[p] = id_to_idx[id(h.h2)]
            
            n_obs[p] = h.n_obs
            mu_fitness[p] = h.mu_fitness
            M2_fitness[p] = h.M2_fitness
            use_counts[p] = h.use_count
            rounds_since_last_use[p] = h.rounds_since_last_use
            fitnesses[p] = h.fitness
            scores[p] = h.score
            is_base[p] = 1 if h.is_base else 0
            
        def _ptr_i(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        def _ptr_f(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        def _ptr_d(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        def _ptr_u8(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        
        lib = _get_pool_lib()
        if self._handle is not None:
            lib.pool_free(self._handle)
            
        self._handle = lib.pool_import(
            self.D, self.max_size, P,
            _ptr_i(types),
            _ptr_f(weights),
            _ptr_i(h1_indices),
            _ptr_i(h2_indices),
            _ptr_i(n_obs),
            _ptr_d(mu_fitness),
            _ptr_d(M2_fitness),
            _ptr_i(use_counts),
            _ptr_i(rounds_since_last_use),
            _ptr_f(fitnesses),
            _ptr_f(scores),
            _ptr_u8(is_base)
        )

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
