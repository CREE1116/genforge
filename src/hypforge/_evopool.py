import ctypes
import os
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from ._tree import _get_lib

_evopool_configured = False


def _get_evopool_lib():
    global _evopool_configured
    lib = _get_lib()
    if _evopool_configured:
        return lib

    _pf = ctypes.POINTER(ctypes.c_float)
    _pi = ctypes.POINTER(ctypes.c_int)

    lib.salot_evo_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.salot_evo_create.restype = ctypes.c_void_p

    lib.salot_evo_free.argtypes = [ctypes.c_void_p]
    lib.salot_evo_free.restype = None

    lib.salot_evo_pool_size.argtypes = [ctypes.c_void_p]
    lib.salot_evo_pool_size.restype = ctypes.c_int

    lib.salot_evo_stats.argtypes = [ctypes.c_void_p, _pf, _pi]
    lib.salot_evo_stats.restype = None

    lib.salot_evo_round.argtypes = [
        ctypes.c_void_p,  # handle
        _pf,              # X
        ctypes.c_int,     # N
        _pf,              # G
        _pf,              # H
        ctypes.c_int,     # K
        _pi,              # sub
        ctypes.c_int,     # Ns
        ctypes.c_int,     # max_depth
        ctypes.c_float,   # reg_lambda
        ctypes.c_uint,    # seed
        _pf,              # out_pred
    ]
    lib.salot_evo_round.restype = ctypes.c_void_p

    lib.salot_predict.argtypes = [
        ctypes.c_void_p,       # tree_handle
        _pf,                   # X
        ctypes.c_int,          # N
        ctypes.c_int,          # K
        _pf,                   # out_pred
    ]
    lib.salot_predict.restype = None

    lib.salot_tree_free.argtypes = [ctypes.c_void_p]
    lib.salot_tree_free.restype = None

    _evopool_configured = True
    return lib


class EvoPoolClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    EvoPool: Fast Pre-binned Oblique Tree Boosting.

    A high-performance oblique gradient boosting model that manages a pool of
    coordinate descent sparse projection features. Tree nodes are split using
    pre-binned codes on the pool without node-wise WLS.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        reg_lambda: float = 1.0,
        pool_size: int = 400,
        subsample: float = 0.8,
        early_stopping_rounds: int | None = 50,
        random_state: int | None = None,
        verbose: bool = False,
        cat_features: list | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.pool_size = pool_size
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose
        self.cat_features = cat_features

        self.trees_ = []
        self.F_init_ = []
        self._pool_handle = None

    def _get_pf(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def _get_pi(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    def fit(self, X, y, eval_set=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_in_ = None

        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int64)

        N, D = X.shape
        K = int(y.max()) + 1
        self.n_features_in_ = D
        self.classes_ = np.unique(y)

        # Initialize pool: D, D_num, cap_extra
        D_num = D
        if self.cat_features:
            D_num = D - len(self.cat_features)

        lib = _get_evopool_lib()
        self._pool_handle = lib.salot_evo_create(D, D_num, self.pool_size)

        # Initial class predictions (log odds of bincount)
        cnt = np.bincount(y, minlength=K).astype(np.float32)
        cw = N / (K * cnt.clip(min=1))
        sw = cw[y]

        lp = np.log(cnt / N + 1e-8)
        lp -= lp.mean()
        self.F_init_ = lp.tolist()

        Fsc = np.tile(lp, (N, 1)).astype(np.float32)
        F_val = None
        X_val, y_val = None, None
        if eval_set:
            X_val, y_val = eval_set[0]
            if hasattr(X_val, "values"):
                X_val = X_val.values
            X_val = np.ascontiguousarray(X_val, dtype=np.float32)
            y_val = np.ascontiguousarray(y_val, dtype=np.int64)
            F_val = np.tile(lp, (X_val.shape[0], 1)).astype(np.float32)

        rng = np.random.default_rng(self.random_state if self.random_state is not None else 42)
        best_val_loss = float("inf")
        best_trees = []
        no_improv = 0
        self.trees_ = []

        for m in range(self.n_estimators):
            Fsh = Fsc - Fsc.max(axis=1, keepdims=True)
            Pm = np.exp(Fsh)
            Pm /= Pm.sum(axis=1, keepdims=True)

            oh = np.zeros((N, K), dtype=np.float32)
            oh[np.arange(N), y] = 1.0
            G = (sw[:, None] * (Pm - oh)).astype(np.float32)
            H = (sw[:, None] * Pm * (1.0 - Pm)).astype(np.float32)

            # Evolve candidates and build tree on pre-binned features
            ns_tree = min(N, max(1000, int(N * self.subsample)))
            tree_sub = rng.choice(N, size=ns_tree, replace=False).astype(np.int32)
            out_pred = np.zeros((N, K), dtype=np.float32)

            seed = (self.random_state if self.random_state is not None else 42) + m

            tree_handle = lib.salot_evo_round(
                self._pool_handle,
                self._get_pf(X),
                N,
                self._get_pf(G),
                self._get_pf(H),
                K,
                self._get_pi(tree_sub),
                ns_tree,
                self.max_depth,
                self.reg_lambda,
                ctypes.c_uint(seed),
                self._get_pf(out_pred),
            )

            from ._tree import BFSTree
            tree = BFSTree.__new__(BFSTree)
            tree._handle = tree_handle
            tree.K = K
            tree.max_depth = self.max_depth
            self.trees_.append(tree)

            Fsc += self.learning_rate * out_pred

            # Validation & early stopping
            val_str = ""
            if X_val is not None:
                pred_val = np.zeros((X_val.shape[0], K), dtype=np.float32)
                lib.salot_predict(
                    tree_handle,
                    self._get_pf(X_val),
                    X_val.shape[0],
                    K,
                    self._get_pf(pred_val),
                )
                F_val += self.learning_rate * pred_val
                Fv_sh = F_val - F_val.max(axis=1, keepdims=True)
                P_val = np.exp(Fv_sh)
                P_val /= P_val.sum(axis=1, keepdims=True)
                val_loss = -np.log(P_val[np.arange(len(y_val)), y_val].clip(1e-8)).mean()
                val_acc = (P_val.argmax(axis=1) == y_val).mean()
                val_str = f" | ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improv = 0
                    best_trees = list(self.trees_)
                else:
                    no_improv += 1

            if self.verbose:
                ll = -np.log(Pm[np.arange(N), y].clip(1e-8)).mean()
                acc = (Pm.argmax(axis=1) == y).mean()
                pool_sz = lib.salot_evo_pool_size(self._pool_handle)
                print(
                    f"  [EvoPool] Round {m+1:3d} | Loss={ll:.4f} | Acc={acc:.4f} | "
                    f"PoolSize={pool_sz}{val_str}"
                )

            if X_val is not None and self.early_stopping_rounds is not None:
                if no_improv >= self.early_stopping_rounds:
                    if self.verbose:
                        print(f"  [EvoPool] Early stopping at round {m+1} (best ValLoss={best_val_loss:.4f})")
                    self.trees_ = best_trees
                    break

        return self

    def predict_proba(self, X):
        check_is_fitted(self, "trees_")
        if hasattr(X, "values"):
            X = X.values
        X = np.ascontiguousarray(X, dtype=np.float32)
        N = X.shape[0]
        K = len(self.classes_)

        F = np.tile(np.array(self.F_init_, dtype=np.float32), (N, 1))
        lib = _get_evopool_lib()

        for tree in self.trees_:
            pred = np.zeros((N, K), dtype=np.float32)
            lib.salot_predict(
                tree._handle,
                self._get_pf(X),
                N,
                K,
                self._get_pf(pred),
            )
            F += self.learning_rate * pred

        Fsh = F - F.max(axis=1, keepdims=True)
        P = np.exp(Fsh)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str) -> None:
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "EvoPoolClassifier":
        return joblib.load(path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pool_handle"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pool_handle = None

    def __del__(self):
        if hasattr(self, "_pool_handle") and self._pool_handle is not None:
            try:
                _get_evopool_lib().salot_evo_free(self._pool_handle)
            except Exception:
                pass
            self._pool_handle = None
