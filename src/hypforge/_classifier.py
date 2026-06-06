from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._pool import Hypothesis, HypForgePool
from ._tree import BFSTree


class HypForgeClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    HypForge: Hypothesis Pool Evolution for Oblique GBDT.

    Gradient-boosted oblique decision trees where split directions are drawn
    from an evolving pool of projections (linear, relu, product combinations
    of gradient-aligned and synergy hypotheses).

    Implements both ``ClassifierMixin`` (predict/predict_proba) and
    ``TransformerMixin`` (transform), so it can be used as a feature-selection
    stage in an sklearn Pipeline::

        Pipeline([
            ('hf', HypForgeClassifier(n_estimators=100)),  # fit+select features
            ('xgb', XGBClassifier()),
        ])

    Or manually::

        clf.fit(X_train, y_train)
        X_sel = clf.transform(X_train)   # only the features HypForge used
        xgb.fit(X_sel, y_train)

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage applied to each tree's leaf values.
    max_depth : int
        Maximum tree depth (0-indexed; depth=6 → 127 nodes).
    reg_lambda : float
        L2 regularisation on leaf weights (Newton step denominator).
    pool_size : int
        Maximum number of hypotheses kept in the pool.
    evolve_every : int
        Evolve the hypothesis pool every N boosting rounds.
    subsample : float
        Fraction of training samples used to build each tree.
    early_stopping_rounds : int or None
        Stop if validation loss does not improve for this many rounds.
    device : str
        'auto' selects MPS > CUDA > CPU.  Pass 'cpu', 'cuda', or 'mps' to force.
    random_state : int or None
        Seed for reproducibility.
    verbose : bool
        Print per-round metrics during training.
    cat_features : list of str or int, optional
        Column names (if X is a DataFrame) or column indices treated as
        categorical.  These features are excluded from numerical projections.
    """

    def __init__(
        self,
        n_estimators: int   = 500,
        learning_rate: float = 0.05,
        max_depth: int       = 4,
        reg_lambda: float    = 1.0,
        pool_size: int       = 400,
        evolve_every: int    = 1,
        subsample: float     = 0.8,
        early_stopping_rounds: int | None = 50,
        device: str          = "auto",
        random_state: int | None = None,
        verbose: bool        = False,
        cat_features: list | None = None,
        crossover_top_k: int = 3,
        elitism_k: int       = 20,
        map_elites_slots: int = 100,
    ):
        self.n_estimators          = n_estimators
        self.learning_rate         = learning_rate
        self.max_depth             = max_depth
        self.reg_lambda            = reg_lambda
        self.pool_size             = pool_size
        self.evolve_every          = evolve_every
        self.subsample             = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.device                = device
        self.random_state          = random_state
        self.verbose               = verbose
        self.cat_features          = cat_features
        self.crossover_top_k       = crossover_top_k
        self.elitism_k             = elitism_k
        self.map_elites_slots      = map_elites_slots

    # ── public fit/predict ────────────────────────────────────────────────────

    def fit(
        self,
        X,
        y,
        eval_set: list[tuple] | None = None,
        sample_weight=None,
    ) -> "HypForgeClassifier":
        """
        Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        eval_set : list of (X_val, y_val) tuples, optional
            First tuple is used for early stopping and validation metrics.
        sample_weight : ignored (class weights computed internally from y)
        """
        import pandas as pd

        # ── feature name bookkeeping ─────────────────────────────────────────
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_in_ = None

        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.int64)

        self.n_features_in_ = X.shape[1]
        self.classes_       = np.unique(y)

        # ── resolve categorical column indices ────────────────────────────────
        D_num = self._resolve_D_num(X.shape[1])

        # ── unpack eval_set ──────────────────────────────────────────────────
        X_val, y_val = None, None
        if eval_set:
            X_val, y_val = eval_set[0]
            if hasattr(X_val, "values"):
                X_val = X_val.values
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)

        self._fit_core(X, y, X_val, y_val, D_num)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "trees_")
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "trees_")

        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        lp    = np.array(self.F_init_, dtype=np.float32)
        Fsc   = np.tile(lp, (X.shape[0], 1))

        # pool.pop creates new Python objects every call so id() never repeats.
        # Key by (type, weight-bytes) instead to deduplicate across rounds.
        z_cache: dict = {}

        def _hyp_key(h):
            if h.hyp_type == "product":
                return ("product", _hyp_key(h.h1), _hyp_key(h.h2))
            return (h.hyp_type, h.w.tobytes())

        def _eval_cached(h):
            key = _hyp_key(h)
            if key not in z_cache:
                if h.hyp_type == "linear":
                    z_cache[key] = X @ h.w
                elif h.hyp_type == "leaky_relu":
                    z = X @ h.w; z_cache[key] = np.where(z > 0, z, 0.01 * z)
                elif h.hyp_type == "product":
                    z_cache[key] = _eval_cached(h.h1) * _eval_cached(h.h2)
            return z_cache[key]

        for tree, snap in zip(self.trees_, self.pool_snaps_):
            Z_rows = [_eval_cached(h) for h in snap]
            Z_pred = np.ascontiguousarray(np.stack(Z_rows))
            Fsc += self.learning_rate * tree.predict(Z_pred)

        Fsc -= Fsc.max(1, keepdims=True)
        exp_F = np.exp(Fsc)
        return exp_F / exp_F.sum(1, keepdims=True)

    # ── feature importance ────────────────────────────────────────────────────

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Gain-weighted feature importances derived from hypothesis projection
        weights.  For each split in every tree the hypothesis weight vector w
        is used as a per-feature attribution; accumulated gain × |w[f]| gives
        the importance of feature f.

        Returns
        -------
        ndarray of shape (n_features_in_,)
        """
        check_is_fitted(self, "trees_")
        imp = np.zeros(self.n_features_in_, dtype=np.float64)
        for h in self._pool_hypotheses_:
            if h.hyp_type not in ("linear", "relu") or h.w is None:
                continue
            w_abs  = np.abs(h.w).astype(np.float64)
            weight = float(h.use_count) if h.use_count > 0 else max(h.fitness, 0.0)
            imp   += w_abs * weight
        total = imp.sum()
        if total > 0:
            imp /= total
        return imp

    def get_feature_importances(self, feature_names: list | None = None) -> dict:
        """
        Return a {feature_name: importance} dict, sorted descending.

        Parameters
        ----------
        feature_names : list of str, optional
            If omitted, uses ``feature_names_in_`` set during fit (if X was a
            DataFrame) or falls back to f0, f1, … notation.
        """
        if feature_names is None:
            feature_names = self.feature_names_in_ or [f"f{i}" for i in range(self.n_features_in_)]
        imps = self.feature_importances_
        return dict(sorted(zip(feature_names, imps), key=lambda kv: kv[1], reverse=True))

    def plot_importance(
        self,
        max_features: int = 20,
        feature_names: list | None = None,
        ax=None,
        title: str = "HypForge Feature Importances",
    ):
        """
        Bar chart of top-N feature importances.

        Requires matplotlib.  Returns the Axes object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Install matplotlib to use plot_importance().")

        imp_dict = self.get_feature_importances(feature_names)
        names    = list(imp_dict.keys())[:max_features]
        values   = list(imp_dict.values())[:max_features]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))

        ax.barh(names[::-1], values[::-1])
        ax.set_xlabel("Importance")
        ax.set_title(title)
        ax.tight_layout()
        return ax

    # ── feature selection & transformation ───────────────────────────────────

    def get_feature_mask(self, threshold: float = 1e-3) -> np.ndarray:
        """
        Boolean mask [n_features_in_] — True for features with non-zero weight
        in at least one hypothesis of the final pool.

        Parameters
        ----------
        threshold : float
            Minimum absolute weight to count a feature as "used".
        """
        check_is_fitted(self, "_pool_hypotheses_")
        mask = np.zeros(self.n_features_in_, dtype=bool)
        for h in self._pool_hypotheses_:
            if h.hyp_type in ("linear", "leaky_relu") and h.w is not None:
                mask |= (np.abs(h.w) > threshold)
        return mask

    def get_used_features(
        self, threshold: float = 1e-3, feature_names: list | None = None
    ) -> list:
        """
        Return the names (or indices) of original features the model actually
        used — i.e. features with non-zero weight in at least one hypothesis.

        Parameters
        ----------
        threshold : float
            Minimum absolute weight to count a feature as "used".
        feature_names : list of str, optional
            Column names.  Falls back to ``feature_names_in_`` or "f{i}" notation.

        Returns
        -------
        list of str or int
            Sorted by feature importance (descending).
        """
        check_is_fitted(self, "_pool_hypotheses_")
        names = feature_names or self.feature_names_in_ or list(range(self.n_features_in_))
        mask  = self.get_feature_mask(threshold)
        imps  = self.feature_importances_
        used  = [(names[i], imps[i]) for i in range(self.n_features_in_) if mask[i]]
        return [name for name, _ in sorted(used, key=lambda kv: kv[1], reverse=True)]

    def transform(self, X, threshold: float = 1e-3) -> np.ndarray:
        """
        sklearn-compatible transformer: filter X to the columns HypForge used.

        Enables use as a feature-selection stage in ``sklearn.pipeline.Pipeline``.
        ``fit_transform(X, y)`` trains HypForge on (X, y) then returns the
        reduced feature matrix in one call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)
        threshold : float
            Minimum hypothesis weight to count a feature as "used".

        Returns
        -------
        ndarray of shape (n_samples, n_used_features)
        """
        return self.select(X, threshold=threshold)

    def select(self, X, threshold: float = 1e-3) -> np.ndarray:
        """
        Filter X to the columns actually used by the model.

        Same as ``transform()`` but more explicit; use whichever reads clearer
        in your code.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)
        threshold : float
            Minimum hypothesis weight to treat a feature as "used".

        Returns
        -------
        ndarray of shape (n_samples, n_used_features)
        """
        if hasattr(X, "values"):
            X = X.values
        X    = np.asarray(X, dtype=np.float32)
        mask = self.get_feature_mask(threshold)
        return X[:, mask]

    def embed(self, X, top_k: int | None = None) -> np.ndarray:
        """
        Project X through the learned hypothesis pool → learned embedding.

        Each column is the projection of X onto one hypothesis direction
        (a nonlinear feature combining raw inputs).  Useful as learned features
        for a downstream model — captures the nonlinear interactions HypForge
        discovered.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_in_)
        top_k : int or None
            Keep only the top-K hypotheses ranked by fitness.  None → all.

        Returns
        -------
        ndarray of shape (n_samples, n_hypotheses)  float32
        """
        check_is_fitted(self, "_pool_hypotheses_")
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        hyps = self._pool_hypotheses_
        if top_k is not None:
            hyps = sorted(hyps, key=lambda h: h.fitness, reverse=True)[:top_k]

        return np.stack([h.eval(X) for h in hyps], axis=1).astype(np.float32)

    def get_hypothesis_summary(self) -> "pd.DataFrame":
        """
        Return a DataFrame summarising every hypothesis in the final pool.

        Columns
        -------
        type        : 'linear' | 'relu' | 'product'
        fitness     : GBDT split-gain score
        use_count   : times this hypothesis was selected as a split direction
        complexity  : number of non-zero weights (or sum for product types)
        top_features: comma-separated list of the highest-weight features
        """
        import pandas as pd

        check_is_fitted(self, "_pool_hypotheses_")
        names = self.feature_names_in_ or [f"f{i}" for i in range(self.n_features_in_)]

        rows = []
        for h in self._pool_hypotheses_:
            top_feats = ""
            if h.hyp_type in ("linear", "leaky_relu") and h.w is not None:
                w_abs = np.abs(h.w)
                top_k = min(5, int((w_abs > 1e-3).sum()))
                if top_k > 0:
                    top_idx  = np.argsort(w_abs)[::-1][:top_k]
                    top_feats = ", ".join(
                        f"{names[i]}({w_abs[i]:.3f})" for i in top_idx
                    )
            elif h.hyp_type == "product":
                def _feat_str(hh):
                    if hh.hyp_type in ("linear", "square", "abs") and hh.w is not None:
                        w_abs = np.abs(hh.w)
                        idx = int(w_abs.argmax())
                        return f"{names[idx]}({w_abs[idx]:.3f})"
                    return "?"
                top_feats = f"[{_feat_str(h.h1)}] × [{_feat_str(h.h2)}]"

            rows.append({
                "type":         h.hyp_type,
                "fitness":      round(h.fitness, 6),
                "use_count":    h.use_count,
                "complexity":   h.complexity(),
                "top_features": top_feats,
            })

        return pd.DataFrame(rows).sort_values("fitness", ascending=False).reset_index(drop=True)

    # ── pool introspection ────────────────────────────────────────────────────

    def get_hypothesis_pool(self) -> list[Hypothesis]:
        """Return the list of Hypothesis objects retained after training."""
        check_is_fitted(self, "_pool_hypotheses_")
        return self._pool_hypotheses_

    def get_n_trees(self) -> int:
        """Return the number of trees actually fitted (may be < n_estimators with early stopping)."""
        check_is_fitted(self, "trees_")
        return len(self.trees_)

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the fitted model to disk.

        Uses joblib internally; the C++ tree structures are serialized to numpy
        arrays so the file is fully portable across processes and machines
        (same OS + Python version required).

        Parameters
        ----------
        path : str
            File path, e.g. ``"model.joblib"`` or ``"checkpoints/round50.pkl"``.

        Examples
        --------
        >>> clf.save("hypforge_model.joblib")
        >>> clf2 = HypForgeClassifier.load("hypforge_model.joblib")
        """
        import joblib
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "HypForgeClassifier":
        """
        Load a model saved with :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the saved file.

        Examples
        --------
        >>> clf = HypForgeClassifier.load("hypforge_model.joblib")
        >>> clf.predict(X_test)
        """
        import joblib
        return joblib.load(path)

    # ── internal ─────────────────────────────────────────────────────────────

    def _resolve_D_num(self, D: int) -> int:
        """Number of numerical (non-categorical) columns."""
        if not self.cat_features:
            return D
        cat_idx = set()
        for cf in self.cat_features:
            if isinstance(cf, int):
                cat_idx.add(cf)
            elif self.feature_names_in_ is not None and cf in self.feature_names_in_:
                cat_idx.add(self.feature_names_in_.index(cf))
        return D - len(cat_idx)

    def _fit_core(self, X, y, X_val, y_val, D_num):
        N, D = X.shape
        K    = int(y.max()) + 1
        seed = self.random_state if self.random_state is not None else 42

        cnt  = np.bincount(y, minlength=K).astype(np.float32)
        cw   = N / (K * cnt.clip(min=1))
        sw   = cw[y]

        lp   = np.log(cnt / N + 1e-8)
        lp  -= lp.mean()
        self.F_init_ = lp.tolist()

        Fsc = np.tile(lp, (N, 1))

        F_val = None
        if X_val is not None:
            F_val = np.tile(lp, (X_val.shape[0], 1))

        pool = HypForgePool(
            D, max_size=self.pool_size,
            op_mode="all",
            crossover_top_k=self.crossover_top_k,
            elitism_k=self.elitism_k,
            alps_mode=True,
            map_elites_slots=self.map_elites_slots,
        )

        rng             = np.random.default_rng(seed)
        best_val_loss   = float("inf")
        best_trees      = []
        best_snaps      = []   # pool snapshots paired with best_trees
        best_pool_snap  = []   # final pool at the best round (for _pool_hypotheses_)
        no_improv       = 0
        self.trees_     = []
        self.pool_snaps_ = []  # per-tree pool snapshots (same order as trees_)

        for m in range(self.n_estimators):
            Fsh = Fsc - Fsc.max(axis=1, keepdims=True)
            Pm  = np.exp(Fsh); Pm /= Pm.sum(axis=1, keepdims=True)

            oh  = np.zeros((N, K), dtype=np.float32)
            oh[np.arange(N), y] = 1.0
            G_w = sw[:, None] * (Pm - oh)
            H_w = sw[:, None] * Pm * (1.0 - Pm)

            # ── build tree: evolve + Z assembly + tree + predict all in C++ ────
            do_evolve = (m % self.evolve_every == 0)
            evolve_sub = rng.choice(N, size=min(N, 10000), replace=False).astype(np.int32)
            if self.subsample < 1.0:
                sub_size = min(N, max(5000, int(N * self.subsample)))
                tree_sub = rng.choice(N, size=sub_size, replace=False).astype(np.int32)
            else:
                tree_sub = np.arange(N, dtype=np.int32)

            tree, pred_np = pool.build_tree(
                X, G_w, H_w,
                evolve_sub, tree_sub,
                D_num, self.max_depth,
                reg_lambda=self.reg_lambda,
                do_evolve=do_evolve,
            )
            Fsc  = Fsc + self.learning_rate * pred_np
            self.trees_.append(tree)
            snap = pool.pop                        # single export per round
            self.pool_snaps_.append(snap)
            # use_count update is performed inside pool.build_tree

            # ── validation & early stopping ──────────────────────────────────
            val_str = ""
            if X_val is not None:
                Z_val_np    = pool.eval(X_val)
                pred_val_np = tree.predict(Z_val_np)
                F_val       = F_val + self.learning_rate * pred_val_np

                Fv_sh    = F_val - F_val.max(axis=1, keepdims=True)
                P_val    = np.exp(Fv_sh); P_val /= P_val.sum(axis=1, keepdims=True)
                val_loss = -np.log(P_val[np.arange(len(y_val)), y_val].clip(1e-8)).mean()
                val_acc  = (P_val.argmax(axis=1) == y_val).mean()
                val_str  = f" | ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss  = val_loss
                    no_improv      = 0
                    best_trees     = list(self.trees_)
                    best_snaps     = list(self.pool_snaps_)
                    best_pool_snap = pool.pop
                else:
                    no_improv += 1

            if self.verbose:
                ll  = -np.log(Pm[np.arange(N), y].clip(1e-8)).mean()
                acc = (Pm.argmax(axis=1) == y).mean()
                best_ucb = snap[0].score if snap else 0.0
                best_mu  = snap[0].mu_fitness if snap else 0.0
                print(
                    f"  [HypForge] Round {m+1:3d} | Loss={ll:.4f} | Acc={acc:.4f} | "
                    f"UCB={best_ucb:.4f} | μ={best_mu:.4f} | Pop={len(snap)}{val_str}"
                )

            if X_val is not None and self.early_stopping_rounds is not None:
                if no_improv >= self.early_stopping_rounds:
                    if self.verbose:
                        print(f"  [HypForge] Early stopping at round {m+1} (best ValLoss={best_val_loss:.4f})")
                    self.trees_       = best_trees
                    self.pool_snaps_  = best_snaps
                    pool.pop          = best_pool_snap
                    break

        self._pool_hypotheses_ = best_pool_snap if best_pool_snap else pool.pop
        for h in self._pool_hypotheses_:
            h.clear_runtime_caches()