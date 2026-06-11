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
        family_max_size: int = 30,
        meta_evolution: bool = True,
        family_lambda: float = 0.1,  # Credit lambda: weight of ancestor credit in UCB scoring
        breeding_beta: float = 0.3,  # Credit gamma: discount decay factor for ancestor credit propagation
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
        self.family_max_size       = family_max_size
        self.meta_evolution        = meta_evolution
        self.family_lambda         = family_lambda
        self.breeding_beta         = breeding_beta

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
                "type":            h.hyp_type,
                "fitness":         round(h.fitness, 6),
                "use_count":       h.use_count,
                "complexity":      h.complexity(),
                "top_features":    top_feats,
                "family_id":       h.family_id,
                "birth_round":     h.birth_round,
                "parent1":         h.parent1,
                "parent2":         h.parent2,
                "family_fitness":  round(h.family_fitness, 6),
                "breeding_value":  round(h.breeding_value, 6),
                "ancestor_credit": round(h.ancestor_credit, 6),
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

    def _resolve_cat_idx(self, D: int) -> list[int]:
        """Sorted column indices declared categorical via ``cat_features``."""
        if not self.cat_features:
            return []
        cat_idx = set()
        for cf in self.cat_features:
            if isinstance(cf, (int, np.integer)):
                cat_idx.add(int(cf))
            elif self.feature_names_in_ is not None and cf in self.feature_names_in_:
                cat_idx.add(self.feature_names_in_.index(cf))
        return sorted(cat_idx)

    def _resolve_D_num(self, D: int) -> int:
        """Number of numerical (non-categorical) columns."""
        return D - len(self._resolve_cat_idx(D))

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
                current_round=m,
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
                print(
                    f"  [HypForge] Round {m+1:3d} | Loss={ll:.4f} | Acc={acc:.4f} | "
                    f"Pop={len(snap)}{val_str}"
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


# ── SALOTClassifier ──────────────────────────────────────────────────────────

class SALOTClassifier(HypForgeClassifier):
    """
    SALOT v9: deterministic pool-free oblique tree boosting.

    Two effective hyperparameters per tree (max_depth, reg_lambda).
    Native missing-value handling (numeric NaN → mean imputation baked into
    the binning context) and native categorical handling (per-round
    gradient-rank target encoding; categories participate in both axis
    scans and oblique projections).  Columns named in ``cat_features`` are
    internally moved after the numeric block; category values must be
    numeric IDs (floats holding integers).  NaN is allowed anywhere.

    Legacy v6 knobs (prune_strength, n_wls_max, d_sub_max, gbaor_alpha,
    n_candidates, honest_split, quant_levels, colsample_bytree) are accepted
    but ignored.
    """

    def __init__(
        self,
        n_estimators:          int   = 500,
        learning_rate:         float = 0.05,
        max_depth:             int   = 4,
        reg_lambda:            float = 1.0,
        pool_size:             int   = 0,     # unused, kept for API compat
        subsample:             float = 0.8,
        colsample_bytree:      float = 1.0,
        early_stopping_rounds: int | None = 50,
        device:                str   = "auto",
        n_jobs:                int   = -1,
        random_state:          int | None = None,
        verbose:               bool  = False,
        cat_features:          list | None = None,
        class_weight:          str | None = "balanced",
        prune_strength:        float = 0.1,
        n_wls_max:             int   = 512,
        d_sub_max:             int   = 32,
        gbaor_alpha:           float = 0.05,
        n_candidates:          int   = 3,
        honest_split:          bool  = False,
        quant_levels:          int   = 0,
    ):
        super().__init__(
            n_estimators          = n_estimators,
            learning_rate         = learning_rate,
            max_depth             = max_depth,
            reg_lambda            = reg_lambda,
            pool_size             = 0,
            subsample             = subsample,
            early_stopping_rounds = early_stopping_rounds,
            device                = device,
            random_state          = random_state,
            verbose               = verbose,
            cat_features          = cat_features,
        )
        self.colsample_bytree = colsample_bytree
        self.n_jobs           = n_jobs
        self.class_weight     = class_weight
        self.prune_strength   = prune_strength
        self.n_wls_max        = n_wls_max
        self.d_sub_max        = d_sub_max
        self.gbaor_alpha      = gbaor_alpha
        self.n_candidates     = n_candidates
        self.honest_split     = honest_split
        self.quant_levels     = quant_levels

    def _fit_core(self, X, y, X_val, y_val, D_num):
        from ._salot import SalotContext

        N, D = X.shape
        K    = int(y.max()) + 1
        seed = self.random_state if self.random_state is not None else 42

        # Move categorical columns after the numeric block (the C++ core
        # uses the [D_num, D) convention); remember the permutation for
        # prediction time.
        cat_idx = self._resolve_cat_idx(D)
        if cat_idx and cat_idx != list(range(D_num, D)):
            perm = [i for i in range(D) if i not in set(cat_idx)] + cat_idx
            self._col_perm_ = np.asarray(perm, dtype=np.intp)
        else:
            self._col_perm_ = None
        if self._col_perm_ is not None:
            X = np.ascontiguousarray(X[:, self._col_perm_])
            if X_val is not None:
                X_val = np.ascontiguousarray(X_val[:, self._col_perm_])

        cnt = np.bincount(y, minlength=K).astype(np.float32)
        if getattr(self, "class_weight", "balanced") == "balanced":
            sw = (N / (K * cnt.clip(min=1)))[y].astype(np.float32)
        else:
            sw = np.ones(N, dtype=np.float32)
        sw_col = sw[:, None]

        lp  = np.log(cnt / N + 1e-8).astype(np.float32); lp -= lp.mean()
        self.F_init_ = lp.tolist()

        Fsc   = np.tile(lp, (N, 1))
        F_val = np.tile(lp, (X_val.shape[0], 1)) if X_val is not None else None

        # One-hot targets are round-invariant — build once.
        oh = np.zeros((N, K), dtype=np.float32)
        oh[np.arange(N), y] = 1.0

        rng = np.random.default_rng(seed)

        best_val_loss = float("inf")
        best_trees:   list = []
        no_improv = 0

        self.trees_: list = []   # list of SALOTTree

        # Bin once, boost many: the context owns the imputed/binned copy of
        # X, so per-round work is gradients + categorical re-ranks + tree.
        ctx = SalotContext(X, D_num=D_num)
        Pm  = np.empty((N, K), dtype=np.float32)
        G_w = np.empty((N, K), dtype=np.float32)
        H_w = np.empty((N, K), dtype=np.float32)
        full_idx = np.arange(N, dtype=np.int32)
        try:
            for m in range(self.n_estimators):
                np.subtract(Fsc, Fsc.max(axis=1, keepdims=True), out=Pm)
                np.exp(Pm, out=Pm)
                Pm /= Pm.sum(axis=1, keepdims=True)

                np.subtract(Pm, oh, out=G_w)
                G_w *= sw_col
                np.subtract(1.0, Pm, out=H_w)
                H_w *= Pm
                H_w *= sw_col

                if self.subsample < 1.0:
                    # Bernoulli row sampling: indices come out sorted (cache-
                    # friendly C++ row access) with no explicit sort.
                    tree_sub = np.flatnonzero(
                        rng.random(N) < self.subsample
                    ).astype(np.int32)
                    if len(tree_sub) < min(N, 1000):
                        tree_sub = full_idx
                else:
                    tree_sub = full_idx

                t, out_pred = ctx.build(
                    G_w, H_w, tree_sub, self.max_depth, self.reg_lambda
                )
                self.trees_.append(t)
                Fsc += self.learning_rate * out_pred

                val_str = ""
                if X_val is not None:
                    pred_val = t.predict(X_val)
                    F_val    = F_val + self.learning_rate * pred_val
                    Fv_sh    = F_val - F_val.max(axis=1, keepdims=True)
                    P_val    = np.exp(Fv_sh); P_val /= P_val.sum(axis=1, keepdims=True)
                    val_loss = -np.log(P_val[np.arange(len(y_val)), y_val].clip(1e-8)).mean()
                    val_acc  = (P_val.argmax(axis=1) == y_val).mean()
                    val_str  = f" | ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f}"

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improv     = 0
                        best_trees    = list(self.trees_)
                    else:
                        no_improv += 1

                if self.verbose:
                    ll  = -np.log(Pm[np.arange(N), y].clip(1e-8)).mean()
                    acc = (Pm.argmax(axis=1) == y).mean()
                    print(
                        f"  [SALOTClassifier] Round {m+1:3d} | Loss={ll:.4f} | "
                        f"Acc={acc:.4f}{val_str}"
                    )

                if X_val is not None and self.early_stopping_rounds is not None:
                    if no_improv >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"  [SALOTClassifier] Early stopping at round {m+1}")
                        self.trees_ = best_trees
                        break
        finally:
            ctx.close()

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values
        X = np.ascontiguousarray(X, dtype=np.float32)
        if getattr(self, "_col_perm_", None) is not None:
            X = np.ascontiguousarray(X[:, self._col_perm_])
        N = X.shape[0]

        F = np.tile(np.array(self.F_init_, dtype=np.float32), (N, 1))
        for t in self.trees_:
            F += self.learning_rate * t.predict(X)

        Fsh = F - F.max(axis=1, keepdims=True)
        P   = np.exp(Fsh); P /= P.sum(axis=1, keepdims=True)
        return P

    def __del__(self):
        pass


TestForgeClassifier = SALOTClassifier


# ── GOSClassifier ─────────────────────────────────────────────────────────────

class GOSClassifier:
    """
    d-GOS (Differentiable Gradient Ordering Search) boosted classifier.

    Each round: evolve B particles via differentiable H-functional gradient,
    then build a BFSTree using the evolved particle projections as split axes.
    """

    def __init__(
        self,
        n_estimators:          int   = 300,
        B:                     int   = 32,
        M:                     int   = 32,
        tau:                   float = 1.0,
        gos_steps:             int   = 10,
        gos_eta:               float = 0.05,
        gos_lam:               float = 0.01,
        gos_gamma:             float = 0.1,
        max_depth:             int   = 8,
        learning_rate:         float = 0.1,
        subsample:             float = 0.9,
        reg_lambda:            float = 1.0,
        early_stopping_rounds: int | None = 50,
        random_state:          int | None = 42,
    ):
        self.n_estimators          = n_estimators
        self.B                     = B
        self.M                     = M
        self.tau                   = tau
        self.gos_steps             = gos_steps
        self.gos_eta               = gos_eta
        self.gos_lam               = gos_lam
        self.gos_gamma             = gos_gamma
        self.max_depth             = max_depth
        self.learning_rate         = learning_rate
        self.subsample             = subsample
        self.reg_lambda            = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state          = random_state

        self._tree_ptrs_:   list = []
        self._pool_snaps_:  list = []
        self.F_init_:       list = []
        self.classes_       = None

    def fit(self, X, y, eval_set=None, verbose: bool = False):
        from ._gos import GOSPool

        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        N, D = X.shape
        K    = int(y.max()) + 1
        self.classes_ = np.arange(K)
        seed = self.random_state if self.random_state is not None else 42

        cnt = np.bincount(y, minlength=K).astype(np.float32)
        cw  = N / (K * cnt.clip(min=1))
        sw  = cw[y]

        lp  = np.log(cnt / N + 1e-8); lp -= lp.mean()
        self.F_init_ = lp.tolist()

        Fsc   = np.tile(lp, (N, 1)).astype(np.float32)
        F_val = None
        X_val, y_val = None, None
        if eval_set:
            X_val, y_val = eval_set[0]
            if hasattr(X_val, "values"):
                X_val = X_val.values
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)
            F_val = np.tile(lp, (len(X_val), 1)).astype(np.float32)

        rng = np.random.default_rng(seed)
        pool = GOSPool(D, B=self.B, M=self.M, tau=self.tau,
                       n_steps=self.gos_steps, eta=self.gos_eta,
                       lam=self.gos_lam, gamma=self.gos_gamma, seed=seed)

        best_val_loss = float("inf")
        best_ptrs:  list = []
        best_snaps: list = []
        no_improv = 0

        self._tree_ptrs_  = []
        self._pool_snaps_ = []

        for m in range(self.n_estimators):
            Fsh = Fsc - Fsc.max(axis=1, keepdims=True)
            Pm  = np.exp(Fsh); Pm /= Pm.sum(axis=1, keepdims=True)

            oh  = np.zeros((N, K), dtype=np.float32)
            oh[np.arange(N), y] = 1.0
            G_w = (sw[:, None] * (Pm - oh)).astype(np.float32)
            H_w = (sw[:, None] * Pm * (1.0 - Pm)).astype(np.float32)

            if self.subsample < 1.0:
                ns  = max(1, int(N * self.subsample))
                sub = rng.choice(N, ns, replace=False).astype(np.int32)
            else:
                sub = np.arange(N, dtype=np.int32)

            pool.evolve(X, G_w, H_w, sub=sub, round_num=m)

            out_pred = np.zeros((N, K), dtype=np.float32)
            tree_ptr = pool.build_tree(X, G_w, H_w, sub, self.max_depth,
                                       self.reg_lambda, out_pred)
            if not tree_ptr:
                continue

            self._tree_ptrs_.append(tree_ptr)
            self._pool_snaps_.append(pool.snapshot())
            Fsc += self.learning_rate * out_pred

            val_str = ""
            if X_val is not None:
                pred_val = np.zeros((len(X_val), K), dtype=np.float32)
                pool.predict_tree(tree_ptr, X_val, pred_val)
                F_val = F_val + self.learning_rate * pred_val
                Fv_sh = F_val - F_val.max(axis=1, keepdims=True)
                P_val = np.exp(Fv_sh); P_val /= P_val.sum(axis=1, keepdims=True)
                val_loss = -np.log(P_val[np.arange(len(y_val)), y_val].clip(1e-8)).mean()
                val_acc  = (P_val.argmax(axis=1) == y_val).mean()
                val_str  = f" | ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improv     = 0
                    best_ptrs     = list(self._tree_ptrs_)
                    best_snaps    = list(self._pool_snaps_)
                else:
                    no_improv += 1

            if verbose and ((m + 1) % 50 == 0 or m == 0):
                ll  = -np.log(Pm[np.arange(N), y].clip(1e-8)).mean()
                acc = (Pm.argmax(axis=1) == y).mean()
                print(f"  [GOS] Round {m+1:3d} | Loss={ll:.4f} | Acc={acc:.4f}{val_str}")

            if X_val is not None and self.early_stopping_rounds is not None:
                if no_improv >= self.early_stopping_rounds:
                    self._tree_ptrs_  = best_ptrs
                    self._pool_snaps_ = best_snaps
                    break

        return self

    def predict_proba(self, X):
        from ._gos import GOSPool

        if hasattr(X, "values"):
            X = X.values
        X  = np.ascontiguousarray(X, dtype=np.float32)
        N  = X.shape[0]
        K  = len(self.classes_)

        F = np.tile(np.array(self.F_init_, dtype=np.float32), (N, 1))
        for ptr, W_snap in zip(self._tree_ptrs_, self._pool_snaps_):
            B  = W_snap.shape[0]
            D  = W_snap.shape[1]
            # Reconstruct a minimal pool snapshot for prediction
            pool_snap = GOSPool.__new__(GOSPool)
            pool_snap.D = D; pool_snap.B = B; pool_snap.M = self.M
            pool_snap.tau = self.tau; pool_snap.W = W_snap
            pred = np.zeros((N, K), dtype=np.float32)
            pool_snap.predict_tree(ptr, X, pred)
            F += self.learning_rate * pred

        Fsh = F - F.max(axis=1, keepdims=True)
        P   = np.exp(Fsh); P /= P.sum(axis=1, keepdims=True)
        return P

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)