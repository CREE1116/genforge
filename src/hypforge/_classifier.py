from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._pool import Hypothesis, HypForgePool, _split_gain_fitness
from ._tree import BFSTree


def _detect_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


class HypForgeClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    HypForge: Hypothesis Pool Evolution for Oblique GBDT.

    Gradient-boosted oblique decision trees where split directions are drawn
    from an evolving pool of projections (linear, square, abs, product
    combinations of gradient-aligned, SVD, and synergy hypotheses).

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
        max_depth: int       = 6,
        reg_lambda: float    = 1.0,
        pool_size: int       = 500,
        evolve_every: int    = 1,
        subsample: float     = 0.8,
        early_stopping_rounds: int | None = 50,
        device: str          = "auto",
        random_state: int | None = None,
        verbose: bool        = False,
        cat_features: list | None = None,
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

        dev   = _detect_device(self.device)
        X_t   = torch.tensor(X, dtype=torch.float32, device=dev)
        lp    = np.array(self.F_init_, dtype=np.float32)
        Fsc   = np.tile(lp, (X.shape[0], 1))

        Z_pred = np.ascontiguousarray(
            np.stack([h.eval(X_t).cpu().numpy() for h in self._pool_hypotheses_])
        )

        for tree in self.trees_:
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
            if h.hyp_type in ("linear", "square", "abs") and h.w is not None:
                w_cpu = h.w.cpu().numpy().astype(np.float64)
                imp += np.abs(w_cpu) * h.use_count
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
            if h.hyp_type in ("linear", "square", "abs") and h.w is not None:
                mask |= (np.abs(h.w.cpu().numpy()) > threshold)
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

        dev = _detect_device(self.device)
        X_t = torch.tensor(X, dtype=torch.float32, device=dev)

        hyps = self._pool_hypotheses_
        if top_k is not None:
            hyps = sorted(hyps, key=lambda h: h.fitness, reverse=True)[:top_k]

        return np.stack([h.eval(X_t).cpu().numpy() for h in hyps], axis=1).astype(np.float32)

    def get_hypothesis_summary(self) -> "pd.DataFrame":
        """
        Return a DataFrame summarising every hypothesis in the final pool.

        Columns
        -------
        type        : 'linear' | 'square' | 'abs' | 'product'
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
            if h.hyp_type in ("linear", "square", "abs") and h.w is not None:
                w_abs = np.abs(h.w.cpu().numpy())
                top_k = min(5, int((w_abs > 1e-3).sum()))
                if top_k > 0:
                    top_idx  = np.argsort(w_abs)[::-1][:top_k]
                    top_feats = ", ".join(
                        f"{names[i]}({w_abs[i]:.3f})" for i in top_idx
                    )
            elif h.hyp_type == "product":
                def _feat_str(hh):
                    if hh.hyp_type in ("linear", "square", "abs") and hh.w is not None:
                        w_abs = np.abs(hh.w.cpu().numpy())
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
        dev  = _detect_device(self.device)
        N, D = X.shape
        K    = int(y.max()) + 1
        seed = self.random_state if self.random_state is not None else 42

        cnt  = np.bincount(y, minlength=K).astype(np.float32)
        cw   = N / (K * cnt.clip(min=1))
        sw   = torch.tensor(cw[y], dtype=torch.float32, device=dev)

        lp   = np.log(cnt / N + 1e-8)
        lp  -= lp.mean()
        self.F_init_ = lp.tolist()

        X_t  = torch.tensor(X, dtype=torch.float32, device=dev)
        y_t  = torch.tensor(y, dtype=torch.long,    device=dev)

        X_val_t = y_val_t = F_val = None
        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=dev)
            y_val_t = torch.tensor(y_val, dtype=torch.long,    device=dev)
            F_val   = torch.tensor(np.tile(lp, (X_val_t.shape[0], 1)), dtype=torch.float32, device=dev)

        Fsc  = torch.tensor(np.tile(lp, (N, 1)), dtype=torch.float32, device=dev)
        pool = HypForgePool(D, max_size=self.pool_size, dev=str(dev))
        for h in pool.pop:
            h.initialize_cache(X_t)

        rng             = torch.Generator(device="cpu").manual_seed(seed)
        best_val_loss   = float("inf")
        best_trees      = []
        no_improv       = 0
        self.trees_     = []

        for m in range(self.n_estimators):
            for h in pool.pop:
                h.rounds_since_last_use += 1

            Fsh = Fsc - Fsc.max(1, keepdim=True).values
            Pm  = Fsh.exp(); Pm /= Pm.sum(1, keepdim=True)

            oh  = torch.zeros(N, K, device=dev).scatter_(1, y_t.unsqueeze(1), 1.0)
            G_w = sw.unsqueeze(1) * (Pm - oh)
            H_w = sw.unsqueeze(1) * Pm * (1.0 - Pm)

            if m % self.evolve_every == 0:
                sub = torch.randperm(N, generator=rng)[:min(N, 10000)].to(dev)
                pool.evolve(X_t, G_w, H_w, sub, D_num, self.reg_lambda)

            # ── build tree (C++) ─────────────────────────────────────────────
            Z_full_np     = np.stack([h.full_cache_cpu for h in pool.pop])        # [P, N]
            thresholds_np = np.stack([h.thresholds_cpu for h in pool.pop], axis=1) # [9, P]

            if self.subsample < 1.0:
                sub_size     = min(N, max(5000, int(N * self.subsample)))
                tree_sub     = torch.randperm(N, generator=rng)[:sub_size]
                tree_sub_cpu = tree_sub.numpy()
                Z_tree_np    = np.ascontiguousarray(Z_full_np[:, tree_sub_cpu])
                G_tree_np    = G_w[tree_sub.to(dev)].cpu().numpy()
                H_tree_np    = H_w[tree_sub.to(dev)].cpu().numpy()
            else:
                Z_tree_np = Z_full_np
                G_tree_np = G_w.cpu().numpy()
                H_tree_np = H_w.cpu().numpy()

            tree = BFSTree()
            tree.build(
                Z=Z_tree_np, thresholds=thresholds_np,
                G=G_tree_np, H=H_tree_np,
                max_depth=self.max_depth, reg_lambda=self.reg_lambda,
            )

            pred_np = tree.predict(Z_full_np)
            Fsc     = Fsc + self.learning_rate * torch.tensor(pred_np, dtype=torch.float32, device=dev)
            self.trees_.append(tree)

            # ── track hypothesis usage for feature importance ─────────────────
            # (done inside pool.evolve via fitness; use_count updated here)

            # ── validation & early stopping ──────────────────────────────────
            val_str = ""
            if X_val_t is not None:
                Z_val_np    = np.ascontiguousarray(np.stack([h.eval(X_val_t).cpu().numpy() for h in pool.pop]))
                pred_val_np = tree.predict(Z_val_np)
                F_val       = F_val + self.learning_rate * torch.tensor(pred_val_np, dtype=torch.float32, device=dev)

                Fv_sh    = F_val - F_val.max(1, keepdim=True).values
                P_val    = Fv_sh.exp(); P_val /= P_val.sum(1, keepdim=True)
                val_loss = -torch.log(P_val.gather(1, y_val_t.unsqueeze(1)).clamp(1e-8)).mean().item()
                val_acc  = (P_val.argmax(1) == y_val_t).float().mean().item()
                val_str  = f" | ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improv     = 0
                    best_trees    = list(self.trees_)
                else:
                    no_improv += 1

            if self.verbose:
                ll  = -torch.log(Pm.gather(1, y_t.unsqueeze(1)).clamp(1e-8)).mean()
                acc = (Pm.argmax(1) == y_t).float().mean()
                best_fit = pool.pop[0].fitness if pool.pop else 0.0
                print(
                    f"  [HypForge] Round {m+1:3d} | Loss={ll.item():.4f} | Acc={acc.item():.4f} | "
                    f"BestFit={best_fit:.4f} | Pop={len(pool.pop)}{val_str}"
                )

            if X_val_t is not None and self.early_stopping_rounds is not None:
                if no_improv >= self.early_stopping_rounds:
                    if self.verbose:
                        print(f"  [HypForge] Early stopping at round {m+1} (best ValLoss={best_val_loss:.4f})")
                    self.trees_ = best_trees
                    break

        self._pool_hypotheses_ = list(pool.pop)
        for h in self._pool_hypotheses_:
            h.clear_runtime_caches()