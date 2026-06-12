"""
THEORY.md §2 — prediction P4: oblique OBLIVIOUS trees.

If A/B's real contribution is capacity control (correlated split directions →
weaker, lower-variance trees), then making the regularization explicit should
work at least as well: one shared split direction per LEVEL (CatBoost-style
oblivious structure, but oblique), per-node thresholds.

Bonus if confirmed: routing is one dot product per level — branchless,
GEMM-vectorizable across samples, directly serving the CPU-utilization goal.

Compares ensembles of standard best-first research trees vs oblivious oblique
trees at matched leaf budgets.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from typing import Optional

from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from oqboost_research import (
    OQBoostResearch, _sis_scores, _dominant_class, _best_threshold,
)


class ObliviousObliqueTree:
    """One shared oblique direction per level; per-node thresholds."""

    def __init__(self, depth=6, reg_lambda=1.0, n_random=16, d_sub=16,
                 seed=0):
        self.depth = depth
        self.reg_lambda = reg_lambda
        self.n_random = n_random
        self.d_sub = d_sub
        self.seed = seed
        self.level_w_: list[torch.Tensor] = []
        self.level_thr_: list[dict[int, float]] = []  # group id -> threshold
        self.leaf_vals_: Optional[torch.Tensor] = None
        self.K_ = 0

    def _candidates(self, X, G, H, k_dom):
        N, D = X.shape
        scores = _sis_scores(X, G, H, k_dom, self.reg_lambda)
        top = scores.topk(min(self.d_sub, D)).indices.tolist()
        cands = []
        for f in top:
            w = torch.zeros(D, dtype=X.dtype)
            w[f] = 1.0
            cands.append(w)
        p_nz = 1.0 / max(2.0, math.sqrt(D))
        for _ in range(self.n_random):
            mask = torch.rand(D) < p_nz
            if not mask.any():
                mask[torch.randint(D, (1,))] = True
            w = torch.zeros(D, dtype=X.dtype)
            signs = torch.where(torch.rand(int(mask.sum())) < 0.5, -1.0, 1.0)
            w[mask] = signs.to(X.dtype)
            cands.append(w / w.norm())
        return cands

    def fit_predict(self, X, G, H):
        torch.manual_seed(self.seed)
        N, D = X.shape
        K = G.shape[1]
        self.K_ = K
        groups = torch.zeros(N, dtype=torch.long)  # group id per sample
        self.level_w_, self.level_thr_ = [], []

        for level in range(self.depth):
            k_dom = _dominant_class(G)
            cands = self._candidates(X, G, H, k_dom)

            best_total = 0.0
            best_w = None
            best_thr: dict[int, float] = {}
            gids = torch.unique(groups).tolist()
            for w in cands:
                proj = X @ w
                total = 0.0
                thr_map = {}
                for gid in gids:
                    m = groups == gid
                    if int(m.sum()) < 8:
                        continue
                    t, g = _best_threshold(proj[m], G[m], H[m], self.reg_lambda)
                    if g > 0:
                        total += g
                        thr_map[gid] = t
                if total > best_total:
                    best_total = total
                    best_w = w
                    best_thr = thr_map
            if best_w is None:
                break
            self.level_w_.append(best_w)
            self.level_thr_.append(best_thr)

            proj = X @ best_w
            new_groups = groups.clone()
            for gid in gids:
                m = groups == gid
                if gid in best_thr:
                    go_right = m & (proj > best_thr[gid])
                    new_groups[m] = gid * 2          # left
                    new_groups[go_right] = gid * 2 + 1
                else:
                    new_groups[m] = gid * 2          # unsplit → follow left
            groups = new_groups

        # Newton leaf values per group
        max_gid = int(groups.max()) + 1
        leaf = torch.zeros(max_gid, K, dtype=X.dtype)
        for gid in torch.unique(groups).tolist():
            m = groups == gid
            leaf[gid] = -G[m].sum(0) / (H[m].sum(0) + self.reg_lambda)
        self.leaf_vals_ = leaf
        return leaf[groups]

    def predict(self, X):
        N = len(X)
        groups = torch.zeros(N, dtype=torch.long)
        for w, thr_map in zip(self.level_w_, self.level_thr_):
            proj = X @ w
            new_groups = groups * 2
            for gid, t in thr_map.items():
                m = (groups == gid) & (proj > t)
                new_groups[m] += 1
            groups = new_groups
        out = torch.zeros(N, self.K_, dtype=X.dtype)
        valid = groups < len(self.leaf_vals_)
        out[valid] = self.leaf_vals_[groups[valid]]
        return out


class ObliviousBooster:
    def __init__(self, n_estimators=60, learning_rate=0.1, depth=4,
                 reg_lambda=1.0, n_random=16, random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.reg_lambda = reg_lambda
        self.n_random = n_random
        self.random_state = random_state
        self.trees_: list[ObliviousObliqueTree] = []
        self.F_init_ = None

    def fit(self, X, y):
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        y_t = torch.tensor(np.asarray(y, dtype=np.int64))
        N = len(X_t)
        K = int(y_t.max()) + 1
        cnt = torch.bincount(y_t, minlength=K).float()
        lp = torch.log(cnt / N + 1e-8)
        lp = lp - lp.mean()
        self.F_init_ = lp
        F = lp.unsqueeze(0).expand(N, -1).clone()
        oh = torch.zeros(N, K)
        oh[torch.arange(N), y_t] = 1.0
        self.trees_ = []
        for m in range(self.n_estimators):
            Fs = F - F.max(1, keepdim=True).values
            P = Fs.exp()
            P = P / P.sum(1, keepdim=True)
            G = P - oh
            H = P * (1 - P)
            tree = ObliviousObliqueTree(
                depth=self.depth, reg_lambda=self.reg_lambda,
                n_random=self.n_random, seed=self.random_state + m)
            pred = tree.fit_predict(X_t, G, H)
            F = F + self.learning_rate * pred
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        N = len(X_t)
        F = self.F_init_.unsqueeze(0).expand(N, -1).clone()
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X_t)
        Fs = F - F.max(1, keepdim=True).values
        P = Fs.exp()
        return (P / P.sum(1, keepdim=True)).numpy()


def run(name, X, y, n_seeds=3):
    print(f"\n=== {name} ===")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    configs = [
        ("standard d4 (16 leaves)", "std", dict(max_depth=4)),
        ("oblivious d4 (16 leaves)", "obl", dict(depth=4)),
        ("oblivious d6 (64 leaves)", "obl", dict(depth=6)),
    ]
    for cname, kind, kw in configs:
        accs, lls = [], []
        for seed in range(n_seeds):
            if kind == "std":
                clf = OQBoostResearch(
                    n_estimators=60, learning_rate=0.1, use_wls=False,
                    n_random=12, random_state=seed, device='cpu', **kw)
            else:
                clf = ObliviousBooster(
                    n_estimators=60, learning_rate=0.1, n_random=16,
                    random_state=seed, **kw)
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_te)
            accs.append(accuracy_score(y_te, p.argmax(1)))
            lls.append(log_loss(y_te, p))
        print(f"  {cname:26s} acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
              f"logloss={np.mean(lls):.4f}±{np.std(lls):.4f}", flush=True)


def main():
    X, y = make_classification(
        n_samples=6000, n_features=30, n_informative=12, n_redundant=10,
        n_classes=2, random_state=7)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    run("synthetic correlated 6k x 30", X, y)

    Xm, ym = make_classification(
        n_samples=5000, n_features=20, n_informative=10, n_classes=4,
        n_clusters_per_class=2, random_state=11)
    Xm = StandardScaler().fit_transform(Xm).astype(np.float32)
    run("multiclass 5k x 20 K=4", Xm, ym)

    d = load_breast_cancer()
    Xb = StandardScaler().fit_transform(d.data).astype(np.float32)
    run("breast_cancer", Xb, d.target)


if __name__ == "__main__":
    main()
