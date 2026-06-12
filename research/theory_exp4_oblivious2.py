"""
THEORY.md §2/P4 follow-up — making oblivious oblique trees stronger.

Round 1 (`theory_exp3_oblivious.py`) compared standard d4 vs oblivious d4/d6
— the d6 win had a leaf-count confound. This round:

  E4a  equal-depth: standard d6 vs oblivious d6, lr ∈ {0.3, 0.1}
       (oblivious should tolerate larger lr — weaker per-leaf structure)
  E4b  variants at d6:
         semi  — shared direction, per-node thresholds (current)
         full  — shared direction AND shared threshold (CatBoost-style;
                 routing = pure bit ops)
         semi+ — semi with doubled level candidate budget (the pooled
                 evaluation makes candidates cheap: one projection serves
                 all 2^l nodes, so the budget can grow where standard
                 trees cannot afford it)
"""
from __future__ import annotations

import numpy as np
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from oqboost_research import OQBoostResearch, _dominant_class, _best_threshold, _node_score
from theory_exp3_oblivious import ObliviousObliqueTree, ObliviousBooster


class ObliviousFullTree(ObliviousObliqueTree):
    """Direction AND threshold shared per level (fully oblivious)."""

    def fit_predict(self, X, G, H):
        torch.manual_seed(self.seed)
        N, D = X.shape
        K = G.shape[1]
        self.K_ = K
        groups = torch.zeros(N, dtype=torch.long)
        self.level_w_, self.level_thr_ = [], []

        for level in range(self.depth):
            k_dom = _dominant_class(G)
            cands = self._candidates(X, G, H, k_dom)
            gids = torch.unique(groups).tolist()

            best_total = 0.0
            best_w, best_t = None, 0.0
            for w in cands:
                proj = X @ w
                # shared threshold: scan pooled quantiles, score per group
                qs = torch.quantile(
                    proj, torch.linspace(0.05, 0.95, 19))
                for t in qs.tolist():
                    total = 0.0
                    for gid in gids:
                        m = groups == gid
                        if int(m.sum()) < 8:
                            continue
                        left = m & (proj <= t)
                        right = m & (proj > t)
                        if int(left.sum()) < 4 or int(right.sum()) < 4:
                            continue
                        g = (_node_score(G[left].sum(0), H[left].sum(0), self.reg_lambda)
                             + _node_score(G[right].sum(0), H[right].sum(0), self.reg_lambda)
                             - _node_score(G[m].sum(0), H[m].sum(0), self.reg_lambda))
                        if g > 0:
                            total += g
                    if total > best_total:
                        best_total = total
                        best_w, best_t = w, t
            if best_w is None:
                break
            self.level_w_.append(best_w)
            # same threshold for every group
            self.level_thr_.append({gid: best_t for gid in gids})

            proj = X @ best_w
            new_groups = groups * 2
            new_groups[proj > best_t] += 1
            groups = new_groups

        max_gid = int(groups.max()) + 1
        leaf = torch.zeros(max_gid, K, dtype=X.dtype)
        for gid in torch.unique(groups).tolist():
            m = groups == gid
            leaf[gid] = -G[m].sum(0) / (H[m].sum(0) + self.reg_lambda)
        self.leaf_vals_ = leaf
        return leaf[groups]


class FullBooster(ObliviousBooster):
    def _make_tree(self, m):
        return ObliviousFullTree(
            depth=self.depth, reg_lambda=self.reg_lambda,
            n_random=self.n_random, seed=self.random_state + m)

    def fit(self, X, y):  # same loop, different tree class
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
            tree = self._make_tree(m)
            pred = tree.fit_predict(X_t, G, H)
            F = F + self.learning_rate * pred
            self.trees_.append(tree)
        return self


def bench(clf_factory, X_tr, y_tr, X_te, y_te, n_seeds=2):
    accs, lls = [], []
    for seed in range(n_seeds):
        clf = clf_factory(seed)
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)
        accs.append(accuracy_score(y_te, p.argmax(1)))
        lls.append(log_loss(y_te, p))
    return np.mean(accs), np.std(accs), np.mean(lls), np.std(lls)


def run(name, X, y):
    print(f"\n=== {name} ===", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)

    rows = []
    for lr in (0.3, 0.1):
        rows.append((f"standard  d6 lr={lr}", lambda s, lr=lr: OQBoostResearch(
            n_estimators=60, learning_rate=lr, max_depth=6, use_wls=False,
            n_random=12, random_state=s, device='cpu')))
        rows.append((f"obliv-semi d6 lr={lr}", lambda s, lr=lr: ObliviousBooster(
            n_estimators=60, learning_rate=lr, depth=6, n_random=16,
            random_state=s)))
    rows.append(("obliv-full d6 lr=0.1", lambda s: FullBooster(
        n_estimators=60, learning_rate=0.1, depth=6, n_random=16,
        random_state=s)))
    rows.append(("obliv-semi+ d6 lr=0.1 (2x cand)", lambda s: ObliviousBooster(
        n_estimators=60, learning_rate=0.1, depth=6, n_random=32,
        random_state=s)))

    for cname, factory in rows:
        a, asd, l, lsd = bench(factory, X_tr, y_tr, X_te, y_te)
        print(f"  {cname:32s} acc={a:.4f}±{asd:.4f} logloss={l:.4f}±{lsd:.4f}",
              flush=True)


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


if __name__ == "__main__":
    main()
