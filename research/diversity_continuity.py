"""
Diversity + Continuity hypothesis test.

Hypothesis (user): the candidate generator works because it supplies BOTH
  - CONTINUITY: near-parent candidates (inherit A/B) — refine the parent's
    boundary, the one region random sampling can never reach, and
  - DIVERSITY: parent-independent random candidates — cover the rest of the
    direction space and decorrelate trees.
Removing either should hurt; neither alone should suffice.

EXP-A  Budget-matched knockouts. Every config gets axis candidates plus
       exactly 12 non-axis candidates:
         full        = 4 inherit + 8 random
         diversity   = 0 inherit + 12 random
         continuity  = 12 inherit + 0 random
         axis_only   = 0 + 0 (reference floor)
EXP-B  On the same fits: direction-diversity metrics
         - feature-usage entropy of winning directions
         - mean pairwise angle between winning dirs of DIFFERENT trees
         - mean pairwise correlation of per-tree prediction increments
       Prediction: diversity-only maximizes these but underuses geometry;
       continuity-only minimizes them (redundant trees).
EXP-D  Family win-share by boosting round quartile (full config) —
       when does continuity matter, when diversity?
"""
from __future__ import annotations

import math
import numpy as np
import torch
from collections import defaultdict

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from oqboost_research import OQBoostResearch

CONFIGS = [
    ("full      ", dict(n_inherit=4,  n_random=8)),
    ("diversity ", dict(inherited_rp_ratio=0.0, n_random=12)),
    ("continuity", dict(n_inherit=12, n_random=0)),
    ("axis_only ", dict(inherited_rp_ratio=0.0, n_random=0)),
]


def feature_entropy(clf) -> float:
    """Entropy (normalized to [0,1]) of |w| mass over features, all splits."""
    D = clf.trees_[0].D_
    mass = np.zeros(D)
    for r in clf.all_split_records():
        mass += r.w.abs().cpu().numpy()
    p = mass / (mass.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(D))


def cross_tree_angle(clf, max_pairs: int = 400) -> float:
    """Mean angle (deg) between winning dirs of different trees."""
    by_tree = defaultdict(list)
    for r in clf.all_split_records():
        by_tree[r.tree_idx].append(r.w)
    keys = sorted(by_tree)
    rng = np.random.default_rng(0)
    angles = []
    for _ in range(max_pairs):
        t1, t2 = rng.choice(keys, 2, replace=False)
        w1 = by_tree[t1][rng.integers(len(by_tree[t1]))]
        w2 = by_tree[t2][rng.integers(len(by_tree[t2]))]
        c = float(torch.clamp(
            (w1 * w2).sum() / (w1.norm() * w2.norm() + 1e-8), -1, 1))
        angles.append(math.degrees(math.acos(abs(c))))
    return float(np.mean(angles))


def tree_pred_correlation(clf, X: np.ndarray, n_trees: int = 30) -> float:
    """Mean pairwise Pearson corr of per-tree prediction increments (class 0)."""
    X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=clf.device)
    preds = []
    for tree in clf.trees_[:n_trees]:
        p = tree.predict(X_t)[:, 0].cpu().numpy()
        preds.append(p - p.mean())
    M = np.stack(preds)
    C = np.corrcoef(M)
    iu = np.triu_indices(len(preds), k=1)
    return float(np.nanmean(C[iu]))


def family_winshare_by_round(clf, n_buckets: int = 4):
    n_trees = len(clf.trees_)
    share = defaultdict(lambda: defaultdict(int))
    for r in clf.all_split_records():
        b = min(n_buckets - 1, r.tree_idx * n_buckets // n_trees)
        fam = ('inherit' if r.winner_type.startswith('inherit')
               else r.winner_type)
        share[b][fam] += 1
    return share


def run_dataset(name, X, y, n_seeds=3):
    print(f"\n{'='*64}\n EXP-A/B — {name}\n{'='*64}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)

    header = (f"  {'config':10s}  {'acc':>14s}  {'logloss':>14s}  "
              f"{'featH':>6s}  {'xAngle':>7s}  {'predCorr':>8s}")
    print(header)
    full_clf = None
    for cname, kw in CONFIGS:
        accs, lls, ents, angs, corrs = [], [], [], [], []
        for seed in range(n_seeds):
            clf = OQBoostResearch(
                n_estimators=60, learning_rate=0.1, max_depth=4,
                use_wls=False, random_state=seed, device='cpu', **kw,
            )
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_te)
            accs.append(accuracy_score(y_te, p.argmax(1)))
            lls.append(log_loss(y_te, p))
            ents.append(feature_entropy(clf))
            angs.append(cross_tree_angle(clf))
            corrs.append(tree_pred_correlation(clf, X_te))
            if cname.startswith("full") and seed == 0:
                full_clf = clf
        print(f"  {cname}  {np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"{np.mean(lls):.4f}±{np.std(lls):.4f}  "
              f"{np.mean(ents):6.3f}  {np.mean(angs):6.1f}°  "
              f"{np.mean(corrs):8.3f}", flush=True)

    if full_clf is not None:
        print("\n EXP-D — family win-share by round quartile (full config)")
        share = family_winshare_by_round(full_clf)
        fams = ['axis', 'inherit', 'random']
        print(f"  {'rounds':10s}" + "".join(f"{f:>10s}" for f in fams))
        n_trees = len(full_clf.trees_)
        for b in sorted(share):
            tot = sum(share[b].values())
            lo, hi = b * n_trees // 4, (b + 1) * n_trees // 4
            row = "".join(f"{share[b].get(f,0)/tot*100:9.1f}%" for f in fams)
            print(f"  {lo:3d}-{hi:3d}   {row}")


def main():
    X1, y1 = make_classification(
        n_samples=6000, n_features=30, n_informative=12, n_redundant=10,
        n_classes=2, random_state=7)
    X1 = StandardScaler().fit_transform(X1).astype(np.float32)
    run_dataset("synthetic correlated 6k x 30", X1, y1)

    X2, y2 = make_classification(
        n_samples=5000, n_features=20, n_informative=10, n_classes=4,
        n_clusters_per_class=2, random_state=11)
    X2 = StandardScaler().fit_transform(X2).astype(np.float32)
    run_dataset("multiclass 5k x 20 K=4", X2, y2)


if __name__ == "__main__":
    main()
