"""
Drift contradiction analysis.

The first analysis run showed mean parent→child direction drift of 74.9°
(median 84.4°) — near-orthogonal. The model's inheritance mechanism
(Strategy A/B: mutate the parent direction) assumes child directions stay
CLOSE to the parent. These two observations appear to contradict.

Hypotheses to test
------------------
H1 (measurement artifact): drift is measured on the WINNING direction
   regardless of which candidate type won. If axis/WLS candidates usually
   win, drift reflects "winner vs parent", not "inherited candidate vs
   parent". Inherited winners should show SMALL drift by construction.
H2 (depth confound): WLS gain dominance (94.6%) may be a root-split
   artifact — root nodes carry the largest n and gain. Check win rate and
   gain share BY DEPTH.
H3 (inheritance usefulness): if inherited candidates rarely win, does
   inheritance still help accuracy? Ablate in both the research impl and
   the production C++ engine (inherited_rp_ratio=0 vs 1).
"""
from __future__ import annotations

import math
import numpy as np
from collections import defaultdict

from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from oqboost_research import OQBoostResearch


def fit_research(X, y, **kw):
    clf = OQBoostResearch(
        n_estimators=60, learning_rate=0.1, max_depth=4, random_state=42, **kw
    )
    clf.fit(X, y)
    return clf


def exp1_drift_by_winner_type(clf):
    """H1: condition drift on which candidate type won."""
    print("\n[Exp 1] Drift by winner type (H1: measurement artifact)")
    by_type = defaultdict(list)
    counts = defaultdict(int)
    for r in clf.all_split_records():
        counts[r.winner_type] += 1
        if not math.isnan(r.angle_from_parent):
            by_type[r.winner_type].append(r.angle_from_parent)
    total = sum(counts.values())
    for t in ['axis', 'wls', 'inherit_A', 'inherit_B']:
        n = counts[t]
        drifts = by_type[t]
        d_str = (f"drift mean={np.mean(drifts):5.1f}° median={np.median(drifts):5.1f}°"
                 if drifts else "drift (root only)")
        print(f"  {t:10s} wins={n:4d} ({n/total*100:5.1f}%)  {d_str}")


def exp2_depth_profile(clf):
    """H2: WLS dominance by depth — root-split artifact?"""
    print("\n[Exp 2] Win rate and gain share by depth (H2: depth confound)")
    by_depth = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(lambda: defaultdict(int))
    for r in clf.all_split_records():
        by_depth[r.depth][r.winner_type] += max(r.gain, 0.0)
        cnt[r.depth][r.winner_type] += 1
    for d in sorted(by_depth):
        tot_gain = sum(by_depth[d].values()) + 1e-12
        tot_cnt = sum(cnt[d].values())
        wls_g = by_depth[d]['wls'] / tot_gain
        inh_g = (by_depth[d]['inherit_A'] + by_depth[d]['inherit_B']) / tot_gain
        wls_c = cnt[d]['wls'] / tot_cnt
        inh_c = (cnt[d]['inherit_A'] + cnt[d]['inherit_B']) / tot_cnt
        print(f"  depth {d}: splits={tot_cnt:4d}  "
              f"wls win={wls_c*100:5.1f}% gain={wls_g*100:5.1f}%  |  "
              f"inherit win={inh_c*100:5.1f}% gain={inh_g*100:5.1f}%")


def exp3_research_ablation(X_tr, y_tr, X_te, y_te):
    """H3a: inheritance on/off in the research implementation."""
    print("\n[Exp 3] Research-impl ablation: inheritance on vs off")
    for name, ratio in [("inherit ON ", 1.0), ("inherit OFF", 0.0)]:
        accs, lls = [], []
        for seed in range(3):
            clf = OQBoostResearch(
                n_estimators=60, learning_rate=0.1, max_depth=4,
                inherited_rp_ratio=ratio, random_state=seed,
            )
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_te)
            accs.append(accuracy_score(y_te, p.argmax(1)))
            lls.append(log_loss(y_te, p))
        print(f"  {name}: acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
              f"logloss={np.mean(lls):.4f}±{np.std(lls):.4f}")


def main():
    # Dataset with correlated features — the case oblique splits should win
    X, y = make_classification(
        n_samples=6000, n_features=30, n_informative=12, n_redundant=10,
        n_classes=2, random_state=7,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    print("=" * 64)
    print(" Drift contradiction analysis — synthetic correlated 6k x 30")
    print("=" * 64)

    clf = fit_research(X_tr, y_tr)
    exp1_drift_by_winner_type(clf)
    exp2_depth_profile(clf)
    exp3_research_ablation(X_tr, y_tr, X_te, y_te)

    # Real data
    data = load_breast_cancer()
    Xb = StandardScaler().fit_transform(data.data).astype(np.float32)
    yb = data.target
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.25, random_state=0)

    print("\n" + "=" * 64)
    print(" Real data — breast_cancer")
    print("=" * 64)
    clf_b = fit_research(Xb_tr, yb_tr)
    exp1_drift_by_winner_type(clf_b)
    exp2_depth_profile(clf_b)


if __name__ == "__main__":
    main()
