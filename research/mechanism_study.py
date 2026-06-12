"""
Mechanism study: WHY does parent-direction inheritance help when winning
child directions are near-orthogonal to the parent?

Working hypothesis (orthogonality principle)
--------------------------------------------
A split on direction w consumes the gradient signal ALONG w: each child is
conditioned on one side of the hyperplane, so within a child the residual
signal concentrates in the orthogonal complement of w. Near-orthogonal
child winners are therefore EXPECTED — not a contradiction. What inheritance
actually transfers is the parent's feature SUPPORT (which features matter),
not the direction itself.

Experiments (all with use_wls=False — candidate set mirrors the C++ engine)
  E5  support overlap vs angle: if Jaccard(support) is high while the
      angle is large, inheritance works through support reuse.
  E6  cone probe: matched candidate budgets in a ~11° cone around the
      parent vs the 90° orthogonal subspace vs unrestricted (same support).
      If orth >= near systematically, mutation targets the wrong subspace.
  E7  accuracy: inherit_mode 'mutate' (current A/B) vs 'orth' (orthogonal
      support reuse) vs 'both' vs no inheritance.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from collections import defaultdict

from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from oqboost_research import OQBoostResearch


def jaccard_support(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = set(torch.where(a.abs() > 1e-8)[0].tolist())
    sb = set(torch.where(b.abs() > 1e-8)[0].tolist())
    if not sa and not sb:
        return float('nan')
    return len(sa & sb) / len(sa | sb)


def e5_support_overlap(clf):
    print("\n[E5] Support overlap vs direction angle (parent → child winner)")
    rows = []
    for r in clf.all_split_records():
        if r.parent_w is None or math.isnan(r.angle_from_parent):
            continue
        rows.append((jaccard_support(r.w, r.parent_w), r.angle_from_parent))
    if not rows:
        print("  (no non-root splits)")
        return
    jac = np.array([x[0] for x in rows])
    ang = np.array([x[1] for x in rows])
    print(f"  n={len(rows)}")
    print(f"  mean support Jaccard : {np.nanmean(jac):.3f}")
    print(f"  mean direction angle : {np.mean(ang):.1f}°")
    hi_ang = ang > 60
    print(f"  splits with angle>60°: {hi_ang.mean()*100:.1f}% "
          f"(their mean Jaccard = {np.nanmean(jac[hi_ang]):.3f})")


def e6_cone_probe(X_tr, y_tr, tag):
    print(f"\n[E6] Cone probe ({tag}): best gain by cone, matched budgets")
    clf = OQBoostResearch(
        n_estimators=40, learning_rate=0.1, max_depth=4,
        use_wls=False, probe_cones=True, random_state=0,
    )
    clf.fit(X_tr, y_tr)
    by_depth = defaultdict(list)
    for depth, g_near, g_orth, g_rand in clf.all_cone_logs():
        by_depth[depth].append((g_near, g_orth, g_rand))
    print("  depth |  near(~11°)   orth(90°)   rand   | orth/near | orth-wins")
    for d in sorted(by_depth):
        arr = np.array(by_depth[d])
        m = arr.mean(0)
        ratio = m[1] / (m[0] + 1e-12)
        owin = (arr[:, 1] > arr[:, 0]).mean()
        print(f"    {d}   |  {m[0]:9.4f}  {m[1]:9.4f}  {m[2]:7.4f} |"
              f"   {ratio:5.2f}   |  {owin*100:5.1f}%")
    return clf


def e7_accuracy(X_tr, y_tr, X_te, y_te, tag, n_seeds=3):
    print(f"\n[E7] Accuracy by inherit mode ({tag}, use_wls=False)")
    configs = [
        ("none  ", dict(inherited_rp_ratio=0.0)),
        ("mutate", dict(inherit_mode='mutate')),
        ("orth  ", dict(inherit_mode='orth')),
        ("both  ", dict(inherit_mode='both')),
    ]
    for name, kw in configs:
        accs, lls = [], []
        for seed in range(n_seeds):
            clf = OQBoostResearch(
                n_estimators=60, learning_rate=0.1, max_depth=4,
                use_wls=False, random_state=seed, **kw,
            )
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_te)
            accs.append(accuracy_score(y_te, p.argmax(1)))
            lls.append(log_loss(y_te, p))
        print(f"  {name}: acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
              f"logloss={np.mean(lls):.4f}±{np.std(lls):.4f}", flush=True)


def main():
    X, y = make_classification(
        n_samples=6000, n_features=30, n_informative=12, n_redundant=10,
        n_classes=2, random_state=7,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    print("=" * 66)
    print(" Mechanism study — synthetic correlated 6k x 30 (use_wls=False)")
    print("=" * 66)
    clf = e6_cone_probe(X_tr, y_tr, "synthetic")
    e5_support_overlap(clf)
    e7_accuracy(X_tr, y_tr, X_te, y_te, "synthetic")

    data = load_breast_cancer()
    Xb = StandardScaler().fit_transform(data.data).astype(np.float32)
    yb = data.target
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.25, random_state=0)

    print("\n" + "=" * 66)
    print(" Mechanism study — breast_cancer")
    print("=" * 66)
    clf_b = e6_cone_probe(Xb_tr, yb_tr, "breast_cancer")
    e5_support_overlap(clf_b)
    e7_accuracy(Xb_tr, yb_tr, Xb_te, yb_te, "breast_cancer")


if __name__ == "__main__":
    main()
