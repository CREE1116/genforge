"""
E7 v2 — budget-matched inherit-mode comparison.

Every mode contributes exactly 4 inherited-slot candidates per node, so the
comparison isolates WHERE the candidates point (near-parent vs orthogonal),
not how many there are. use_wls=False mirrors the C++ candidate set.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from oqboost_research import OQBoostResearch


def run(X_tr, y_tr, X_te, y_te, tag, n_seeds=5):
    print(f"\n[E7-matched] {tag} (4 candidates per mode, {n_seeds} seeds)")
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
    run(X_tr, y_tr, X_te, y_te, "synthetic correlated 6k x 30")

    Xm, ym = make_classification(
        n_samples=5000, n_features=20, n_informative=10, n_classes=4,
        n_clusters_per_class=2, random_state=11,
    )
    Xm = StandardScaler().fit_transform(Xm).astype(np.float32)
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.25, random_state=0)
    run(Xm_tr, ym_tr, Xm_te, ym_te, "multiclass 5k x 20 K=4", n_seeds=3)

    data = load_breast_cancer()
    Xb = StandardScaler().fit_transform(data.data).astype(np.float32)
    yb = data.target
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.25, random_state=0)
    run(Xb_tr, yb_tr, Xb_te, yb_te, "breast_cancer")


if __name__ == "__main__":
    main()
