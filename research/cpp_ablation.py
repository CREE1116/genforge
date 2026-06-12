"""
Production C++ engine ablation: inherited_rp_ratio 0 / 0.5 / 1.0.

Run as a SEPARATE process from drift_analysis.py: importing torch and the
oqboost C++ extension in one process loads two OpenMP runtimes and deadlocks.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from oqboost import OQBoostClassifier


def ablate(X_tr, y_tr, X_te, y_te, tag):
    print(f"\n[C++ ablation: {tag}] inherited_rp_ratio sweep")
    for name, ratio in [("ratio=1.0", 1.0), ("ratio=0.5", 0.5), ("ratio=0.0", 0.0)]:
        accs, lls = [], []
        for seed in range(3):
            clf = OQBoostClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                inherited_rp_ratio=ratio, random_state=seed,
                early_stopping_rounds=None,
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
    ablate(X_tr, y_tr, X_te, y_te, "synthetic correlated 6k x 30")

    data = load_breast_cancer()
    Xb = StandardScaler().fit_transform(data.data).astype(np.float32)
    yb = data.target
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.25, random_state=0)
    ablate(Xb_tr, yb_tr, Xb_te, yb_te, "breast_cancer")


if __name__ == "__main__":
    main()
