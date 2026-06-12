"""
Quick demo: fit OQBoostResearch on sklearn datasets and run analysis.

Run: python research/demo.py
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from oqboost_research import OQBoostResearch, analyze, ablation_comparison


def run_breast_cancer():
    print("\n▶  Breast Cancer dataset")
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target
    X = StandardScaler().fit_transform(X).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = OQBoostResearch(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=1.0,
        d_sub=16,
        verbose=True,
    )
    clf.fit(X_tr, y_tr)
    analyze(clf, X_tr, y_tr, feature_names=list(data.feature_names))

    test_acc = float((clf.predict(X_te) == y_te).mean())
    print(f"Test accuracy: {test_acc:.4f}\n")


def run_ablation():
    print("\n▶  Ablation: axis-only vs oblique-no-inherit vs full OQBoost")
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )
    X = X.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    ablation_comparison(X_tr, y_tr, X_te, y_te, n_estimators=50, max_depth=4)


if __name__ == '__main__':
    run_breast_cancer()
    run_ablation()