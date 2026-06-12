"""Quick benchmark for engine optimization work: accuracy + wall time."""
import sys
import time
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from oqboost import OQBoostClassifier


def bench(name, X_tr, y_tr, X_te, y_te, **kw):
    t0 = time.perf_counter()
    clf = OQBoostClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        random_state=42, early_stopping_rounds=None, **kw,
    )
    clf.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = clf.predict_proba(X_te)
    pred_t = time.perf_counter() - t0

    acc = accuracy_score(y_te, proba.argmax(1))
    ll = log_loss(y_te, proba)
    print(f"{name:24s} fit={fit_t:7.2f}s pred={pred_t:6.3f}s "
          f"acc={acc:.4f} logloss={ll:.4f}")
    return fit_t, pred_t, acc, ll


def main():
    rows = []

    # Synthetic with correlated features (oblique-friendly)
    X, y = make_classification(
        n_samples=20000, n_features=50, n_informative=20,
        n_redundant=15, n_classes=2, random_state=7,
    )
    X = X.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    rows.append(bench("synthetic 20k x 50", X_tr, y_tr, X_te, y_te))

    # Multiclass
    Xm, ym = make_classification(
        n_samples=10000, n_features=30, n_informative=15,
        n_classes=4, n_clusters_per_class=2, random_state=7,
    )
    Xm = Xm.astype(np.float32)
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.25, random_state=0)
    rows.append(bench("multiclass 10k x 30 K=4", Xm_tr, ym_tr, Xm_te, ym_te))

    # Small real
    data = load_breast_cancer()
    Xb = data.data.astype(np.float32)
    yb = data.target
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.25, random_state=0)
    rows.append(bench("breast_cancer 569 x 30", Xb_tr, yb_tr, Xb_te, yb_te))

    total_fit = sum(r[0] for r in rows)
    print(f"\nTOTAL fit time: {total_fit:.2f}s")


if __name__ == "__main__":
    main()
