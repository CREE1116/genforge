import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from oqboost import OQBoostClassifier

def main():
    print("Generating large classification dataset (200k samples, 80 features)...")
    X, y = make_classification(
        n_samples=200000, n_features=80, n_informative=40,
        n_redundant=20, n_classes=2, random_state=42
    )
    X = X.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    print("Fitting OQBoostClassifier (100 estimators)...")
    t0 = time.perf_counter()
    clf = OQBoostClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        subsample=0.8, random_state=42, early_stopping_rounds=None
    )
    clf.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - t0
    print(f"Fit completed in {fit_t:.2f} seconds.")

if __name__ == "__main__":
    main()
