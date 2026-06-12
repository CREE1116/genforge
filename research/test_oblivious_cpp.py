import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from oqboost import OQBoostClassifier

def test():
    print("Loading data...")
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data).astype(np.float32)
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    # 1. Standard tree mode
    print("\n--- Training Standard Tree Mode (tree_mode=0) ---")
    clf_std = OQBoostClassifier(
        n_estimators=30,
        max_depth=4,
        tree_mode=0,
        random_state=42,
        verbose=True
    )
    clf_std.fit(X_tr, y_tr)
    acc_std = (clf_std.predict(X_te) == y_te).mean()
    print(f"Standard Tree Test Accuracy: {acc_std:.4f}")

    # 2. Oblivious tree mode
    print("\n--- Training Oblivious Tree Mode (tree_mode=1) ---")
    clf_obl = OQBoostClassifier(
        n_estimators=30,
        max_depth=4,
        tree_mode=1,
        random_state=42,
        verbose=True
    )
    clf_obl.fit(X_tr, y_tr)
    acc_obl = (clf_obl.predict(X_te) == y_te).mean()
    print(f"Oblivious Tree Test Accuracy: {acc_obl:.4f}")

if __name__ == "__main__":
    test()
