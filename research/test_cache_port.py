"""
Verify that the ported global direction cache (Strategy C) baseline accumulates
valid, non-duplicate directions and trains successfully in PyTorch.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

# Add parent directory to path to ensure oqboost_research can be imported
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch


def main():
    print("Loading Digits dataset...")
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data).astype(np.float32)
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    print("Initializing OQBoostResearch with dir_cache and inherit_mode='cache_blend'...")
    clf = OQBoostResearch(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=4,
        use_wls=True,
        inherit_mode='cache_blend',
        inherited_rp_ratio=0.5,
        n_random=4,
        n_inherit=4,
        random_state=42,
        device='cpu',
    )

    print("Fitting model...")
    clf.fit(X_tr, y_tr)

    print("\n--- Cache Verification ---")
    cache = clf.dir_cache_
    print(f"Number of directions in global cache: {len(cache)}")
    
    if len(cache) > 0:
        print("Verifying cache properties:")
        # 1. Norms
        norms = [float(w['w'].norm()) for w in cache]
        mean_norm = np.mean(norms)
        print(f"  Mean direction norm: {mean_norm:.6f} (expected close to 1.0)")
        
        # 2. Uniqueness (dot products)
        max_dot = 0.0
        for i in range(len(cache)):
            for j in range(i + 1, len(cache)):
                dot = float(abs(torch.dot(cache[i]['w'], cache[j]['w'])))
                max_dot = max(max_dot, dot)
        print(f"  Max absolute cosine similarity between cached directions: {max_dot:.6f} (expected <= 0.95)")

    # Predictions
    p = clf.predict_proba(X_te)
    preds = p.argmax(axis=1)
    acc = accuracy_score(y_te, preds)
    ll = log_loss(y_te, p)
    print(f"\nModel Performance on Digits:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Log Loss: {ll:.4f}")

    print("\nPort baseline verification complete!")


if __name__ == "__main__":
    main()
