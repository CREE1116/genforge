"""
Experiment Runner: Analytical Coordinate Descent (proxy_cd).

Compares:
1. proxy_search_16 (Guided Random Search)
2. proxy_cd_16 (Guided Analytical Coordinate Descent on J(w))
3. pure_random_16 (Unguided Random Search)

across Rotated Synthetic and Digits (binary) datasets.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch


def make_rotation_matrix(D: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    from scipy.stats import ortho_group
    R = ortho_group.rvs(D, random_state=int(rng.integers(1 << 31)))
    return (1.0 - strength) * np.eye(D) + strength * R


def run_comparison(X_tr, y_tr, X_te, y_te, dataset_name, n_seeds=3):
    print(f"\n==================================================")
    print(f"Dataset: {dataset_name}")
    print(f"==================================================")
    
    modes = ["proxy_search_16", "proxy_cd_4", "proxy_cd_1", "proxy_cov_1"]
    records = []
    
    for mode in modes:
        accs = []
        losses = []
        oblique_gains = []
        runtimes = []
        
        print(f"Running {mode}...", end="", flush=True)
        
        for seed in range(n_seeds):
            t0 = time.time()
            clf = OQBoostResearch(
                n_estimators=20,
                learning_rate=0.1,
                max_depth=4,
                use_wls=True,
                inherit_mode=mode,
                inherited_rp_ratio=1.0,
                n_random=0,
                n_inherit=4,
                random_state=seed,
                device='cpu',
                record_alignment=False,
            )
            clf.fit(X_tr, y_tr)
            elapsed = time.time() - t0
            runtimes.append(elapsed)
            
            p = clf.predict_proba(X_te)
            preds = p.argmax(axis=1)
            
            accs.append(accuracy_score(y_te, preds))
            losses.append(log_loss(y_te, p))
            
            # Extract oblique gains
            for tree in clf.trees_:
                for r in tree.split_records_:
                    if r.best_oblique_gain is not None:
                        oblique_gains.append(r.best_oblique_gain)
                        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_loss = np.mean(losses)
        mean_time = np.mean(runtimes)
        mean_gain = np.mean(oblique_gains) if oblique_gains else 0.0
        
        print(f" done. Acc: {mean_acc:.4f}±{std_acc:.4f} | Loss: {mean_loss:.4f} | E[max G]: {mean_gain:.4f} ({mean_time:.2f}s/run)")
        
        records.append({
            "mode": mode,
            "acc": f"{mean_acc:.4f} ± {std_acc:.4f}",
            "loss": f"{mean_loss:.4f}",
            "gain": f"{mean_gain:.4f}",
            "time": f"{mean_time:.2f}s"
        })
        
    return records


def main():
    # 1. Rotated Synthetic
    base_X, base_y = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=10,
        random_state=42,
    )
    rng = np.random.default_rng(42)
    R = make_rotation_matrix(30, 1.0, rng)
    X_rot = (base_X @ R).astype(np.float32)
    X_rot = StandardScaler().fit_transform(X_rot).astype(np.float32)
    X_tr_rot, X_te_rot, y_tr_rot, y_te_rot = train_test_split(X_rot, base_y, test_size=0.25, random_state=42)
    
    # 2. Digits (binary: 0 vs 1)
    digits = load_digits()
    mask = (digits.target == 0) | (digits.target == 1)
    X_dg = StandardScaler().fit_transform(digits.data[mask]).astype(np.float32)
    y_dg = digits.target[mask]
    X_tr_dg, X_te_dg, y_tr_dg, y_te_dg = train_test_split(X_dg, y_dg, test_size=0.25, random_state=42)

    rot_results = run_comparison(X_tr_rot, y_tr_rot, X_te_rot, y_te_rot, "Rotated Synthetic (binary)")
    dg_results = run_comparison(X_tr_dg, y_tr_dg, X_te_dg, y_te_dg, "Digits (binary, 0 vs 1)")

    # Print comparative Markdown tables
    for name, results in [("Rotated Synthetic", rot_results), ("Digits (binary, 0 vs 1)", dg_results)]:
        print(f"\n\n### Comparison Table: {name}")
        print("| Candidate Mode | Test Accuracy ↑ | Test Log Loss ↓ | Expected Best Oblique Gain E[max G] | Time/Run ↓ |")
        print("|---|---|---|---|---|")
        for r in results:
            print(f"| {r['mode']} | {r['acc']} | {r['loss']} | {r['gain']} | {r['time']} |")


if __name__ == "__main__":
    main()
