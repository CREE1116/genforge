"""
Experiment Runner: Verifying the Proxy Search Hypothesis (Experiment PX).

Compares:
1. PX-A: Oracle Search (Derivative-free CD optimization, budget = 4)
2. PX-B: GG-SRP (Standard OQBoost, budget = 16)
3. PX-C: Proxy Search (SIS +-1 random, budget = 16, 32, 64)
4. PX-D: Pure Random (Uniform random +-1, budget = 16, 32, 64)

Evaluates accuracy, log-loss, runtime, and node-level cosine alignment and gain ratio.
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch
from sklearn.datasets import load_digits, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

# Add parent directory to path to ensure oqboost_research can be imported
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch


def make_rotation_matrix(D: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal rotation blended with identity by `strength` ∈ [0, 1]."""
    from scipy.stats import ortho_group
    R = ortho_group.rvs(D, random_state=int(rng.integers(1 << 31)))
    return (1.0 - strength) * np.eye(D) + strength * R


def run_experiment(X_tr, y_tr, X_te, y_te, dataset_name, n_seeds=3):
    print(f"\n==================================================")
    print(f"Dataset: {dataset_name} ({X_tr.shape[0]} train, {X_te.shape[0]} test, {X_tr.shape[1]} features)")
    print(f"==================================================")
    
    configs = [
        {"name": "oracle (B=4)", "inherit_mode": "oracle"},
        {"name": "gg_srp (B=16)", "inherit_mode": "gg_srp"},
        {"name": "proxy_search (B=16)", "inherit_mode": "proxy_search_16"},
        {"name": "proxy_search (B=32)", "inherit_mode": "proxy_search_32"},
        {"name": "proxy_search (B=64)", "inherit_mode": "proxy_search_64"},
        {"name": "pure_random (B=16)", "inherit_mode": "pure_random_16"},
        {"name": "pure_random (B=32)", "inherit_mode": "pure_random_32"},
        {"name": "pure_random (B=64)", "inherit_mode": "pure_random_64"},
    ]
    results = {}
    
    for cfg in configs:
        name = cfg["name"]
        accs = []
        losses = []
        coss = []
        ratios = []
        runtimes = []
        
        print(f"Running '{name}' strategy...", end="", flush=True)
        for seed in range(n_seeds):
            t0 = time.time()
            clf = OQBoostResearch(
                n_estimators=20,
                learning_rate=0.1,
                max_depth=4,
                use_wls=True,
                inherit_mode=cfg["inherit_mode"],
                inherited_rp_ratio=1.0,
                n_random=0,
                n_inherit=4,
                random_state=seed,
                device='cpu',
                record_alignment=True,
            )
            clf.fit(X_tr, y_tr)
            elapsed = time.time() - t0
            runtimes.append(elapsed)
            
            p = clf.predict_proba(X_te)
            preds = p.argmax(axis=1)
            
            accs.append(accuracy_score(y_te, preds))
            losses.append(log_loss(y_te, p))
            
            # Retrieve node-level alignment metrics
            for tree in clf.trees_:
                for r in tree.split_records_:
                    if r.cos_oracle is not None:
                        coss.append(r.cos_oracle)
                    if r.gain_ratio is not None:
                        ratios.append(r.gain_ratio)
            
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        mean_time = np.mean(runtimes)
        
        mean_cos = np.mean(coss) if coss else float('nan')
        std_cos = np.std(coss) if coss else float('nan')
        mean_ratio = np.mean(ratios) if ratios else float('nan')
        std_ratio = np.std(ratios) if ratios else float('nan')
        
        results[name] = {
            "acc_mean": mean_acc,
            "acc_std": std_acc,
            "loss_mean": mean_loss,
            "loss_std": std_loss,
            "time_mean": mean_time,
            "cos_mean": mean_cos,
            "cos_std": std_cos,
            "ratio_mean": mean_ratio,
            "ratio_std": std_ratio,
        }
        print(f" done. Acc: {mean_acc:.4f}±{std_acc:.4f} | Loss: {mean_loss:.4f}±{std_loss:.4f} | Alignment: {mean_cos:.3f} | Gain Ratio: {mean_ratio:.3f}")
        
    return results


def main():
    # 1. Rotated Synthetic
    print("Generating Rotated Synthetic dataset...")
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
    
    # 3. Wine
    wine = load_wine()
    X_wn = StandardScaler().fit_transform(wine.data).astype(np.float32)
    y_wn = wine.target
    X_tr_wn, X_te_wn, y_tr_wn, y_te_wn = train_test_split(X_wn, y_wn, test_size=0.25, random_state=42)

    # 4. Synthetic correlated
    X_syn, y_syn = make_classification(
        n_samples=1500, n_features=20, n_informative=10, n_redundant=5,
        n_classes=2, random_state=7,
    )
    X_syn = StandardScaler().fit_transform(X_syn).astype(np.float32)
    X_tr_syn, X_te_syn, y_tr_syn, y_te_syn = train_test_split(X_syn, y_syn, test_size=0.25, random_state=42)

    all_results = {}
    all_results["rotated_synthetic"] = run_experiment(X_tr_rot, y_tr_rot, X_te_rot, y_te_rot, "Rotated Synthetic (binary)", n_seeds=3)
    all_results["digits"] = run_experiment(X_tr_dg, y_tr_dg, X_te_dg, y_te_dg, "Digits (binary, 0 vs 1)", n_seeds=3)
    all_results["wine"] = run_experiment(X_tr_wn, y_tr_wn, X_te_wn, y_te_wn, "Wine (multiclass, 3 classes)", n_seeds=3)
    all_results["synthetic"] = run_experiment(X_tr_syn, y_tr_syn, X_te_syn, y_te_syn, "Synthetic Correlated", n_seeds=3)

    # Print markdown summary table
    print("\n\n### Summary of Results (Mean ± Std over 3 seeds)\n")
    print("| Dataset | Strategy | Test Accuracy ↑ | Test Log Loss ↓ | Cosine Alignment (w_oracle) | Gain Ratio (G_proxy / G_oracle) | Time/Run ↓ |")
    print("|---|---|---|---|---|---|---|")
    for dataset, res in all_results.items():
        for name in [
            "oracle (B=4)",
            "gg_srp (B=16)",
            "proxy_search (B=16)",
            "proxy_search (B=32)",
            "proxy_search (B=64)",
            "pure_random (B=16)",
            "pure_random (B=32)",
            "pure_random (B=64)",
        ]:
            acc_str = f"{res[name]['acc_mean']:.4f} ± {res[name]['acc_std']:.4f}"
            loss_str = f"{res[name]['loss_mean']:.4f} ± {res[name]['loss_std']:.4f}"
            cos_str = f"{res[name]['cos_mean']:.3f} ± {res[name]['cos_std']:.3f}" if not np.isnan(res[name]['cos_mean']) else "N/A"
            ratio_str = f"{res[name]['ratio_mean']:.3f} ± {res[name]['ratio_std']:.3f}" if not np.isnan(res[name]['ratio_mean']) else "N/A"
            time_str = f"{res[name]['time_mean']:.2f}s"
            print(f"| {dataset} | **{name}** | {acc_str} | {loss_str} | {cos_str} | {ratio_str} | {time_str} |")


if __name__ == "__main__":
    main()
