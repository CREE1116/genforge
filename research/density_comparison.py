"""
Compare adaptive child direction mutation with density-preserving candidate budgeting:
1. adaptive_gradient_less: Gradient-adaptive cosine constraints, fixed depth-decaying budget (control baseline)
2. adaptive_gradient_density: Gradient-adaptive cosine constraints, density-preserving budget
3. adaptive_size_density: Sample-ratio-adaptive cosine constraints, density-preserving budget
"""
from __future__ import annotations

import os
import time
import numpy as np
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
        {"name": "adaptive_gradient_less", "inherit_mode": "cosine_adaptive_gradient", "budget_strategy": "less"},
        {"name": "adaptive_gradient_density", "inherit_mode": "cosine_adaptive_gradient", "budget_strategy": "density"},
        {"name": "adaptive_size_density", "inherit_mode": "cosine_adaptive_size", "budget_strategy": "density"},
    ]
    results = {}
    
    for cfg in configs:
        name = cfg["name"]
        accs = []
        losses = []
        
        print(f"Running '{name}' strategy...", end="", flush=True)
        t0 = time.time()
        for seed in range(n_seeds):
            clf = OQBoostResearch(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=4,
                use_wls=True,
                inherit_mode=cfg["inherit_mode"],
                budget_strategy=cfg["budget_strategy"],
                random_state=seed,
                device='cpu',  # CPU is faster for this specific logic
            )
            clf.fit(X_tr, y_tr)
            
            p = clf.predict_proba(X_te)
            preds = p.argmax(axis=1)
            
            accs.append(accuracy_score(y_te, preds))
            losses.append(log_loss(y_te, p))
            
        elapsed = time.time() - t0
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        mean_time = elapsed / n_seeds
        
        results[name] = {
            "acc_mean": mean_acc,
            "acc_std": std_acc,
            "loss_mean": mean_loss,
            "loss_std": std_loss,
            "time_mean": mean_time,
        }
        print(f" done. Acc: {mean_acc:.4f}±{std_acc:.4f} | Loss: {mean_loss:.4f}±{std_loss:.4f} ({mean_time:.2f}s/run)")
        
    return results


def main():
    # 1. Rotated Synthetic (4000 samples)
    print("Generating Rotated Synthetic dataset...")
    base_X, base_y = make_classification(
        n_samples=4000,
        n_features=50,
        n_informative=10,
        random_state=42,
    )
    rng = np.random.default_rng(42)
    R = make_rotation_matrix(50, 1.0, rng)
    X_rot = (base_X @ R).astype(np.float32)
    X_rot = StandardScaler().fit_transform(X_rot).astype(np.float32)
    X_tr_rot, X_te_rot, y_tr_rot, y_te_rot = train_test_split(X_rot, base_y, test_size=0.25, random_state=42)
    
    # 2. Digits
    digits = load_digits()
    X_dg = StandardScaler().fit_transform(digits.data).astype(np.float32)
    y_dg = digits.target
    X_tr_dg, X_te_dg, y_tr_dg, y_te_dg = train_test_split(X_dg, y_dg, test_size=0.25, random_state=42)
    
    # 3. Wine
    wine = load_wine()
    X_wn = StandardScaler().fit_transform(wine.data).astype(np.float32)
    y_wn = wine.target
    X_tr_wn, X_te_wn, y_tr_wn, y_te_wn = train_test_split(X_wn, y_wn, test_size=0.25, random_state=42)

    # 4. Synthetic correlated (2000 samples)
    X_syn, y_syn = make_classification(
        n_samples=2000, n_features=30, n_informative=12, n_redundant=10,
        n_classes=2, random_state=7,
    )
    X_syn = StandardScaler().fit_transform(X_syn).astype(np.float32)
    X_tr_syn, X_te_syn, y_tr_syn, y_te_syn = train_test_split(X_syn, y_syn, test_size=0.25, random_state=42)

    all_results = {}
    all_results["rotated_synthetic"] = run_experiment(X_tr_rot, y_tr_rot, X_te_rot, y_te_rot, "Rotated Synthetic (binary)", n_seeds=3)
    all_results["digits"] = run_experiment(X_tr_dg, y_tr_dg, X_te_dg, y_te_dg, "Digits (multiclass, 10 classes)", n_seeds=3)
    all_results["wine"] = run_experiment(X_tr_wn, y_tr_wn, X_te_wn, y_te_wn, "Wine (multiclass, 3 classes)", n_seeds=3)
    all_results["synthetic"] = run_experiment(X_tr_syn, y_tr_syn, X_te_syn, y_te_syn, "Synthetic Correlated", n_seeds=3)

    # Print markdown table
    print("\n\n### Summary of Results (Mean ± Std over 3 seeds)\n")
    print("| Dataset | Strategy | Test Accuracy ↑ | Test Log Loss ↓ | Mean Time/Run ↓ |")
    print("|---|---|---|---|---|")
    for dataset, res in all_results.items():
        for name in ["adaptive_gradient_less", "adaptive_gradient_density", "adaptive_size_density"]:
            acc_str = f"{res[name]['acc_mean']:.4f} ± {res[name]['acc_std']:.4f}"
            loss_str = f"{res[name]['loss_mean']:.4f} ± {res[name]['loss_std']:.4f}"
            time_str = f"{res[name]['time_mean']:.2f}s"
            print(f"| {dataset} | **{name}** | {acc_str} | {loss_str} | {time_str} |")


if __name__ == "__main__":
    main()
