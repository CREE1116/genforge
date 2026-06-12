"""
Experiment Runner: Candidate Width Scaling (Experiment CW).

Sweeps B in {1, 2, 4, 8, 16, 32, 64, 128} to analyze how:
1. Model accuracy (Acc(B))
2. Node-level expected best oblique gain (E[max G(B)])
scale with candidate budget B.

Saves results to research/candidate_width_results.csv.
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

# Add parent directory to path to ensure oqboost_research can be imported
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch


def make_rotation_matrix(D: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal rotation blended with identity by `strength` ∈ [0, 1]."""
    from scipy.stats import ortho_group
    R = ortho_group.rvs(D, random_state=int(rng.integers(1 << 31)))
    return (1.0 - strength) * np.eye(D) + strength * R


def run_sweep(X_tr, y_tr, X_te, y_te, dataset_name, n_seeds=3):
    print(f"\n==================================================")
    print(f"Dataset: {dataset_name} ({X_tr.shape[0]} train, {X_te.shape[0]} test, {X_tr.shape[1]} features)")
    print(f"==================================================")
    
    budgets = [1, 2, 4, 8, 16, 32, 64, 128]
    strategies = ["proxy_search", "pure_random"]
    
    records = []
    
    for strategy in strategies:
        for B in budgets:
            accs = []
            losses = []
            oblique_gains = []
            runtimes = []
            
            inherit_mode = f"{strategy}_{B}"
            print(f"Running {strategy} (B={B})...", end="", flush=True)
            
            for seed in range(n_seeds):
                t0 = time.time()
                clf = OQBoostResearch(
                    n_estimators=20,
                    learning_rate=0.1,
                    max_depth=4,
                    use_wls=True,
                    inherit_mode=inherit_mode,
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
                
                # Extract best oblique gain from split records
                for tree in clf.trees_:
                    for r in tree.split_records_:
                        if r.best_oblique_gain is not None:
                            oblique_gains.append(r.best_oblique_gain)
                            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            mean_time = np.mean(runtimes)
            mean_gain = np.mean(oblique_gains) if oblique_gains else 0.0
            std_gain = np.std(oblique_gains) if oblique_gains else 0.0
            
            print(f" done. Acc: {mean_acc:.4f}±{std_acc:.4f} | Loss: {mean_loss:.4f} | E[max G]: {mean_gain:.4f} ({mean_time:.2f}s/run)")
            
            records.append({
                "dataset": dataset_name,
                "strategy": strategy,
                "B": B,
                "acc_mean": mean_acc,
                "acc_std": std_acc,
                "loss_mean": mean_loss,
                "loss_std": std_loss,
                "gain_mean": mean_gain,
                "gain_std": std_gain,
                "time_mean": mean_time,
            })
            
    return records


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

    all_records = []
    
    # Run sweeps
    all_records.extend(run_sweep(X_tr_rot, y_tr_rot, X_te_rot, y_te_rot, "Rotated Synthetic (binary)", n_seeds=3))
    all_records.extend(run_sweep(X_tr_dg, y_tr_dg, X_te_dg, y_te_dg, "Digits (binary, 0 vs 1)", n_seeds=3))

    # Save to CSV
    df = pd.DataFrame(all_records)
    csv_path = "/Users/leejongmin/code/OQBoost/research/candidate_width_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep results to {csv_path}")

    # Output Markdown Tables
    for dataset in ["Rotated Synthetic (binary)", "Digits (binary, 0 vs 1)"]:
        print(f"\n\n### Results for {dataset} (Mean ± Std over 3 seeds)\n")
        print("| Strategy | B | Test Accuracy ↑ | Test Log Loss ↓ | Expected Best Oblique Gain E[max G] | Time/Run ↓ |")
        print("|---|---|---|---|---|---|")
        df_sub = df[df["dataset"] == dataset]
        for _, row in df_sub.iterrows():
            strategy = row["strategy"]
            B = row["B"]
            acc_str = f"{row['acc_mean']:.4f} ± {row['acc_std']:.4f}"
            loss_str = f"{row['loss_mean']:.4f} ± {row['loss_std']:.4f}"
            gain_str = f"{row['gain_mean']:.4f} ± {row['gain_std']:.4f}"
            time_str = f"{row['time_mean']:.2f}s"
            print(f"| {strategy} | {B} | {acc_str} | {loss_str} | {gain_str} | {time_str} |")


if __name__ == "__main__":
    main()
