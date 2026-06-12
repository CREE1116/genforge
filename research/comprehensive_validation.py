"""
Comprehensive Validation Script:
1. Verification 1: Node-level cosine alignment between w_cov and w_cd/WLS.
2. Verification 2 & 3: Benchmarks on real tabular datasets (breast, adult, credit_default) comparing
   axis + oblique combinations against baseline.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.datasets import fetch_openml, load_breast_cancer, make_classification

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch


def load_dataset(name: str):
    print(f"Loading dataset {name}...")
    if name == "adult":
        ds = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        df = ds.frame
        y = (ds.target == ">50K").astype(int).values
        X = df.drop(columns=[ds.target.name]) if ds.target.name in df.columns else df
        # Convert categoricals
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
        # Handle Nans
        X = X.fillna(X.median()).values.astype(np.float32)
        # subsample to 5000 for reasonable CPU speed
        idx = np.random.default_rng(0).choice(len(X), 5000, replace=False)
        X, y = X[idx], y[idx]
        
    elif name == "credit_default":
        ds = fetch_openml(data_id=42477, as_frame=True, parser="auto")
        df = ds.frame.copy()
        y = ds.target.astype(int).values
        if ds.target.name in df.columns:
            df = df.drop(columns=[ds.target.name])
        for col in df.select_dtypes(include=["category", "object"]).columns:
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
        X = df.fillna(df.median()).values.astype(np.float32)
        # subsample to 5000
        idx = np.random.default_rng(0).choice(len(X), 5000, replace=False)
        X, y = X[idx], y[idx]
        
    elif name == "breast":
        d = load_breast_cancer()
        X, y = d.data.astype(np.float32), d.target
        
    else:  # rotated_synthetic
        base_X, base_y = make_classification(
            n_samples=2000, n_features=30, n_informative=10, random_state=42
        )
        # Rotate features
        from scipy.stats import ortho_group
        rng = np.random.default_rng(42)
        R = ortho_group.rvs(30, random_state=int(rng.integers(1 << 31)))
        X = (base_X @ R).astype(np.float32)
        y = base_y
        
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X, y


def run_node_level_comparison():
    print("\n==================================================")
    print("Verification 1: Node-Level Alignment cos(w_cov, w_cd)")
    print("==================================================")
    
    # We will hook into OQBoostResearch to extract intermediate w_cov and w_cd directions at each node
    for ds_name in ["rotated_synthetic", "breast"]:
        X, y = load_dataset(ds_name)
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # We temporarily patch _coordinate_descent_j inside OQBoostResearch to log cosine sims
        cos_sims = []
        
        clf = OQBoostResearch(
            n_estimators=10,
            learning_rate=0.1,
            max_depth=4,
            use_wls=True,
            inherit_mode="proxy_cd_1",
            inherited_rp_ratio=1.0,
            n_random=0,
            n_inherit=4,
            random_state=42,
            device='cpu',
            record_alignment=False,
        )
        
        # Hook into tree build by subclassing or monkey patching _coordinate_descent_j
        import research.oqboost_research as oq_res
        orig_cd_j = oq_res._coordinate_descent_j
        
        def patched_cd_j(w_start_sub, G_vec, A, n_epochs=3):
            # w_start_sub might be unnormalized G_vec, so normalize it
            w_cov = w_start_sub / (w_start_sub.norm() + 1e-8)
            w_cd = orig_cd_j(w_start_sub, G_vec, A, n_epochs)
            # Compute cosine similarity
            cos = float(abs(torch.dot(w_cov, w_cd)).item())
            cos_sims.append(cos)
            return w_cd
            
        oq_res._coordinate_descent_j = patched_cd_j
        try:
            clf.fit(X_tr, y_tr)
        finally:
            oq_res._coordinate_descent_j = orig_cd_j
            
        print(f"Dataset: {ds_name} | Evaluated {len(cos_sims)} nodes")
        print(f"  Mean cos(w_cov, w_cd): {np.mean(cos_sims):.4f} ± {np.std(cos_sims):.4f}")
        print(f"  Min cos: {np.min(cos_sims):.4f} | Max cos: {np.max(cos_sims):.4f}")


def run_real_benchmarks():
    print("\n==================================================")
    print("Verification 2 & 3: Real Tabular Benchmarks")
    print("==================================================")
    
    datasets = ["breast", "adult", "credit_default"]
    modes = [
        "proxy_search_16",
        "proxy_cd_axis_1",
        "proxy_cov_axis_1",
        "proxy_cov_1"
    ]
    
    results = []
    
    for ds_name in datasets:
        X, y = load_dataset(ds_name)
        
        for mode in modes:
            accs, aucs, runtimes = [], [], []
            print(f"Running {ds_name} - {mode}...", end="", flush=True)
            
            for seed in range(3):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.25, random_state=seed, stratify=y
                )
                t0 = time.time()
                clf = OQBoostResearch(
                    n_estimators=100, # 100 trees for robust ensemble evaluation
                    learning_rate=0.1,
                    max_depth=5,
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
                runtimes.append(time.time() - t0)
                
                p = clf.predict_proba(X_te)
                accs.append(accuracy_score(y_te, p.argmax(1)))
                aucs.append(roc_auc_score(y_te, p[:, 1]))
                
            print(f" done. AUC: {np.mean(aucs):.4f} | Acc: {np.mean(accs):.4f} | Time: {np.mean(runtimes):.2f}s")
            results.append({
                "dataset": ds_name,
                "mode": mode,
                "auc": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
                "acc": f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
                "time": f"{np.mean(runtimes):.2f}s"
            })
            
    # Output comparison tables
    for ds_name in datasets:
        print(f"\n\n### Comparison Table: {ds_name} (3 Seeds, 100 Trees)")
        print("| Candidate Mode | Test AUC ↑ | Test Accuracy ↑ | Mean Time/Run ↓ |")
        print("|---|---|---|---|")
        for r in results:
            if r["dataset"] == ds_name:
                print(f"| {r['mode']} | {r['auc']} | {r['acc']} | {r['time']} |")


if __name__ == "__main__":
    run_node_level_comparison()
    run_real_benchmarks()
