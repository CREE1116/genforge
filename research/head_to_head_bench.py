"""
Head-to-Head Benchmark Script:
Runs C++ and PyTorch engines in separate processes to prevent OpenMP duplicate library conflicts.
"""
from __future__ import annotations

import os
import sys
import json
import time
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import fetch_openml, load_breast_cancer


def load_dataset(name: str):
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
        
    else:
        raise ValueError(f"Unknown dataset: {name}")
        
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X, y


def run_cpp_benchmark():
    # Only import oqboost inside this function to ensure clean env
    from oqboost import OQBoostClassifier
    
    datasets = ["breast", "adult", "credit_default"]
    seeds = [0, 1, 2]
    results = {}
    
    for ds_name in datasets:
        X, y = load_dataset(ds_name)
        cpp_aucs, cpp_accs, cpp_times = [], [], []
        
        for seed in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=y
            )
            t0 = time.time()
            clf = OQBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                reg_lambda=1.0,
                subsample=0.8,
                early_stopping_rounds=None,
                random_state=seed,
                verbose=False
            )
            clf.fit(X_tr, y_tr)
            cpp_times.append(time.time() - t0)
            
            p = clf.predict_proba(X_te)
            cpp_accs.append(float(accuracy_score(y_te, p.argmax(1))))
            cpp_aucs.append(float(roc_auc_score(y_te, p[:, 1])))
            
        results[ds_name] = {
            "auc_mean": float(np.mean(cpp_aucs)),
            "auc_std": float(np.std(cpp_aucs)),
            "acc_mean": float(np.mean(cpp_accs)),
            "acc_std": float(np.std(cpp_accs)),
            "time_mean": float(np.mean(cpp_times))
        }
    
    with open("research/cpp_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("C++ Benchmark finished successfully.")


def run_py_benchmark():
    # Only import torch and PyTorch classifier inside this function
    from research.oqboost_cov_pytorch import OQBoostCovClassifier
    
    datasets = ["breast", "adult", "credit_default"]
    seeds = [0, 1, 2]
    results = {}
    
    for ds_name in datasets:
        X, y = load_dataset(ds_name)
        py_aucs, py_accs, py_times = [], [], []
        
        for seed in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=y
            )
            t0 = time.time()
            clf = OQBoostCovClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                reg_lambda=1.0,
                subsample=0.8,
                d_sub=16,
                random_state=seed,
                verbose=False,
                device='cpu'
            )
            clf.fit(X_tr, y_tr)
            py_times.append(time.time() - t0)
            
            p = clf.predict_proba(X_te)
            py_accs.append(float(accuracy_score(y_te, p.argmax(1))))
            py_aucs.append(float(roc_auc_score(y_te, p[:, 1])))
            
        results[ds_name] = {
            "auc_mean": float(np.mean(py_aucs)),
            "auc_std": float(np.std(py_aucs)),
            "acc_mean": float(np.mean(py_accs)),
            "acc_std": float(np.std(py_accs)),
            "time_mean": float(np.mean(py_times))
        }
        
    with open("research/py_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("PyTorch Benchmark finished successfully.")


def main():
    if "--run-cpp" in sys.argv:
        run_cpp_benchmark()
        return
    if "--run-py" in sys.argv:
        run_py_benchmark()
        return
        
    print("==================================================")
    print("Starting Head-to-Head Benchmark in separate processes...")
    print("==================================================")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir
    
    # 1. Spawn C++ benchmark
    print("Running C++ Engine Process...")
    cpp_proc = subprocess.run(
        [sys.executable, "research/head_to_head_bench.py", "--run-cpp"],
        env=env, capture_output=True, text=True
    )
    if cpp_proc.returncode != 0:
        print("C++ Process crashed! Stderr:")
        print(cpp_proc.stderr)
        return
    print(cpp_proc.stdout)
    
    # 2. Spawn PyTorch benchmark
    print("Running PyTorch Engine Process...")
    env_py = env.copy()
    env_py["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    py_proc = subprocess.run(
        [sys.executable, "research/head_to_head_bench.py", "--run-py"],
        env=env_py, capture_output=True, text=True
    )
    if py_proc.returncode != 0:
        print("PyTorch Process crashed! Stderr:")
        print(py_proc.stderr)
        return
    print(py_proc.stdout)
    
    # 3. Read results and print table
    with open("research/cpp_results.json") as f:
        cpp_results = json.load(f)
    with open("research/py_results.json") as f:
        py_results = json.load(f)
        
    print("\n\n==================================================")
    print("Head-to-Head Comparison: C++ Production vs PyTorch Covariance")
    print("==================================================")
    
    for ds_name in ["breast", "adult", "credit_default"]:
        print(f"\n### Dataset: {ds_name} (100 Trees, Depth 5, 3 Seeds)")
        print("| Engine / Philosophy | Test AUC ↑ | Test Accuracy ↑ | Fit Time ↓ |")
        print("|---|---|---|---|")
        
        c = cpp_results[ds_name]
        p = py_results[ds_name]
        
        print(f"| C++ Production (Tournament) | {c['auc_mean']:.4f} ± {c['auc_std']:.4f} | {c['acc_mean']:.4f} ± {c['acc_std']:.4f} | {c['time_mean']:.2f}s |")
        print(f"| PyTorch Covariance (No Tourn) | {p['auc_mean']:.4f} ± {p['auc_std']:.4f} | {p['acc_mean']:.4f} ± {p['acc_std']:.4f} | {p['time_mean']:.2f}s |")
        
    # Cleanup temp files
    try:
        os.remove("research/cpp_results.json")
        os.remove("research/py_results.json")
    except OSError:
        pass


if __name__ == "__main__":
    main()
