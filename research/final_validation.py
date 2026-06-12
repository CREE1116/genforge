"""
Final validation before touching the engine default.

Candidate configurations of the inherited slot:
  abc r=1.0  — production today
  c   r=1.0  — cache-blend only (A/B dropped)
  c   r=0.5  — proposed simplification: half cache, half global random
  none r=0.0 — all global random

Decision rule: change the engine only if a non-abc config is >= abc on ALL
datasets (within noise) and beats it somewhere. Known so far: A/B alone is
harmful, but {abc, c, none} were within noise on two synthetic sets with
opposite orderings — real data must break the tie. adult is the critical
case: the cache stores numeric-only directions, so cache-heavy configs may
underperform on categorical-heavy data.
"""
from __future__ import annotations

import os
import subprocess
import sys

WORKER = r'''
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from oqboost import OQBoostClassifier

ratio = float(sys.argv[1])
dataset = sys.argv[2]

cat_features = None
if dataset == "adult":
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = ds.frame
    y = (ds.target == ">50K").astype(int).values
    X = df.drop(columns=[ds.target.name]) if ds.target.name in df.columns else df
    # subsample for runtime
    idx = np.random.default_rng(0).choice(len(X), 20000, replace=False)
    X = X.iloc[idx].reset_index(drop=True)
    y = y[idx]
elif dataset == "credit_default":
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    df = ds.frame.copy()
    y = ds.target.astype(int).values
    if ds.target.name in df.columns:
        df = df.drop(columns=[ds.target.name])
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    X = df.values.astype(np.float32)
elif dataset == "breast":
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    X, y = d.data.astype(np.float32), d.target
else:  # synthetic50
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=20000, n_features=50, n_informative=20, n_redundant=15,
        n_classes=2, random_state=7)
    X = X.astype(np.float32)

accs, lls, aucs = [], [], []
for seed in range(5):
    if hasattr(X, "iloc"):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = OQBoostClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        inherited_rp_ratio=ratio, random_state=seed,
        early_stopping_rounds=None)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)
    accs.append(accuracy_score(y_te, p.argmax(1)))
    lls.append(log_loss(y_te, p))
    aucs.append(roc_auc_score(y_te, p[:, 1]))
print(f"acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
      f"auc={np.mean(aucs):.4f}±{np.std(aucs):.4f} "
      f"logloss={np.mean(lls):.4f}±{np.std(lls):.4f}")
'''

CONFIGS = [
    ("abc r=1.0", "",   1.0),
    ("c   r=1.0", "c",  1.0),
    ("c   r=0.5", "c",  0.5),
    ("none r=0 ", "",   0.0),
]

DATASETS = ["adult", "credit_default", "breast", "synthetic50"]


def main():
    for dataset in DATASETS:
        print(f"\n=== {dataset} (5 seeds, 200 trees, no ES) ===", flush=True)
        for name, strat, ratio in CONFIGS:
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = "8"
            if strat:
                env["OQB_STRATEGIES"] = strat
            else:
                env.pop("OQB_STRATEGIES", None)
            out = subprocess.run(
                [sys.executable, "-c", WORKER, str(ratio), dataset],
                env=env, capture_output=True, text=True,
            )
            line = out.stdout.strip() or (
                out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "FAILED")
            print(f"  {name}: {line}", flush=True)


if __name__ == "__main__":
    main()
