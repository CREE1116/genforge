"""
Diagnose the real-benchmark regression after the Strategy O port.

Symptom: adult/credit_default/gmsc AUC dropped slightly and train time
collapsed 4-5x → early stopping fires much earlier → underfit.

Sweep inherited_rp_ratio with the SAME tuned hyperparameters the benchmark
uses, tracking n_trees actually fitted (early-stop round). If ratio=0
recovers old scores, Strategy O hurts on this data; if no ratio recovers,
the GEMM/cache-eval change is implicated; if mid ratios recover, it's a
hyperparameter-rebalance issue.
"""
from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

from oqboost import OQBoostClassifier

BEST = json.loads(
    (Path(__file__).parent.parent / "benchmark/results/best_params.json").read_text()
)


def load_credit_default():
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    df = ds.frame.copy()
    y = ds.target.astype(int).values if hasattr(ds.target, "astype") else None
    target_col = ds.target.name
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    return df.values.astype(np.float32), y


def load_credit_g():
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = ds.frame.copy()
    target_col = ds.target.name
    y = (df[target_col].astype(str).str.lower().isin(["bad", "2", "1"])).astype(int).values
    df = df.drop(columns=[target_col])
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    return df.values.astype(np.float32), y


def sweep(name, X, y, params):
    print(f"\n=== {name} (tuned: {params}) ===")
    for ratio in [1.0, 0.5, 0.25, 0.0]:
        aucs, lls, ntrees = [], [], []
        for rep in range(3):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=rep, stratify=y
            )
            X_tr2, X_val, y_tr2, y_val = train_test_split(
                X_tr, y_tr, test_size=0.15, random_state=0, stratify=y_tr
            )
            clf = OQBoostClassifier(
                n_estimators=1000,
                early_stopping_rounds=50,
                random_state=rep,
                inherited_rp_ratio=ratio,
                **params,
            )
            clf.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)])
            p = clf.predict_proba(X_te)
            aucs.append(roc_auc_score(y_te, p[:, 1]))
            lls.append(log_loss(y_te, p))
            ntrees.append(clf.get_n_trees())
        print(f"  ratio={ratio:4.2f}: auc={np.mean(aucs):.4f}±{np.std(aucs):.4f} "
              f"logloss={np.mean(lls):.4f} n_trees={np.mean(ntrees):.0f}", flush=True)


def main():
    Xc, yc = load_credit_default()
    pc = dict(BEST["credit_default"]["OQBoost"])
    sweep("credit_default 30k x 23", Xc, yc, pc)

    Xg, yg = load_credit_g()
    pg = dict(BEST["give_me_some_credit"]["OQBoost"])
    sweep("credit-g 1k x 20 (gmsc fallback)", Xg, yg, pg)


if __name__ == "__main__":
    main()
