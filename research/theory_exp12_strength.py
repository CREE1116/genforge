"""
THEORY.md §2 — predictions P1, P2, P3.

P1: within-tree direction coherence (mean pairwise |cos| of internal split
    directions) is higher with inherited_rp_ratio=1 than 0.
P2: per-tree training-loss reduction is smaller with ratio=1 (weaker trees).
P3: staged validation curves (no ES): at large lr, ratio=1 improves for more
    rounds and reaches a lower minimum; at small lr the gap shrinks/reverses.

Pure C++ engine — tree directions extracted via OQBoostTree.__getstate__.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from oqboost import OQBoostClassifier


def tree_directions(tree, D):
    """Internal-node unit directions (oblique and axis) from a fitted tree."""
    s = tree.__getstate__()
    if s["handle"] is None:
        return np.zeros((0, D), dtype=np.float32)
    W = np.asarray(s["split_weights"], dtype=np.float32).reshape(-1, D)
    is_leaf = np.asarray(s["is_leaf"], dtype=bool)
    W = W[~is_leaf]
    norms = np.linalg.norm(W, axis=1)
    W = W[norms > 1e-8]
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def within_tree_coherence(clf, D, max_trees=40):
    """Mean pairwise |cos| between split directions inside the same tree."""
    vals = []
    for tree in clf.trees_[:max_trees]:
        W = tree_directions(tree, D)
        if len(W) < 2:
            continue
        C = np.abs(W @ W.T)
        iu = np.triu_indices(len(W), k=1)
        vals.append(float(C[iu].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def staged_losses(clf, X, y, lr):
    """Per-round cumulative log loss by accumulating per-tree predictions."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    if getattr(clf, "_col_perm_", None) is not None:
        X = np.ascontiguousarray(X[:, clf._col_perm_])
    N = len(X)
    K = len(clf.F_init_)
    F = np.tile(np.asarray(clf.F_init_, dtype=np.float32), (N, 1))
    losses = []
    for tree in clf.trees_:
        F = F + lr * tree.predict(X)
        Fs = F - F.max(1, keepdims=True)
        P = np.exp(Fs)
        P /= P.sum(1, keepdims=True)
        losses.append(float(log_loss(y, P, labels=list(range(K)))))
    return np.asarray(losses)


def run(name, X, y, lrs=(0.3, 0.1, 0.03), n_trees=400, seeds=(0, 1, 2)):
    print(f"\n{'='*72}\n {name}\n{'='*72}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    D = X_tr.shape[1]

    print(f"{'lr':>5s} {'ratio':>6s} | {'P1 coher':>9s} {'P2 trainΔ/tree':>15s} |"
          f" {'min val ll':>11s} {'@round':>7s} {'final ll':>9s}")
    for lr in lrs:
        for ratio in (1.0, 0.0):
            coher, dstep, vmin, argm, vfin = [], [], [], [], []
            for seed in seeds:
                clf = OQBoostClassifier(
                    n_estimators=n_trees, learning_rate=lr, max_depth=6,
                    inherited_rp_ratio=ratio, random_state=seed,
                    early_stopping_rounds=None)
                clf.fit(X_tr, y_tr)
                coher.append(within_tree_coherence(clf, D))
                tr_l = staged_losses(clf, X_tr, y_tr, lr)
                # mean per-tree train-loss drop over the first 50 rounds
                n0 = min(50, len(tr_l) - 1)
                dstep.append(float((tr_l[0] - tr_l[n0]) / n0))
                va_l = staged_losses(clf, X_te, y_te, lr)
                vmin.append(float(va_l.min()))
                argm.append(int(va_l.argmin()) + 1)
                vfin.append(float(va_l[-1]))
            print(f"{lr:5.2f} {ratio:6.1f} | {np.mean(coher):9.4f} "
                  f"{np.mean(dstep):15.5f} | {np.mean(vmin):11.4f} "
                  f"{np.mean(argm):7.0f} {np.mean(vfin):9.4f}", flush=True)


def main():
    X, y = make_classification(
        n_samples=6000, n_features=30, n_informative=12, n_redundant=10,
        n_classes=2, random_state=7)
    run("synthetic correlated 6k x 30", X.astype(np.float32), y)

    ds = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    df = ds.frame.copy()
    yc = ds.target.astype(int).values
    if ds.target.name in df.columns:
        df = df.drop(columns=[ds.target.name])
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    Xc = df.values.astype(np.float32)
    run("credit_default 30k x 23", Xc, yc, n_trees=300, seeds=(0, 1))


if __name__ == "__main__":
    main()
