"""
POBS evaluation (user-proposed architecture) — THEORY.md §1 frame.

Three diversity-slot generators, budget-matched (12 candidates, axis +
diversity only, inheritance off to isolate the family):

  iid       — current engine behavior: iid sparse ±1, P(nz)=1/√D
  pobs      — user spec: Haar D×D orthogonal block, columns sliced, random
              sparsity mask applied afterwards (mask BREAKS orthogonality —
              tested faithfully as specified)
  pobs_sis  — refinement: SIS-weighted support S (|S|=√D), exact K×K
              orthogonal block ON S → sparse AND orthogonal AND
              gradient-informed

Order-statistics prediction: within-budget candidate redundancy wastes
tournament draws, so pobs_sis ≥ iid; plain pobs depends on how much the mask
destroys (orthogonality loss vs iid's natural near-orthogonality in high D —
in D=20-30 with K=√D≈5 active dims, masked-pobs ≈ iid is plausible).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from oqboost_research import OQBoostResearch


def run(name, X, y, n_seeds=4):
    print(f"\n=== {name} ===")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    for mode in ("iid", "pobs", "pobs_sis"):
        accs, lls = [], []
        for seed in range(n_seeds):
            clf = OQBoostResearch(
                n_estimators=60, learning_rate=0.1, max_depth=4,
                use_wls=False, inherited_rp_ratio=0.0,
                n_random=12, diversity_mode=mode,
                random_state=seed, device='cpu',
            )
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_te)
            accs.append(accuracy_score(y_te, p.argmax(1)))
            lls.append(log_loss(y_te, p))
        print(f"  {mode:9s} acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
              f"logloss={np.mean(lls):.4f}±{np.std(lls):.4f}", flush=True)


def main():
    X, y = make_classification(
        n_samples=6000, n_features=30, n_informative=12, n_redundant=10,
        n_classes=2, random_state=7)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    run("synthetic correlated 6k x 30", X, y)

    Xm, ym = make_classification(
        n_samples=5000, n_features=20, n_informative=10, n_classes=4,
        n_clusters_per_class=2, random_state=11)
    Xm = StandardScaler().fit_transform(Xm).astype(np.float32)
    run("multiclass 5k x 20 K=4", Xm, ym)

    d = load_breast_cancer()
    Xb = StandardScaler().fit_transform(d.data).astype(np.float32)
    run("breast_cancer", Xb, d.target)


if __name__ == "__main__":
    main()
