"""
C++ inherited-slot strategy ablation via OQB_STRATEGIES env gate.

Separates the two continuity scales in the production engine:
  a = Strategy A: perturb parent direction       (within-tree continuity)
  b = Strategy B: parent + gradient-chosen axis  (within-tree continuity,
                                                  gradient-informed)
  c = Strategy C: cache blend                    (cross-tree continuity)
Plus ratio=0.0 (inherited slot replaced by global sparse random) as the
diversity-only reference.

The env var is read once per process, so each config runs in a fresh
subprocess.
"""
from __future__ import annotations

import os
import subprocess
import sys

WORKER = r'''
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from oqboost import OQBoostClassifier

import sys as _sys
ratio = float(_sys.argv[1])
dataset = _sys.argv[2] if len(_sys.argv) > 2 else "binary"
if dataset == "multiclass":
    X, y = make_classification(
        n_samples=10000, n_features=30, n_informative=15, n_classes=4,
        n_clusters_per_class=2, random_state=7)
else:
    X, y = make_classification(
        n_samples=6000, n_features=30, n_informative=12, n_redundant=10,
        n_classes=2, random_state=7)
X = StandardScaler().fit_transform(X).astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

accs, lls = [], []
for seed in range(5):
    clf = OQBoostClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        inherited_rp_ratio=ratio, random_state=seed,
        early_stopping_rounds=None)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)
    accs.append(accuracy_score(y_te, p.argmax(1)))
    lls.append(log_loss(y_te, p))
print(f"acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
      f"logloss={np.mean(lls):.4f}±{np.std(lls):.4f}")
'''

CONFIGS = [
    ("abc (production)", "",    1.0),
    ("ab  (no cache)  ", "ab",  1.0),
    ("c   (cache only)", "c",   1.0),
    ("a   (perturb)   ", "a",   1.0),
    ("b   (grad-axis) ", "b",   1.0),
    ("none (random)   ", "",    0.0),
]


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "binary"
    print(f"C++ strategy ablation — {dataset}, 5 seeds")
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
        line = out.stdout.strip() or out.stderr.strip().splitlines()[-1]
        print(f"  {name}: {line}", flush=True)


if __name__ == "__main__":
    main()
