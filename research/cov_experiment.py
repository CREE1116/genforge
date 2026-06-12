"""Deployment-protocol validation of the covariance oblique candidate.

Spec: docs/cov_oqboost.md — analytical w_cov = -X^T g / ||.|| on the SIS
top-d_sub support, gated in the engine by OQB_COV_MODE:
  0 = production pool (baseline)
  1 = pool + 2 cov candidates (raw covariance + diagonal-Newton scaling)
  2 = cov candidates REPLACE the random/inherited pool (lightweight claim)

Protocol matches benchmark/_utils.py: 80/20 test split, 10% of train as
val for ES(50), tuned params from benchmark/results/best_params.json,
3 reps. Each mode runs in its own subprocess (the engine reads the env
once per process; also avoids any libomp state interplay).

Usage:
  python research/cov_experiment.py            # orchestrate all runs
  python research/cov_experiment.py WORKER <dataset> <mode>   # internal
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "src"))

DATASETS = ["adult", "credit_default", "give_me_some_credit"]
MODES = [0, 1, 2]
N_REPS = 3
TEST_SIZE = 0.2
VAL_FRAC = 0.1


def _load_dataset(name):
    if name == "adult":
        import adult as m
        X, y = m.load_data()
        return X, y, m.CAT_IDX
    if name == "credit_default":
        import credit_default as m
        return (*m.load_data(), None)
    if name == "give_me_some_credit":
        import give_me_some_credit as m
        return (*m.load_data(), None)
    raise ValueError(name)


def worker(dataset: str, mode: int) -> None:
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from oqboost import OQBoostClassifier

    X, y, cat_idx = _load_dataset(dataset)
    with open(ROOT / "benchmark" / "results" / "best_params.json") as f:
        best = json.load(f).get(dataset, {}).get("OQBoost", {})

    rows = []
    for rep in range(N_REPS):
        seed = 42 + rep
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=VAL_FRAC, random_state=seed, stratify=y_tr)

        params = {
            "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.8, "early_stopping_rounds": 50,
            "cat_features": cat_idx, "class_weight": "balanced",
            "prior_alpha": 0.5, "verbose": False, "random_state": seed,
        }
        params.update(best)
        clf = OQBoostClassifier(**params)

        t0 = time.perf_counter()
        clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)])
        fit_s = time.perf_counter() - t0

        P = clf.predict_proba(X_te)
        ll = log_loss(y_te, P)
        acc = accuracy_score(y_te, clf.predict(X_te))
        auc = roc_auc_score(y_te, P[:, 1]) if P.shape[1] == 2 else float("nan")
        rows.append({"rep": rep, "logloss": ll, "acc": acc, "auc": auc,
                     "fit_s": fit_s, "n_trees": clf.get_n_trees()})

    print("RESULT " + json.dumps({"dataset": dataset, "mode": mode, "rows": rows}))


def orchestrate() -> None:
    out_path = ROOT / "research" / "cov_experiment_results.json"
    results = []
    for dataset in DATASETS:
        for mode in MODES:
            env = dict(os.environ, OQB_COV_MODE=str(mode))
            t0 = time.perf_counter()
            p = subprocess.run(
                [sys.executable, __file__, "WORKER", dataset, str(mode)],
                env=env, capture_output=True, text=True, cwd=str(ROOT))
            wall = time.perf_counter() - t0
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if not line:
                print(f"[FAIL] {dataset} mode={mode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
                continue
            rec = json.loads(line[-1][len("RESULT "):])
            rec["wall_s"] = wall
            results.append(rec)
            rows = rec["rows"]
            mean = lambda k: sum(r[k] for r in rows) / len(rows)
            print(f"{dataset:22s} mode={mode}  ll={mean('logloss'):.4f}  "
                  f"acc={mean('acc'):.4f}  auc={mean('auc'):.4f}  "
                  f"fit={mean('fit_s'):.1f}s  trees={mean('n_trees'):.0f}")
            out_path.write_text(json.dumps(results, indent=1))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "WORKER":
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        orchestrate()
