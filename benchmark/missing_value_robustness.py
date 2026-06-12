"""Missing-value robustness study — Figure 7 in the benchmark spec."""
from __future__ import annotations

import numpy as np
import pandas as pd
from _utils import (
    RESULTS_DIR, N_REPS, evaluate_one,
    _make_xgboost, _make_lightgbm, _make_catboost,
    _make_oqboost, _make_oqboost_plain,
)
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

MISSING_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEED = 42

METRIC_COLS = [
    "accuracy", "balanced_accuracy", "recall_macro", "specificity_macro",
    "f1_macro", "f1_weighted", "roc_auc", "pr_auc", "log_loss",
    "train_time", "infer_time",
]


def inject_missing(X: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    X = X.copy()
    mask = rng.random(X.shape) < ratio
    X[mask] = np.nan
    return X


def load_base():
    ds = fetch_openml(data_id=42477, as_frame=False, parser="auto")
    X = ds.data.astype(np.float32)
    y = ds.target.astype(int)
    return X, y


def run_missing_robustness(n_reps: int = N_REPS) -> pd.DataFrame:
    X_clean, y = load_base()
    n_classes = int(y.max()) + 1
    records = []

    for ratio in MISSING_RATIOS:
        print(f"\n--- Missing ratio = {ratio:.0%} ---")
        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            X = inject_missing(X_clean, ratio, rng)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=rep, stratify=y
            )
            models = {
                "XGBoost":        _make_xgboost(n_classes, random_state=rep),
                "LightGBM":       _make_lightgbm(n_classes, random_state=rep),
                "CatBoost":       _make_catboost(None, random_state=rep),
                "OQBoost":       _make_oqboost_plain(None, random_state=rep),
                "OQBoost-balanced": _make_oqboost(None, random_state=rep),
            }
            for mname, model in models.items():
                print(f"  Rep {rep+1}/{n_reps} missing={ratio:.0%} {mname} ...", end="", flush=True)
                try:
                    m = evaluate_one(mname, model, X_train, y_train, X_test, y_test, n_classes)
                    records.append({
                        "missing_ratio": ratio, "model": mname, "rep": rep,
                        **{k: m[k] for k in METRIC_COLS},
                    })
                    print(
                        f" acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f}"
                        f" rec={m['recall_macro']:.4f} roc={m['roc_auc']:.4f}"
                    )
                except Exception as exc:
                    print(f" ERROR: {exc}")

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "missing_value_robustness.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df


if __name__ == "__main__":
    run_missing_robustness()
