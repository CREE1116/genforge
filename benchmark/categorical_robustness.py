"""Categorical cardinality robustness study — Figure 8 in the benchmark spec."""
from __future__ import annotations

import numpy as np
import pandas as pd
from _utils import (
    RESULTS_DIR, N_REPS, evaluate_one, _encode_cats,
    _make_xgboost, _make_lightgbm, _make_catboost,
    _make_genforge, _make_genforge_plain,
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

CARDINALITIES = [2, 5, 10, 20, 50, 100]
N_SAMPLES = 50_000
N_NUM_FEATURES = 10
N_CAT_FEATURES = 5
SEED = 42

METRIC_COLS = [
    "accuracy", "balanced_accuracy", "recall_macro", "specificity_macro",
    "f1_macro", "f1_weighted", "roc_auc", "pr_auc", "log_loss",
    "train_time", "infer_time",
]


def make_cat_dataset(cardinality: int, seed: int):
    X_num, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_NUM_FEATURES,
        n_informative=6,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    X_cat = rng.integers(0, cardinality, size=(N_SAMPLES, N_CAT_FEATURES)).astype(np.float32)
    X = np.concatenate([X_num.astype(np.float32), X_cat], axis=1)
    cat_idx = list(range(N_NUM_FEATURES, N_NUM_FEATURES + N_CAT_FEATURES))
    return X, y.astype(int), cat_idx


def run_cat_robustness(n_reps: int = N_REPS) -> pd.DataFrame:
    records = []

    for cardinality in CARDINALITIES:
        print(f"\n--- Cardinality = {cardinality} ---")
        for rep in range(n_reps):
            X, y, cat_idx = make_cat_dataset(cardinality, seed=rep)
            n_classes = 2
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=rep
            )
            X_train_enc, X_test_enc = _encode_cats(X_train, X_test, cat_idx)

            models = {
                "XGBoost":        _make_xgboost(n_classes),
                "LightGBM":       _make_lightgbm(n_classes),
                "CatBoost":       _make_catboost(cat_idx),
                "GenForge":       _make_genforge_plain(cat_idx),
                "GenForge-balanced": _make_genforge(cat_idx),
            }
            for mname, model in models.items():
                print(f"  Rep {rep+1}/{n_reps} card={cardinality} {mname} ...", end="", flush=True)
                try:
                    m = evaluate_one(
                        mname, model,
                        X_train_enc, y_train, X_test_enc, y_test,
                        n_classes, cat_idx=cat_idx,
                    )
                    records.append({
                        "cardinality": cardinality, "model": mname, "rep": rep,
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
    out = RESULTS_DIR / "categorical_robustness.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df


if __name__ == "__main__":
    run_cat_robustness()
