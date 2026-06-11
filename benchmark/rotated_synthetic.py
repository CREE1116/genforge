"""Rotated synthetic benchmark — oblique-split robustness under feature-space rotations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from _utils import run_benchmark, RESULTS_DIR, N_REPS, VAL_FRAC
from _utils import _make_xgboost, _make_lightgbm, _make_catboost
from _utils import _make_genforge, _make_genforge_plain
from _utils import evaluate_one
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

ROTATION_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SAMPLES = 100_000
N_FEATURES = 50
N_INFORMATIVE = 10
SEED = 42

METRIC_COLS = [
    "accuracy", "balanced_accuracy", "recall_macro", "specificity_macro",
    "f1_macro", "f1_weighted", "roc_auc", "pr_auc", "log_loss",
    "train_time", "infer_time",
]


def make_rotation_matrix(D: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal rotation blended with identity by `strength` ∈ [0, 1]."""
    from scipy.stats import ortho_group
    R = ortho_group.rvs(D, random_state=int(rng.integers(1 << 31)))
    return (1.0 - strength) * np.eye(D) + strength * R


def load_base():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        random_state=SEED,
    )
    return X.astype(np.float32), y.astype(int)


def run_rotation_robustness(n_reps: int = N_REPS) -> pd.DataFrame:
    """Sweep rotation strengths; save per-strength CSV."""
    X_base, y = load_base()
    D = X_base.shape[1]
    n_classes = 2
    records = []

    for strength in ROTATION_STRENGTHS:
        print(f"\n--- Rotation strength = {strength:.2f} ---")
        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            R = make_rotation_matrix(D, strength, rng)
            X = (X_base @ R).astype(np.float32)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=rep
            )
            models = {
                "XGBoost":        _make_xgboost(n_classes, random_state=rep),
                "LightGBM":       _make_lightgbm(n_classes, random_state=rep),
                "CatBoost":       _make_catboost(None, random_state=rep),
                "GenForge":       _make_genforge_plain(None, random_state=rep),
                "GenForge-balanced": _make_genforge(None, random_state=rep),
            }
            for mname, model in models.items():
                print(f"  Rep {rep+1}/{n_reps} strength={strength} {mname} ...", end="", flush=True)
                try:
                    m = evaluate_one(mname, model, X_train, y_train, X_test, y_test, n_classes)
                    records.append({
                        "rotation_strength": strength, "model": mname, "rep": rep,
                        **{k: m[k] for k in METRIC_COLS},
                    })
                    print(
                        f" acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f}"
                        f" roc={m['roc_auc']:.4f} train={m['train_time']:.1f}s"
                    )
                except Exception as exc:
                    print(f" ERROR: {exc}")

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rotation_robustness.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df


def main():
    # 1. Main benchmark: standard run at strength=1.0
    rng = np.random.default_rng(SEED)
    R_full = make_rotation_matrix(N_FEATURES, 1.0, rng)

    def load_rotated():
        X, y = load_base()
        return (X @ R_full).astype(np.float32), y

    run_benchmark("rotated_synthetic", load_rotated)

    # 2. Rotation-strength sweep for Figure 6
    run_rotation_robustness()


if __name__ == "__main__":
    main()
