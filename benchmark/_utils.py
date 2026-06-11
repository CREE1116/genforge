"""Shared benchmark utilities for GenForge evaluation suite."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

RESULTS_DIR = Path(__file__).parent / "results" / "csv"
N_REPS = 3
TEST_SIZE = 0.2
VAL_FRAC = 0.1  # fraction of train used for early stopping


def _load_best_params(dataset_name: str, model_name: str) -> dict:
    try:
        import json
        from pathlib import Path
        best_params_path = Path(__file__).parent / "results" / "best_params.json"
        if best_params_path.exists():
            with open(best_params_path, "r") as f:
                data = json.load(f)
            return data.get(dataset_name, {}).get(model_name, {})
    except Exception:
        pass
    return {}


def _make_xgboost(n_classes: int, random_state: int, dataset_name: str = ""):
    from xgboost import XGBClassifier
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "tree_method": 'hist',
        "early_stopping_rounds": 50,
        "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
        "verbosity": 0,
        "random_state": random_state,
    }
    best = _load_best_params(dataset_name, "XGBoost")
    params.update(best)
    return XGBClassifier(**params)


def _make_lightgbm(n_classes: int, random_state: int, dataset_name: str = ""):
    from lightgbm import LGBMClassifier
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.8,
        "bagging_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "verbose": -1,
        "random_state": random_state,
    }
    best = _load_best_params(dataset_name, "LightGBM")
    params.update(best)
    return LGBMClassifier(**params)


def _make_catboost(cat_col_indices: list[int] | None, random_state: int, dataset_name: str = ""):
    from catboost import CatBoostClassifier
    params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "early_stopping_rounds": 50,
        "cat_features": cat_col_indices,
        "verbose": 0,
        "random_seed": random_state,
    }
    best = _load_best_params(dataset_name, "CatBoost")
    params.update(best)
    return CatBoostClassifier(**params)


def _make_genforge(cat_col_indices: list[int] | None, random_state: int, dataset_name: str = ""):
    from genforge import GenforgeClassifier
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "early_stopping_rounds": 50,
        "cat_features": cat_col_indices,
        "class_weight": "balanced",
        "prior_alpha": 0.5,
        "verbose": False,
        "random_state": random_state,
    }
    best = _load_best_params(dataset_name, "GenForge")
    params.update(best)
    return GenforgeClassifier(**params)


def _make_genforge_plain(cat_col_indices: list[int] | None, random_state: int, dataset_name: str = ""):
    """GenForge without prior correction — isolates structural (oblique-split) bias."""
    from genforge import GenforgeClassifier
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "early_stopping_rounds": 50,
        "cat_features": cat_col_indices,
        "class_weight": None,
        "prior_alpha": 0.0,
        "verbose": False,
        "random_state": random_state,
    }
    best = _load_best_params(dataset_name, "GenForge")  # share same params as GenForge
    params.update(best)
    return GenforgeClassifier(**params)




def _encode_cats(X_train: np.ndarray, X_test: np.ndarray,
                 cat_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """OrdinalEncode cat columns; NaN → -1 then nan for native models."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = X_train.copy().astype(float)
    X_test = X_test.copy().astype(float)
    X_train[:, cat_idx] = enc.fit_transform(X_train[:, cat_idx])
    X_test[:, cat_idx] = enc.transform(X_test[:, cat_idx])
    X_train[:, cat_idx] = np.where(X_train[:, cat_idx] < 0, np.nan, X_train[:, cat_idx])
    X_test[:, cat_idx] = np.where(X_test[:, cat_idx] < 0, np.nan, X_test[:, cat_idx])
    return X_train, X_test


def _catboost_X(X: np.ndarray, cat_idx: list[int] | None) -> np.ndarray:
    """CatBoost requires cat columns as str/int, numeric columns stay float."""
    if not cat_idx:
        return X
    X_obj = X.astype(float).astype(object)  # object array, values are Python floats
    for c in cat_idx:
        col = X[:, c].astype(float)
        codes = np.empty(len(col), dtype=object)
        for i, v in enumerate(col):
            codes[i] = "nan" if (v != v) else str(int(v))  # NaN check via v!=v
        X_obj[:, c] = codes
    return X_obj


def evaluate_one(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    cat_idx: list[int] | None = None,
) -> dict:
    """Fit, time, and evaluate one model. Returns metrics dict."""
    import warnings
    Xtr, Xval, ytr, yval = train_test_split(
        X_train, y_train, test_size=VAL_FRAC, random_state=0, stratify=y_train
    )

    t0 = time.perf_counter()
    if model_name == "XGBoost":
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    elif model_name == "LightGBM":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                Xtr, ytr,
                eval_set=[(Xval, yval)],
                callbacks=[
                    __import__("lightgbm").early_stopping(stopping_rounds=50, verbose=False),
                    __import__("lightgbm").log_evaluation(period=-1),
                ],
            )
    elif model_name == "CatBoost":
        Xtr_cb  = _catboost_X(Xtr,    cat_idx)
        Xval_cb = _catboost_X(Xval,   cat_idx)
        model.fit(Xtr_cb, ytr, eval_set=(Xval_cb, yval))
    else:
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)])
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_infer = _catboost_X(X_test, cat_idx) if model_name == "CatBoost" else X_test
    y_pred = model.predict(X_infer)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_infer)
    else:
        y_proba = None
    infer_time = time.perf_counter() - t0

    # Clip proba for numerical stability and re-normalize to sum to 1
    if y_proba is not None:
        y_proba = np.clip(y_proba, 1e-7, 1 - 1e-7)
        y_proba /= y_proba.sum(axis=1, keepdims=True)

    labels = np.arange(n_classes)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # Macro specificity: TN/(TN+FP) averaged over OvR decomposition
    specs = []
    for k in range(n_classes):
        tp = cm[k, k]; fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp; tn = cm.sum() - tp - fn - fp
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    specificity = float(np.mean(specs))

    roc_auc = float("nan")
    pr_auc  = float("nan")
    if y_proba is not None:
        try:
            multi = "ovr" if n_classes > 2 else "raise"
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                pr_auc  = average_precision_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr",
                                        average="macro", labels=labels)
                pr_auc  = float(np.mean([
                    average_precision_score(
                        (y_test == k).astype(int), y_proba[:, k]
                    ) for k in range(n_classes)
                ]))
        except Exception:
            pass

    return {
        "accuracy":          accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "recall_macro":      recall_score(y_test, y_pred, average="macro", zero_division=0),
        "specificity_macro": specificity,
        "f1_macro":          f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted":       f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc":           roc_auc,
        "pr_auc":            pr_auc,
        "log_loss":          log_loss(y_test, y_proba) if y_proba is not None else float("nan"),
        "train_time":        train_time,
        "infer_time":        infer_time,
    }


def run_benchmark(
    dataset_name: str,
    load_fn: Callable[[], tuple[np.ndarray, np.ndarray]],
    cat_idx: list[int] | None = None,
    n_reps: int = N_REPS,
) -> pd.DataFrame:
    """Full benchmark: load → split × n_reps × 5 models → save CSV."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    X, y = load_fn()
    y = y.astype(int)
    n_classes = int(y.max()) + 1
    print(f"  Shape: {X.shape}, Classes: {n_classes}")

    results = []
    for rep in range(n_reps):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=rep,
            stratify=y if n_classes < 20 else None,
        )

        if cat_idx:
            X_train, X_test = _encode_cats(X_train, X_test, cat_idx)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        models = {
            "XGBoost":       _make_xgboost(n_classes, random_state=rep, dataset_name=dataset_name),
            "LightGBM":      _make_lightgbm(n_classes, random_state=rep, dataset_name=dataset_name),
            "CatBoost":      _make_catboost(cat_idx, random_state=rep, dataset_name=dataset_name),
            "GenForge":      _make_genforge_plain(cat_idx, random_state=rep, dataset_name=dataset_name),
            "GenForge-balanced": _make_genforge(cat_idx, random_state=rep, dataset_name=dataset_name),
        }



        for model_name, model in models.items():
            print(f"  Rep {rep+1}/{n_reps} — {model_name} ...", end="", flush=True)
            try:
                metrics = evaluate_one(
                    model_name, model,
                    X_train, y_train, X_test, y_test,
                    n_classes, cat_idx=cat_idx,
                )
                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "rep": rep,
                    **metrics,
                }
                results.append(row)
                print(
                    f" acc={metrics['accuracy']:.4f}"
                    f" bal={metrics['balanced_accuracy']:.4f}"
                    f" rec={metrics['recall_macro']:.4f}"
                    f" roc={metrics['roc_auc']:.4f}"
                    f" pr={metrics['pr_auc']:.4f}"
                    f" train={metrics['train_time']:.1f}s"
                )
            except Exception as exc:
                print(f" ERROR: {exc}")

    df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{dataset_name}.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")
    return df


def aggregate_results() -> pd.DataFrame:
    """Load all CSVs, compute mean ± std per (dataset, model)."""
    frames = []
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    metric_cols = ["accuracy", "balanced_accuracy", "recall_macro", "specificity_macro",
                   "f1_macro", "f1_weighted", "roc_auc", "pr_auc",
                   "log_loss", "train_time", "infer_time"]
    agg = df.groupby(["dataset", "model"])[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()
