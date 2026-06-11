"""Optuna hyperparameter tuning script for XGBoost, LightGBM, CatBoost, and GenForge."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import optuna
from sklearn.model_selection import train_test_split

# Import benchmark datasets
from adult import load_data as load_adult
from credit_default import load_data as load_credit_default
from give_me_some_credit import load_data as load_gmsc
from covertype import load_data as load_covertype
from higgs import load_data as load_higgs
from rotated_synthetic import load_base as load_rotated


DATASETS = {
    "adult": load_adult,
    "credit_default": load_credit_default,
    "give_me_some_credit": load_gmsc,
    "covertype": load_covertype,
    "higgs": load_higgs,
    "rotated_synthetic": load_rotated,
}

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BEST_PARAMS_FILE = RESULTS_DIR / "best_params.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="covertype",
        choices=list(DATASETS.keys()),
        help="Dataset to tune on",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100000,
        help="Subsample training size for speed; set 0 for full dataset",
    )
    return parser.parse_args()


def get_data(dataset_name: str, subset_size: int):
    print(f"Loading dataset: {dataset_name}...")
    load_fn = DATASETS[dataset_name]
    X, y = load_fn()
    y = y.astype(int)
    n_classes = int(y.max()) + 1

    # Train/Val/Test Split
    # We use a 70/15/15 split for tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y if n_classes < 20 else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1765, random_state=42, stratify=y_train if n_classes < 20 else None
    )

    if subset_size > 0 and len(y_train) > subset_size:
        print(f"Subsampling training set to {subset_size} samples...")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_train), subset_size, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    print(f"Shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, n_classes


def tune_xgboost(X_train, y_train, X_val, y_val, n_classes, n_trials):
    print("\n==================================================")
    print("Tuning XGBoost...")
    print("==================================================")
    from xgboost import XGBClassifier

    def objective(trial):
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
            "tree_method": "hist",
            "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
            "verbosity": 0,
            "random_state": 42,
        }
        
        model = XGBClassifier(**params, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        acc = (preds == y_val).mean()
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best XGBoost Trial: {study.best_trial.value:.4f}")
    return study.best_params


def tune_lightgbm(X_train, y_train, X_val, y_val, n_classes, n_trials):
    print("\n==================================================")
    print("Tuning LightGBM...")
    print("==================================================")
    from lightgbm import LGBMClassifier
    import lightgbm as lgb_mod

    def objective(trial):
        max_depth_choice = trial.suggest_int("max_depth", 4, 12)
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": max_depth_choice,
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "bagging_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
            "verbose": -1,
            "random_state": 42,
        }
        
        model = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb_mod.early_stopping(stopping_rounds=50, verbose=False)],
            )
        
        preds = model.predict(X_val)
        acc = (preds == y_val).mean()
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best LightGBM Trial: {study.best_trial.value:.4f}")
    return study.best_params


def tune_catboost(X_train, y_train, X_val, y_val, n_classes, n_trials):
    print("\n==================================================")
    print("Tuning CatBoost...")
    print("==================================================")
    from catboost import CatBoostClassifier

    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
            "early_stopping_rounds": 50,
            "verbose": 0,
            "random_seed": 42,
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        preds = model.predict(X_val).ravel()
        acc = (preds == y_val).mean()
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best CatBoost Trial: {study.best_trial.value:.4f}")
    return study.best_params


def tune_genforge(X_train, y_train, X_val, y_val, n_classes, n_trials):
    print("\n==================================================")
    print("Tuning GenForge...")
    print("==================================================")
    from genforge import GenforgeClassifier

    def objective(trial):
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0),
            "early_stopping_rounds": 50,
            "verbose": False,
            "random_state": 42,
        }
        
        model = GenforgeClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        preds = model.predict(X_val)
        acc = (preds == y_val).mean()
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best GenForge Trial: {study.best_trial.value:.4f}")
    return study.best_params


def main():
    args = parse_args()
    
    X_train, y_train, X_val, y_val, X_test, y_test, n_classes = get_data(
        args.dataset, args.subset_size
    )
    
    results = {}
    if BEST_PARAMS_FILE.exists():
        try:
            with open(BEST_PARAMS_FILE, "r") as f:
                results = json.load(f)
        except Exception:
            pass

    if args.dataset not in results:
        results[args.dataset] = {}

    # Run tuners
    t0 = time.perf_counter()
    xgb_best = tune_xgboost(X_train, y_train, X_val, y_val, n_classes, args.trials)
    results[args.dataset]["XGBoost"] = xgb_best
    
    lgb_best = tune_lightgbm(X_train, y_train, X_val, y_val, n_classes, args.trials)
    results[args.dataset]["LightGBM"] = lgb_best
    
    cb_best = tune_catboost(X_train, y_train, X_val, y_val, n_classes, args.trials)
    results[args.dataset]["CatBoost"] = cb_best
    
    gf_best = tune_genforge(X_train, y_train, X_val, y_val, n_classes, args.trials)
    results[args.dataset]["GenForge"] = gf_best
    
    total_time = time.perf_counter() - t0
    print(f"\nHyperparameter tuning completed in {total_time:.1f} seconds.")

    # Save to file
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved best parameters to {BEST_PARAMS_FILE}")


if __name__ == "__main__":
    main()
