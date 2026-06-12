"""Run all OQBoost benchmarks sequentially."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BENCHMARKS = [
    ("adult", "adult", {}),
    ("credit_default", "credit_default", {}),
    ("give_me_some_credit", "give_me_some_credit", {}),
    ("covertype", "covertype", {}),
    ("higgs", "higgs", {}),
    ("rotated_synthetic", "rotated_synthetic", {"main_only": False}),
]

ROBUSTNESS = [
    "missing_value_robustness",
    "categorical_robustness",
]


def run(
    skip: list[str] | None = None,
    tune: bool = False,
    tune_trials: int = 50,
    tune_subset_size: int = 100000,
):
    skip = set(skip or [])

    if tune:
        from optuna_tune import tune_dataset
        for module_name, _, _ in BENCHMARKS:
            if module_name in skip:
                continue
            print(f"\n>>> Optuna Hyperparameter Tuning for {module_name} ...")
            try:
                tune_dataset(module_name, tune_trials, tune_subset_size)
            except Exception as e:
                print(f" ERROR during tuning {module_name}: {e}")

    for module_name, _, kwargs in BENCHMARKS:
        if module_name in skip:
            print(f"Skipping {module_name}")
            continue
        mod = __import__(module_name)
        if hasattr(mod, "main"):
            mod.main(**{k: v for k, v in kwargs.items() if k != "main_only"})
        else:
            fn = getattr(mod, "load_data")
            cat_idx = getattr(mod, "CAT_IDX", None)
            from _utils import run_benchmark
            run_benchmark(module_name, fn, cat_idx=cat_idx)

    for module_name in ROBUSTNESS:
        if module_name in skip:
            print(f"Skipping {module_name}")
            continue
        mod = __import__(module_name)
        if hasattr(mod, "run_missing_robustness"):
            mod.run_missing_robustness()
        elif hasattr(mod, "run_cat_robustness"):
            mod.run_cat_robustness()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip", nargs="*", default=[], help="Dataset module names to skip")
    p.add_argument("--tune", action="store_true", help="Tune hyperparameters with Optuna before running benchmarks")
    p.add_argument("--tune-trials", type=int, default=50, help="Number of Optuna tuning trials per model")
    p.add_argument("--tune-subset-size", type=int, default=100000, help="Subsample training size for speed during tuning; set 0 for full")
    args = p.parse_args()
    run(
        skip=args.skip,
        tune=args.tune,
        tune_trials=args.tune_trials,
        tune_subset_size=args.tune_subset_size,
    )
    print("\nAll benchmarks complete. Run generate_tables.py and generate_figures.py next.")

