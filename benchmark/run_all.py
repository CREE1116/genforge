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
):
    skip = set(skip or [])

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
    args = p.parse_args()
    run(
        skip=args.skip,
    )
    print("\nAll benchmarks complete. Run generate_tables.py and generate_figures.py next.")
