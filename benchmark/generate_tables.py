"""Generate benchmark summary tables from CSV results."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from _utils import RESULTS_DIR

RESULTS_PARENT = RESULTS_DIR.parent
SUMMARY_PATH = RESULTS_PARENT / "summary.md"

METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Bal. Acc.",
    "f1_macro": "F1 Macro",
    "log_loss": "Log Loss",
    "train_time": "Train (s)",
    "infer_time": "Infer (s)",
}

DATASET_ORDER = [
    "adult", "credit_default", "give_me_some_credit",
    "covertype", "higgs", "rotated_synthetic",
]
MODEL_ORDER = ["XGBoost", "LightGBM", "CatBoost", "OQBoost", "OQBoost-balanced"]


def load_all() -> pd.DataFrame:
    frames = []
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        if p.stem in ("rotation_robustness", "missing_value_robustness", "categorical_robustness"):
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        print("No results CSVs found. Run benchmarks first.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_main_table(df: pd.DataFrame) -> str:
    """Main benchmark table: mean ± std per (dataset, model, metric)."""
    cols = ["accuracy", "balanced_accuracy", "f1_macro", "log_loss", "train_time", "infer_time"]
    agg = (
        df.groupby(["dataset", "model"])[cols]
        .agg(["mean", "std"])
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    lines = ["# OQBoost Benchmark Results\n"]
    lines.append("## Main Benchmark Table\n")
    header = "| Dataset | Model | Accuracy | Bal. Acc. | F1 Macro | Log Loss | Train (s) | Infer (s) |"
    sep = "|---------|-------|----------|-----------|----------|----------|-----------|-----------|"
    lines.append(header)
    lines.append(sep)

    datasets = [d for d in DATASET_ORDER if d in agg["dataset"].values]
    for ds in datasets:
        sub = agg[agg["dataset"] == ds]
        models = [m for m in MODEL_ORDER if m in sub["model"].values]
        for i, model in enumerate(models):
            row = sub[sub["model"] == model].iloc[0]
            ds_label = ds.replace("_", " ").title() if i == 0 else ""

            def fmt(col):
                mean = row.get(f"{col}_mean", float("nan"))
                std = row.get(f"{col}_std", 0.0)
                if col in ("train_time", "infer_time"):
                    return f"{mean:.2f}±{std:.2f}"
                return f"{mean:.4f}±{std:.4f}"

            lines.append(
                f"| {ds_label} | {model} | {fmt('accuracy')} | {fmt('balanced_accuracy')} "
                f"| {fmt('f1_macro')} | {fmt('log_loss')} | {fmt('train_time')} | {fmt('infer_time')} |"
            )
    return "\n".join(lines)


def make_summary_stats(df: pd.DataFrame) -> str:
    gf = df[df["model"] == "OQBoost"]
    lines = ["\n## OQBoost Highlights\n"]
    if gf.empty:
        return "\n".join(lines)
    best_bal = gf.groupby("dataset")["balanced_accuracy"].mean().max()
    fastest_train = gf["train_time"].min()
    fastest_infer = gf["infer_time"].min()
    largest = gf.groupby("dataset").size().index[-1]
    lines += [
        f"- **Best Balanced Accuracy**: {best_bal:.4f}",
        f"- **Fastest Training**: {fastest_train:.2f}s",
        f"- **Fastest Inference**: {fastest_infer:.4f}s",
        f"- **Largest Dataset**: {largest}",
    ]
    return "\n".join(lines)


def main():
    df = load_all()
    if df.empty:
        return

    table = make_main_table(df)
    stats = make_summary_stats(df)
    content = table + "\n" + stats + "\n"

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(content)
    print(f"Summary written → {SUMMARY_PATH}")
    print(content)


if __name__ == "__main__":
    main()
