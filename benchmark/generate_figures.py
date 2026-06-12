"""Generate all benchmark figures from CSV results."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _utils import RESULTS_DIR

FIGURES_DIR = RESULTS_DIR.parent / "figures"
MODEL_COLORS = {
    "XGBoost": "#e07b39",
    "LightGBM": "#3bb34a",
    "CatBoost": "#7b4fce",
    "OQBoost": "#888888",
    "OQBoost-balanced": "#2470c5",
}
MODEL_ORDER = ["XGBoost", "LightGBM", "CatBoost", "OQBoost", "OQBoost-balanced"]
DATASET_LABELS = {
    "adult": "Adult",
    "credit_default": "Credit\nDefault",
    "give_me_some_credit": "Give Me\nCredit",
    "covertype": "CoverType",
    "higgs": "Higgs",
    "rotated_synthetic": "Rotated\nSynth.",
}
DATASET_ORDER = list(DATASET_LABELS.keys())


def _load_main() -> pd.DataFrame:
    frames = []
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        if p.stem in ("rotation_robustness", "missing_value_robustness", "categorical_robustness"):
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _agg(df: pd.DataFrame, metric: str):
    g = df.groupby(["dataset", "model"])[metric].agg(["mean", "std"]).reset_index()
    return g


def _bar_chart(ax, df, metric, datasets):
    n = len(datasets)
    m = len(MODEL_ORDER)
    w = 0.8 / m
    x = np.arange(n)
    for i, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model].set_index("dataset")
        means = [sub.loc[d, "mean"] if d in sub.index else 0.0 for d in datasets]
        stds  = [sub.loc[d, "std"]  if d in sub.index else 0.0 for d in datasets]
        ax.bar(
            x + (i - (m - 1) / 2) * w, means, w,
            yerr=stds, label=model,
            color=MODEL_COLORS.get(model, "grey"),
            capsize=3, error_kw={"linewidth": 0.8},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=8)


def fig1_accuracy(df):
    agg = _agg(df, "accuracy")
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].values]
    fig, ax = plt.subplots(figsize=(10, 4))
    _bar_chart(ax, agg, "accuracy", datasets)
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 1 — Accuracy Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(bottom=max(0, agg["mean"].min() - 0.05))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_accuracy.png", dpi=150)
    plt.close(fig)
    print("Saved fig1_accuracy.png")


def fig2_balanced_accuracy(df):
    agg = _agg(df, "balanced_accuracy")
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].values]
    fig, ax = plt.subplots(figsize=(10, 4))
    _bar_chart(ax, agg, "balanced_accuracy", datasets)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Figure 2 — Balanced Accuracy Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(bottom=max(0, agg["mean"].min() - 0.05))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_balanced_accuracy.png", dpi=150)
    plt.close(fig)
    print("Saved fig2_balanced_accuracy.png")


def fig3_train_time(df):
    agg = _agg(df, "train_time")
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].values]
    fig, ax = plt.subplots(figsize=(10, 4))
    _bar_chart(ax, agg, "train_time", datasets)
    ax.set_yscale("log")
    ax.set_ylabel("Training Time (s, log scale)")
    ax.set_title("Figure 3 — Training Time Comparison")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_train_time.png", dpi=150)
    plt.close(fig)
    print("Saved fig3_train_time.png")


def fig4_infer_time(df):
    agg = _agg(df, "infer_time")
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].values]
    fig, ax = plt.subplots(figsize=(10, 4))
    _bar_chart(ax, agg, "infer_time", datasets)
    ax.set_ylabel("Inference Time (s)")
    ax.set_title("Figure 4 — Inference Time Comparison")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_infer_time.png", dpi=150)
    plt.close(fig)
    print("Saved fig4_infer_time.png")


def fig5_perf_vs_cost(df):
    agg_ba = _agg(df, "balanced_accuracy").rename(columns={"mean": "bal_acc_mean"})
    agg_tr = _agg(df, "train_time").rename(columns={"mean": "train_time_mean"})
    merged = agg_ba.merge(agg_tr[["dataset", "model", "train_time_mean"]], on=["dataset", "model"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODEL_ORDER:
        sub = merged[merged["model"] == model]
        ax.scatter(
            sub["train_time_mean"], sub["bal_acc_mean"],
            label=model, color=MODEL_COLORS.get(model, "grey"),
            s=80, zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                DATASET_LABELS.get(row["dataset"], row["dataset"]),
                (row["train_time_mean"], row["bal_acc_mean"]),
                fontsize=6, alpha=0.7, xytext=(4, 2), textcoords="offset points",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Training Time (s, log scale)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Figure 5 — Performance vs Training Cost")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_perf_vs_cost.png", dpi=150)
    plt.close(fig)
    print("Saved fig5_perf_vs_cost.png")


def fig6_rotation_robustness():
    p = RESULTS_DIR / "rotation_robustness.csv"
    if not p.exists():
        print("rotation_robustness.csv not found; skipping Figure 6.")
        return
    df = pd.read_csv(p)
    agg = df.groupby(["rotation_strength", "model"])["balanced_accuracy"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in MODEL_ORDER:
        sub = agg[agg["model"] == model].sort_values("rotation_strength")
        ax.plot(sub["rotation_strength"], sub["balanced_accuracy"],
                marker="o", label=model, color=MODEL_COLORS.get(model, "grey"))
    ax.set_xlabel("Rotation Strength (0=identity, 1=full random)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Figure 6 — Rotation Robustness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_rotation_robustness.png", dpi=150)
    plt.close(fig)
    print("Saved fig6_rotation_robustness.png")


def fig7_missing_robustness():
    p = RESULTS_DIR / "missing_value_robustness.csv"
    if not p.exists():
        print("missing_value_robustness.csv not found; skipping Figure 7.")
        return
    df = pd.read_csv(p)
    agg = df.groupby(["missing_ratio", "model"])["balanced_accuracy"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in MODEL_ORDER:
        sub = agg[agg["model"] == model].sort_values("missing_ratio")
        ax.plot(sub["missing_ratio"] * 100, sub["balanced_accuracy"],
                marker="o", label=model, color=MODEL_COLORS.get(model, "grey"))
    ax.set_xlabel("Missing Ratio (%)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Figure 7 — Missing Value Robustness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_missing_value.png", dpi=150)
    plt.close(fig)
    print("Saved fig7_missing_value.png")


def fig8_categorical_robustness():
    p = RESULTS_DIR / "categorical_robustness.csv"
    if not p.exists():
        print("categorical_robustness.csv not found; skipping Figure 8.")
        return
    df = pd.read_csv(p)
    agg = df.groupby(["cardinality", "model"])["balanced_accuracy"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in MODEL_ORDER:
        sub = agg[agg["model"] == model].sort_values("cardinality")
        ax.plot(sub["cardinality"], sub["balanced_accuracy"],
                marker="o", label=model, color=MODEL_COLORS.get(model, "grey"))
    ax.set_xlabel("Number of Categories")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Figure 8 — Categorical Cardinality Robustness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_categorical.png", dpi=150)
    plt.close(fig)
    print("Saved fig8_categorical.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_main()
    if df.empty:
        print("No main results found. Checking robustness-only figures.")
    else:
        fig1_accuracy(df)
        fig2_balanced_accuracy(df)
        fig3_train_time(df)
        fig4_infer_time(df)
        fig5_perf_vs_cost(df)
    fig6_rotation_robustness()
    fig7_missing_robustness()
    fig8_categorical_robustness()
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
