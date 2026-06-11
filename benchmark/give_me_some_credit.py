"""Give Me Some Credit benchmark — missing-value robustness, financial risk."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from _utils import run_benchmark

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "cs-training.csv"


def load_data():
    """
    Prefer local Kaggle CSV (download from:
      https://www.kaggle.com/competitions/GiveMeSomeCredit/data
      → save as benchmark/data/cs-training.csv).
    Falls back to OpenML credit-g (German credit) if not found.
    """
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, index_col=0)
        X = df.drop("SeriousDlqin2yrs", axis=1).values.astype(np.float32)
        y = df["SeriousDlqin2yrs"].values.astype(int)
        return X, y

    print(f"  Local CSV not found at {CSV_PATH}. Falling back to OpenML credit-g.")
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=31, as_frame=True, parser="auto")  # credit-g (German credit)
    df = ds.frame.copy()
    target_col = ds.target.name if hasattr(ds.target, "name") else "class"

    y = (df[target_col].astype(str).str.lower().isin(["bad", "2", "1"])).astype(int).values

    df = df.drop(columns=[target_col])
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

    X = df.values.astype(np.float32)
    return X, y


if __name__ == "__main__":
    run_benchmark("give_me_some_credit", load_data)
