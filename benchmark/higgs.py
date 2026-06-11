"""Higgs benchmark — large-scale binary classification (up to 11M samples)."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from _utils import run_benchmark

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "HIGGS.csv.gz"
MAX_ROWS = 2_000_000  # cap to keep runtime reasonable; set None for full 11M


def load_data():
    """
    Prefers local HIGGS.csv.gz (download from UCI:
      https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
      → save as benchmark/data/HIGGS.csv.gz).
    Falls back to OpenML higgs-small (98 050 rows) if not found.
    """
    import pandas as pd
    if CSV_PATH.exists():
        print(f"  Loading HIGGS from {CSV_PATH} (max_rows={MAX_ROWS}) ...")
        df = pd.read_csv(CSV_PATH, header=None, nrows=MAX_ROWS)
        X = df.iloc[:, 1:].values.astype(np.float32)
        y = df.iloc[:, 0].values.astype(int)
        return X, y

    print("  Local HIGGS.csv.gz not found. Using OpenML higgs (subset).")
    from sklearn.datasets import fetch_openml
    ds = fetch_openml("higgs", version=2, as_frame=False, parser="auto")
    X = ds.data.astype(np.float32)
    y = ds.target.astype(int)
    if MAX_ROWS and len(y) > MAX_ROWS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    return X, y


if __name__ == "__main__":
    run_benchmark("higgs", load_data)
