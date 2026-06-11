"""CoverType benchmark — multiclass at scale (581 012 samples, 54 features, 7 classes)."""
from __future__ import annotations

import numpy as np
from _utils import run_benchmark


def load_data():
    from sklearn.datasets import fetch_covtype
    ds = fetch_covtype()
    X = ds.data.astype(np.float32)
    y = (ds.target - 1).astype(int)  # 1-indexed → 0-indexed
    return X, y


if __name__ == "__main__":
    run_benchmark("covertype", load_data)
