"""Credit Default benchmark — class-imbalanced binary classification."""
from __future__ import annotations

import numpy as np
from _utils import run_benchmark


def load_data():
    """Taiwan credit card default dataset (30 000 samples, 23 features)."""
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=42477, as_frame=False, parser="auto")
    X, y = ds.data.astype(np.float32), ds.target.astype(int)
    return X, y


if __name__ == "__main__":
    run_benchmark("credit_default", load_data)
