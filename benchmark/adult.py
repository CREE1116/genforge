"""Adult Income benchmark — mixed numerical + categorical features, binary classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from _utils import run_benchmark


def load_data():
    from sklearn.datasets import fetch_openml
    ds = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df: pd.DataFrame = ds.frame.copy()

    # Drop rows with missing target
    df = df.dropna(subset=["class"])
    y = (df["class"].str.strip() == ">50K").astype(int).values

    df = df.drop(columns=["class"])

    # Ordinal-encode string columns → float (NaN preserved)
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

    X = df.values.astype(np.float32)
    return X, y


# Categorical column indices after dropping target and encoding strings.
# workclass(1), education(3), marital-status(5), occupation(6),
# relationship(7), race(8), sex(9), native-country(13)
CAT_IDX = [1, 3, 5, 6, 7, 8, 9, 13]


if __name__ == "__main__":
    run_benchmark("adult", load_data, cat_idx=CAT_IDX)
