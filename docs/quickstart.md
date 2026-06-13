# OQBoost Quickstart

## Installation

```bash
pip install oqboost
```

Requires a C++ compiler (`clang++` on macOS, `g++` on Linux). The compiled engine is bundled in the wheel — no compilation required for supported platforms.

---

## Basic Usage

```python
from oqboost import OQBoostClassifier

clf = OQBoostClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)

clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
```

---

## With Validation and Early Stopping

```python
clf = OQBoostClassifier(
    n_estimators=1000,
    early_stopping_rounds=50,
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
print(f"Used {clf.get_n_trees()} trees (early stop)")
```

---

## Pandas DataFrame with Categorical Features

```python
import pandas as pd

df = pd.read_csv("dataset.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Option 1: auto-detect pandas Categorical / object columns
clf = OQBoostClassifier(n_estimators=500)
clf.fit(X, y)

# Option 2: explicitly name categorical columns
clf = OQBoostClassifier(
    n_estimators=500,
    cat_features=["city", "product_type", "category"],
)
clf.fit(X, y)
```

---

## Handling Missing Values

OQBoost handles NaN natively — no imputation step needed.

```python
import numpy as np

X_train[50, 3] = np.nan   # NaN is fine
X_test[10, 7] = np.nan    # NaN in test is also fine

clf = OQBoostClassifier()
clf.fit(X_train, y_train)
clf.predict(X_test)        # NaN → column mean imputation at runtime
```

---

## Save and Load

```python
clf.save("model.joblib")

from oqboost import load_model
clf2 = load_model("model.joblib")
proba = clf2.predict_proba(X_test)
```

---

## sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", OQBoostClassifier(n_estimators=300, max_depth=5)),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

---

## Tips

**Tuning:** OQBoost has two primary hyperparameters: `max_depth` (4–8) and `reg_lambda` (0.1–10). Start with defaults.

**Speed vs accuracy:** Reduce `n_estimators` + increase `learning_rate` for faster training. Use early stopping to find the optimal round count.

**Imbalanced data:** `class_weight="balanced"` reweights gradients by inverse class frequency. Evaluate with `balanced_accuracy_score` rather than raw accuracy.

**Rotation-robust:** OQBoost's oblique splits naturally adapt to rotated feature spaces. If your data has features that are informative as linear combinations, OQBoost often outperforms axis-aligned models without feature engineering.

---

[한국어 버전 (Korean Version)](quickstart.ko.md)
