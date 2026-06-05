# HypForge

Gradient-boosted oblique decision tree classifier. Runs on CPU, CUDA, and Apple MPS.

## Install

```bash
pip install hypforge
```

> Requires a C++ compiler (`clang++` on macOS, `g++` on Linux, MSVC on Windows).  
> The C++ tree engine is compiled automatically on first use.

---

## Basic usage

```python
from hypforge import HypForgeClassifier

clf = HypForgeClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    device="auto",       # auto-selects MPS → CUDA → CPU
    random_state=42,
)

clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
```

### With validation & early stopping

```python
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

### With a pandas DataFrame

```python
import pandas as pd

df_train = pd.read_csv("train.csv")
X = df_train.drop(columns=["target"])
y = df_train["target"]

clf.fit(X, y, eval_set=[(X_val, y_val)])
```

### Categorical features

```python
# Pass column names (DataFrame) or integer indices (ndarray)
clf = HypForgeClassifier(cat_features=["city", "product_type"])
clf.fit(X, y)
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | `500` | Number of boosting rounds |
| `learning_rate` | `0.05` | Shrinkage applied to each tree |
| `max_depth` | `6` | Tree depth (depth 6 → up to 127 nodes) |
| `reg_lambda` | `1.0` | L2 regularisation on leaf values |
| `pool_size` | `500` | Max hypothesis directions kept in pool |
| `evolve_every` | `1` | Evolve pool every N rounds |
| `subsample` | `0.8` | Row fraction used per tree |
| `early_stopping_rounds` | `50` | Stop if val loss stagnates for N rounds |
| `device` | `"auto"` | `"auto"` / `"cpu"` / `"cuda"` / `"mps"` |
| `cat_features` | `None` | Categorical column names or indices |
| `random_state` | `None` | Seed |
| `verbose` | `False` | Print per-round metrics |

---

## Feature analysis

### Which features did the model use?

```python
# List of feature names sorted by importance (most important first)
clf.get_used_features()
# → ['age', 'income', 'score', ...]

# Dict {feature_name: importance}
clf.get_feature_importances()
# → {'age': 0.31, 'income': 0.24, ...}

# Numpy array aligned to training columns
clf.feature_importances_
# → array([0.04, 0.31, 0.00, 0.24, ...])
```

### Filter X to only the features the model used

```python
# Returns ndarray with only the relevant columns
X_selected = clf.select(X)

# Same thing — sklearn TransformerMixin convention
X_selected = clf.transform(X)
```

### Detailed hypothesis table

```python
df = clf.get_hypothesis_summary()
#    type     fitness  use_count  complexity  top_features
#    linear   0.8312   42         3           age(0.71), income(0.52), ...
#    product  0.7441   18         6           [age(0.71)] × [score(0.88)]
#    abs      0.6903   11         2           income(0.91), ...
```

### Plot feature importances

```python
clf.plot_importance(max_features=20)   # requires matplotlib
```

---

## Use as a sklearn Pipeline step

`HypForgeClassifier` implements `TransformerMixin`, so `transform(X)` returns the
feature-selected matrix. This lets you chain it with any downstream model.

```python
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

pipe = Pipeline([
    ("hf",  HypForgeClassifier(n_estimators=100, device="auto")),
    ("xgb", XGBClassifier(n_estimators=500)),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

```python
# Or manually: train HypForge, extract features, train any model
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

X_tr_sel = clf.transform(X_train)   # only the features HypForge used
X_te_sel = clf.transform(X_test)

from lightgbm import LGBMClassifier
lgb = LGBMClassifier()
lgb.fit(X_tr_sel, y_train)
lgb.predict(X_te_sel)
```

---

## Learned embeddings

Project raw features through the learned hypothesis pool to get a compact
nonlinear embedding. Useful as input to a second-stage model.

```python
# All hypotheses in the pool
emb_train = clf.embed(X_train)          # (n_samples, n_hypotheses)

# Top-50 hypotheses only
emb_train = clf.embed(X_train, top_k=50)
emb_test  = clf.embed(X_test,  top_k=50)
```

---

## Device support

| Device | How to enable |
|--------|---------------|
| CPU | Default fallback |
| CUDA | `device="cuda"` or `device="auto"` with CUDA-enabled PyTorch |
| Apple MPS | `device="mps"` or `device="auto"` on macOS 12.3+ |

`device="auto"` picks the best available device automatically.

---

## License

MIT
