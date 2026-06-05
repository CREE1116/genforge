# HypForge

**Hypothesis Pool Evolution for Oblique GBDT**

Gradient-boosted oblique decision trees where split directions evolve from a pool of gradient-aligned, SVD, and synergy projections. Tree building runs in a C++ BFS engine with zero GPU–CPU sync overhead. Works on CPU, CUDA, and Apple MPS (M-series).

## Install

```bash
pip install hypforge
```

Requires a C++ compiler (`clang++` on macOS, `g++` on Linux). The shared library is compiled automatically on first use.

## Quickstart

```python
from hypforge import HypForgeClassifier

clf = HypForgeClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    device="auto",         # auto-selects MPS > CUDA > CPU
    random_state=42,
    verbose=True,
)

clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
proba = clf.predict_proba(X_test)
```

## Feature Analysis

```python
# Which original features did the model actually use?
used = clf.get_used_features()          # ['feat_3', 'feat_7', ...]

# Filter X to only those columns (same order as training)
X_sel = clf.select(X_test)             # ndarray (n_samples, n_used)

# Detailed table: hypothesis type, fitness, top features
df = clf.get_hypothesis_summary()

# Normalized importance array (aligned to training columns)
imp = clf.feature_importances_         # ndarray (n_features,)
clf.plot_importance(max_features=20)   # requires matplotlib
```

## Feature Selection Pipeline

HypForge implements `TransformerMixin`, so it can feed selected features directly into any downstream model:

```python
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

pipe = Pipeline([
    ("hf",  HypForgeClassifier(n_estimators=100, device="auto")),
    ("xgb", XGBClassifier(n_estimators=300)),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)

# Or manually:
clf.fit(X_train, y_train)
X_sel_train = clf.transform(X_train)   # == clf.select(X_train)
X_sel_test  = clf.transform(X_test)
xgb.fit(X_sel_train, y_train)
xgb.predict(X_sel_test)
```

## Learned Embeddings

```python
# Project X through the hypothesis pool → learned nonlinear features
emb = clf.embed(X_test, top_k=50)     # (n_samples, 50) float32
# Use emb as input features for any downstream model
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 500 | Number of boosting rounds |
| `learning_rate` | 0.05 | Shrinkage per tree |
| `max_depth` | 6 | Tree depth (depth 6 → 127 nodes) |
| `reg_lambda` | 1.0 | L2 regularisation on leaf weights |
| `pool_size` | 500 | Max hypotheses in pool |
| `evolve_every` | 1 | Evolve pool every N rounds |
| `subsample` | 0.8 | Row subsampling per tree |
| `early_stopping_rounds` | 50 | Stop if val loss stagnates |
| `device` | `"auto"` | `"auto"` / `"cpu"` / `"cuda"` / `"mps"` |
| `cat_features` | None | Column names or indices for categorical features |
| `random_state` | None | Seed for reproducibility |
| `verbose` | False | Print per-round metrics |

## Device Support

| Device | Requirement |
|--------|-------------|
| CPU | Any PyTorch install |
| CUDA | `torch` with CUDA support |
| MPS (Apple Silicon) | macOS 12.3+, PyTorch ≥ 2.0 |

Hypothesis evolution (matrix projections) runs on GPU; tree building runs in the C++ BFS engine on CPU — zero GPU–CPU synchronisation during tree build.

## License

MIT