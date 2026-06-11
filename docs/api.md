# GenForge API Reference

## `GenforgeClassifier`

```python
from genforge import GenforgeClassifier
```

Gradient-boosted oblique decision tree classifier. Implements the scikit-learn `BaseEstimator`, `ClassifierMixin`, and `TransformerMixin` interfaces.

### Constructor

```python
GenforgeClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    reg_lambda=1.0,
    subsample=0.8,
    early_stopping_rounds=50,
    random_state=None,
    verbose=False,
    cat_features=None,
    class_weight="balanced",
    inherited_rp_ratio=1.0,
    mutation_rate=0.1,
    mutation_strength=0.5,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 500 | Number of boosting rounds |
| `learning_rate` | float | 0.05 | Shrinkage per tree |
| `max_depth` | int | 6 | Max tree depth; leaf budget = 2^max_depth (64 leaves) |
| `reg_lambda` | float | 1.0 | L2 leaf regularization |
| `subsample` | float | 0.8 | Row fraction per tree (0 < subsample ≤ 1) |
| `early_stopping_rounds` | int or None | 50 | Stop if val loss stagnates; requires eval_set |
| `random_state` | int or None | None | Random seed |
| `verbose` | bool | False | Print per-round metrics |
| `cat_features` | list or None | None | Categorical column names (DataFrame) or indices |
| `class_weight` | str or None | "balanced" | "balanced" reweights by inverse class frequency |
| `inherited_rp_ratio` | float | 1.0 | Cache-direction candidate fraction |
| `mutation_rate` | float | 0.1 | Noise scale for inherited directions |
| `mutation_strength` | float | 0.5 | Weight of borrowed feature in hybrid directions |

---

### Methods

#### `fit(X, y, eval_set=None, sample_weight=None)`

Fit the model.

```python
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

- `X`: array-like (N, D) or pandas DataFrame. NaN allowed.
- `y`: array-like (N,), integer class labels starting from 0.
- `eval_set`: list of (X_val, y_val) tuples; first tuple used for early stopping.

Returns `self`.

---

#### `predict(X)`

Predict class labels.

```python
y_pred = clf.predict(X_test)  # → np.ndarray (N,)
```

---

#### `predict_proba(X)`

Predict class probabilities.

```python
proba = clf.predict_proba(X_test)  # → np.ndarray (N, K)
```

Rows sum to 1. Uses softmax over ensemble leaf values.

---

#### `save(path)` / `load(path)`

Serialize and deserialize a fitted model.

```python
clf.save("model.joblib")
clf2 = GenforgeClassifier.load("model.joblib")
```

---

#### `get_n_trees()`

Return number of trees actually fitted (after early stopping).

---

### sklearn Compatibility

`GenforgeClassifier` is compatible with `sklearn.pipeline.Pipeline`, `clone`, `GridSearchCV`, and `check_estimator`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GenforgeClassifier(n_estimators=300)),
])
pipe.fit(X_train, y_train)
```

---

## `GenforgeTree`

```python
from genforge import GenforgeTree
```

Single pool-free oblique tree. Useful when you want to manage the boosting loop yourself.

### Constructor

```python
GenforgeTree(
    max_depth=4,
    reg_lambda=1.0,
    subsample=1.0,
    random_state=None,
)
```

### Methods

#### `fit_predict(X, G, H, D_num=None, subset=None, ...)`

Build one tree from gradients G and Hessians H. Returns predictions `(N, K)`.

#### `predict(X)`

Predict leaf values `(N, K)`.

---

## `GenforgeContext`

```python
from genforge._genforge import GenforgeContext
```

Reusable binning context for a boosting loop. Bins X once; reuses codes across rounds.

### Constructor

```python
ctx = GenforgeContext(X, D_num=None)
```

### Methods

#### `build(G, H, sub, max_depth, reg_lambda, ...)`

One boosting round. Returns `(GenforgeTree, np.ndarray)` — the fitted tree and predictions.

#### `close()`

Free C++ memory. Called automatically by `__del__`.

---

## `load_model(path)`

```python
from genforge import load_model
clf = load_model("model.joblib")
```

Shorthand for `GenforgeClassifier.load(path)`.
