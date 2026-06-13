# OQBoost API Reference

## `OQBoostClassifier`

```python
from oqboost import OQBoostClassifier
```

Gradient-boosted oblique decision tree classifier. Implements the scikit-learn `BaseEstimator`, `ClassifierMixin`, and `TransformerMixin` interfaces.

### Constructor

```python
OQBoostClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    reg_lambda=1.0,
    subsample=0.8,
    early_stopping_rounds=50,
    random_state=None,
    verbose=False,
    cat_features=None,
    class_weight=None,
    prior_alpha=0.5,
    inherited_rp_ratio=1.0,
    mutation_rate=0.1,
    mutation_strength=0.5,
    pobs=False,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 1000 | Number of boosting rounds |
| `learning_rate` | float | 0.03 | Shrinkage per tree |
| `max_depth` | int | 6 | Max tree depth; leaf budget = 2^max_depth (64 leaves) |
| `reg_lambda` | float | 1.0 | L2 leaf regularization |
| `subsample` | float | 0.8 | Row fraction per tree (0 < subsample ≤ 1) |
| `early_stopping_rounds` | int or None | 50 | Stop if val loss stagnates; requires eval_set |
| `random_state` | int or None | None | Random seed |
| `verbose` | bool | False | Print per-round metrics |
| `cat_features` | list or None | None | Categorical column names (DataFrame) or indices |
| `class_weight` | str or None | None | "balanced" reweights by inverse class frequency |
| `prior_alpha` | float | 0.5 | Strength of balanced reweighting prior correction (0 to 1) |
| `inherited_rp_ratio` | float | 1.0 | Cache-direction candidate fraction |
| `mutation_rate` | float | 0.1 | Noise scale for inherited directions |
| `mutation_strength` | float | 0.5 | Weight of borrowed feature in hybrid directions |
| `pobs` | bool | False | Inject Haar-orthogonal POBS candidates into every node's tournament |

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
clf2 = OQBoostClassifier.load("model.joblib")
```

---

#### `get_n_trees()`

Return number of trees actually fitted (after early stopping).

---

### sklearn Compatibility

`OQBoostClassifier` is compatible with `sklearn.pipeline.Pipeline`, `clone`, `GridSearchCV`, and `check_estimator`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", OQBoostClassifier(n_estimators=300)),
])
pipe.fit(X_train, y_train)
```

---

## `OQBoostTree`

```python
from oqboost import OQBoostTree
```

Single pool-free oblique tree. Useful when you want to manage the boosting loop yourself.

### Constructor

```python
OQBoostTree(
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

## `OQBoostContext`

```python
from oqboost._oqboost import OQBoostContext
```

Reusable binning context for a boosting loop. Bins X once; reuses codes across rounds.

### Constructor

```python
ctx = OQBoostContext(X, D_num=None)
```

### Methods

#### `build(G, H, sub, max_depth, reg_lambda, ...)`

One boosting round. Returns `(OQBoostTree, np.ndarray)` — the fitted tree and predictions.

#### `close()`

Free C++ memory. Called automatically by `__del__`.

---

## `load_model(path)`

```python
from oqboost import load_model
clf = load_model("model.joblib")
```

Shorthand for `OQBoostClassifier.load(path)`.

---

[한국어 버전 (Korean Version)](api.ko.md)
