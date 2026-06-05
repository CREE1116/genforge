"""
Smoke tests for HypForgeClassifier.
Fast: uses tiny datasets + 5 estimators so CI finishes in seconds.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.utils.estimator_checks import parametrize_with_checks

from hypforge import HypForgeClassifier


TINY = dict(n_estimators=5, max_depth=2, pool_size=10, verbose=False,
            device="cpu", random_state=0, early_stopping_rounds=None)


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    return X.astype(np.float32), y.astype(np.int64)


@pytest.fixture
def multi_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=0
    )
    return X.astype(np.float32), y.astype(np.int64)


# ── basic fit/predict ─────────────────────────────────────────────────────────

def test_fit_predict_binary(binary_data):
    X, y = binary_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)
    assert set(preds).issubset({0, 1})


def test_fit_predict_multiclass(multi_data):
    X, y = multi_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)


def test_predict_proba_sums_to_one(multi_data):
    X, y = multi_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ── eval_set / early stopping ─────────────────────────────────────────────────

def test_eval_set(binary_data):
    X, y = binary_data
    X_tr, X_val = X[:150], X[150:]
    y_tr, y_val = y[:150], y[150:]
    clf = HypForgeClassifier(**{**TINY, "early_stopping_rounds": 2})
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    assert clf.get_n_trees() <= 5


# ── feature analysis ──────────────────────────────────────────────────────────

def test_feature_importances(binary_data):
    X, y = binary_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    imp = clf.feature_importances_
    assert imp.shape == (10,)
    np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)


def test_get_used_features(binary_data):
    X, y = binary_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    used = clf.get_used_features()
    assert isinstance(used, list)
    assert len(used) <= 10
    assert len(used) > 0


def test_select_reduces_columns(binary_data):
    X, y = binary_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    X_sel = clf.select(X)
    assert X_sel.ndim == 2
    assert X_sel.shape[0] == X.shape[0]
    assert X_sel.shape[1] <= X.shape[1]


def test_transform_is_column_selection(binary_data):
    """transform() should behave as sklearn feature selector (column filter)."""
    X, y = binary_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    X_sel = clf.transform(X)
    assert X_sel.shape[0] == X.shape[0]
    assert X_sel.shape[1] <= X.shape[1]


def test_fit_transform(binary_data):
    """fit_transform(X, y) from TransformerMixin should work end-to-end."""
    X, y = binary_data
    clf = HypForgeClassifier(**TINY)
    X_sel = clf.fit_transform(X, y)
    assert X_sel.shape[0] == X.shape[0]
    assert X_sel.shape[1] <= X.shape[1]


def test_embed_shape(binary_data):
    X, y = binary_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    emb = clf.embed(X)
    assert emb.shape[0] == X.shape[0]
    assert emb.shape[1] == len(clf.get_hypothesis_pool())


def test_embed_top_k(binary_data):
    X, y = binary_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    emb = clf.embed(X, top_k=3)
    assert emb.shape == (X.shape[0], 3)


def test_hypothesis_summary(multi_data):
    X, y = multi_data
    clf = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    df = clf.get_hypothesis_summary()
    assert "type" in df.columns
    assert "fitness" in df.columns
    assert "top_features" in df.columns
    assert len(df) == len(clf.get_hypothesis_pool())


# ── DataFrame input ───────────────────────────────────────────────────────────

def test_dataframe_input(binary_data):
    pd = pytest.importorskip("pandas")
    X, y = binary_data
    cols = [f"feat_{i}" for i in range(X.shape[1])]
    df   = pd.DataFrame(X, columns=cols)
    clf  = HypForgeClassifier(**TINY)
    clf.fit(df, y)
    assert clf.feature_names_in_ == cols
    used = clf.get_used_features()
    assert all(isinstance(u, str) for u in used)
    X_sel = clf.select(df)
    assert X_sel.shape[1] <= len(cols)


# ── cat_features ──────────────────────────────────────────────────────────────

def test_cat_features_index(binary_data):
    X, y = binary_data
    clf  = HypForgeClassifier(**{**TINY, "cat_features": [0, 1]})
    clf.fit(X, y)
    clf.predict(X)


def test_cat_features_name():
    pd = pytest.importorskip("pandas")
    np.random.seed(0)
    X  = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100), "cat": np.random.randint(0, 3, 100)})
    y  = np.random.randint(0, 2, 100)
    clf = HypForgeClassifier(**{**TINY, "cat_features": ["cat"]})
    clf.fit(X, y)


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_load(binary_data, tmp_path):
    from hypforge import load_model
    X, y = binary_data
    clf  = HypForgeClassifier(**TINY)
    clf.fit(X, y)
    proba_before = clf.predict_proba(X)

    path = str(tmp_path / "model.joblib")
    clf.save(path)
    clf2 = load_model(path)

    proba_after = clf2.predict_proba(X)
    np.testing.assert_allclose(proba_before, proba_after, atol=1e-5)


def test_save_load_preserves_feature_info(binary_data, tmp_path):
    pd = pytest.importorskip("pandas")
    X, y = binary_data
    df   = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    clf  = HypForgeClassifier(**TINY)
    clf.fit(df, y)

    path = str(tmp_path / "model.joblib")
    clf.save(path)
    clf2 = HypForgeClassifier.load(path)

    assert clf2.feature_names_in_ == clf.feature_names_in_
    assert clf2.get_n_trees() == clf.get_n_trees()


# ── sklearn pipeline ──────────────────────────────────────────────────────────

def test_pipeline_with_downstream_model(binary_data):
    """HypForge → XGBoost (or LogisticRegression) via sklearn Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    X, y = binary_data
    pipe = Pipeline([
        ("hf",  HypForgeClassifier(**TINY)),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(y),)


# ── sklearn compatibility ─────────────────────────────────────────────────────

def test_get_params_set_params():
    clf = HypForgeClassifier(n_estimators=10, max_depth=3)
    params = clf.get_params()
    assert params["n_estimators"] == 10
    clf.set_params(n_estimators=20)
    assert clf.n_estimators == 20


def test_clone():
    from sklearn.base import clone
    clf  = HypForgeClassifier(**TINY)
    clf2 = clone(clf)
    assert clf2.n_estimators == clf.n_estimators
