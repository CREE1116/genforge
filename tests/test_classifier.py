"""Smoke tests for OQBoostClassifier.
Fast: 5 estimators, tiny datasets — CI finishes in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from oqboost import OQBoostClassifier, load_model

TINY = dict(n_estimators=5, max_depth=2, verbose=False, random_state=0,
            early_stopping_rounds=None)


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
    clf = OQBoostClassifier(**TINY)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)
    assert set(preds).issubset({0, 1})


def test_fit_predict_multiclass(multi_data):
    X, y = multi_data
    clf = OQBoostClassifier(**TINY)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)


def test_predict_proba_sums_to_one(multi_data):
    X, y = multi_data
    clf = OQBoostClassifier(**TINY)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ── eval_set / early stopping ─────────────────────────────────────────────────

def test_eval_set(binary_data):
    X, y = binary_data
    X_tr, X_val = X[:150], X[150:]
    y_tr, y_val = y[:150], y[150:]
    clf = OQBoostClassifier(**{**TINY, "early_stopping_rounds": 2})
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    assert clf.get_n_trees() <= 5


# ── NaN + categorical ─────────────────────────────────────────────────────────

def test_nan_handling(binary_data):
    X, y = binary_data
    X_nan = X.copy()
    X_nan[10:20, 0] = np.nan
    X_nan[30:40, 3] = np.nan
    clf = OQBoostClassifier(**TINY)
    clf.fit(X_nan, y)
    preds = clf.predict(X_nan)
    proba = clf.predict_proba(X_nan)
    assert not np.isnan(proba).any()
    assert preds.shape == (len(y),)


def test_categorical_features():
    np.random.seed(42)
    X = np.random.randn(200, 4).astype(np.float32)
    X[:, 2] = np.random.choice([0.0, 1.0, 2.0], size=200)
    X[:, 3] = np.random.choice([0.0, 1.0], size=200)
    X[10:20, 0] = np.nan
    y = np.random.randint(0, 2, 200)

    clf = OQBoostClassifier(**{**TINY, "cat_features": [2, 3]})
    clf.fit(X, y)
    preds = clf.predict(X)
    proba = clf.predict_proba(X)
    assert preds.shape == (200,)
    assert not np.isnan(proba).any()


def test_unseen_categories():
    np.random.seed(0)
    X = np.random.randn(200, 3).astype(np.float32)
    X[:, 2] = np.random.choice([0.0, 1.0, 2.0], size=200)
    y = np.random.randint(0, 2, 200)
    clf = OQBoostClassifier(**{**TINY, "cat_features": [2]})
    clf.fit(X, y)

    X_test = np.random.randn(10, 3).astype(np.float32)
    X_test[:, 2] = 99.0  # unseen category
    preds = clf.predict(X_test)
    assert preds.shape == (10,)


# ── DataFrame input ───────────────────────────────────────────────────────────

def test_dataframe_input(binary_data):
    pd = pytest.importorskip("pandas")
    X, y = binary_data
    cols = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    clf = OQBoostClassifier(**TINY)
    clf.fit(df, y)
    assert hasattr(clf, "feature_names_in_")
    preds = clf.predict(df)
    assert preds.shape == (len(y),)


def test_dataframe_cat_features():
    pd = pytest.importorskip("pandas")
    np.random.seed(0)
    df = pd.DataFrame({
        "num_a": np.random.randn(100),
        "num_b": np.random.randn(100),
        "cat": pd.Categorical(np.random.choice(["a", "b", "c"], 100)),
    })
    y = np.random.randint(0, 2, 100)
    clf = OQBoostClassifier(**{**TINY, "cat_features": ["cat"]})
    clf.fit(df, y)
    clf.predict(df)


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_load(binary_data, tmp_path):
    X, y = binary_data
    clf = OQBoostClassifier(**TINY)
    clf.fit(X, y)
    proba_before = clf.predict_proba(X)

    path = str(tmp_path / "model.joblib")
    clf.save(path)
    clf2 = load_model(path)

    proba_after = clf2.predict_proba(X)
    np.testing.assert_allclose(proba_before, proba_after, atol=1e-5)


# ── sklearn compatibility ─────────────────────────────────────────────────────

def test_get_params_set_params():
    clf = OQBoostClassifier(n_estimators=10, max_depth=3)
    params = clf.get_params()
    assert params["n_estimators"] == 10
    clf.set_params(n_estimators=20)
    assert clf.n_estimators == 20


def test_clone():
    from sklearn.base import clone
    clf = OQBoostClassifier(**TINY)
    clf2 = clone(clf)
    assert clf2.n_estimators == clf.n_estimators


def test_pipeline_compatibility(binary_data):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = binary_data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", OQBoostClassifier(**TINY)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(y),)


def test_goss_sampling(binary_data):
    X, y = binary_data
    clf = OQBoostClassifier(n_estimators=5, max_depth=2, verbose=False, random_state=0, goss=True)
    clf.fit(X, y)
    preds = clf.predict(X)
    proba = clf.predict_proba(X)
    assert preds.shape == (len(y),)
    assert not np.isnan(proba).any()
