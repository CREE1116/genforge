import numpy as np
import pytest
from sklearn.datasets import make_regression
from oqboost import OQBoostRegressor, load_model

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    return X.astype(np.float32), y.astype(np.float32)

def test_regressor_losses(regression_data):
    X, y = regression_data
    for loss in ["squared_error", "absolute_error", "huber"]:
        reg = OQBoostRegressor(n_estimators=10, max_depth=3, loss=loss, random_state=42)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (len(y),)
        assert not np.isnan(preds).any()

def test_regressor_early_stopping(regression_data):
    X, y = regression_data
    X_tr, X_val = X[:150], X[150:]
    y_tr, y_val = y[:150], y[150:]
    reg = OQBoostRegressor(n_estimators=20, max_depth=2, early_stopping_rounds=2, random_state=42)
    reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    assert reg.get_n_trees() <= 20

def test_regressor_goss(regression_data):
    X, y = regression_data
    reg = OQBoostRegressor(n_estimators=5, max_depth=2, goss=True, random_state=42)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (len(y),)

def test_regressor_sample_weight(regression_data):
    X, y = regression_data
    # Fit with weights zeroing out half of the samples
    sample_weight = np.concatenate([np.ones(100), np.zeros(100)]).astype(np.float32)
    reg = OQBoostRegressor(n_estimators=5, max_depth=2, random_state=42)
    reg.fit(X, y, sample_weight=sample_weight)
    preds = reg.predict(X)
    assert preds.shape == (len(y),)

def test_regressor_save_load(regression_data, tmp_path):
    X, y = regression_data
    reg = OQBoostRegressor(n_estimators=5, max_depth=3, random_state=42)
    reg.fit(X, y)
    preds_before = reg.predict(X)
    
    path = str(tmp_path / "reg_model.joblib")
    reg.save(path)
    
    reg2 = load_model(path)
    preds_after = reg2.predict(X)
    np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)
