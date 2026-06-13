import numpy as np
import pytest
from sklearn.datasets import make_classification
from oqboost import OQBoostClassifier

def test_l1_regularization():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    # Fit with reg_alpha = 0.0
    clf_no_l1 = OQBoostClassifier(n_estimators=3, max_depth=3, reg_alpha=0.0, early_stopping_rounds=None, random_state=42)
    clf_no_l1.fit(X, y)
    
    # Fit with very high reg_alpha = 1000.0 (forces weights to 0)
    clf_high_l1 = OQBoostClassifier(n_estimators=3, max_depth=3, reg_alpha=1000.0, early_stopping_rounds=None, random_state=42)
    clf_high_l1.fit(X, y)
    
    # Ensure high L1 penalty shrinks/zeros leaf values
    for tree in clf_high_l1.trees_:
        # Retrieve the leaf values
        # The underlying array is state of tree, let's look at getstate
        state = tree.__getstate__()
        if state["handle"] == "serialized":
            # For leaves, check that their values are shrunk
            leaf_vals = state["leaf_vals"]
            # With huge L1, leaf values should be very close to 0
            np.testing.assert_allclose(leaf_vals, 0.0, atol=1e-3)

def test_gamma_minimum_gain():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    # Fit with gamma = 0.0 (allows normal splits)
    clf_no_gamma = OQBoostClassifier(n_estimators=1, max_depth=3, gamma=0.0, early_stopping_rounds=None, random_state=42)
    clf_no_gamma.fit(X, y)
    
    # Fit with huge gamma = 1000.0 (prevents all splits, leaving tree as root only)
    clf_high_gamma = OQBoostClassifier(n_estimators=1, max_depth=3, gamma=1000.0, early_stopping_rounds=None, random_state=42)
    clf_high_gamma.fit(X, y)
    
    # Check tree structure
    state_no = clf_no_gamma.trees_[0].__getstate__()
    state_high = clf_high_gamma.trees_[0].__getstate__()
    
    # clf_high_gamma tree should have no splits committed, i.e., all nodes are leaves
    # or is_leaf[0] is 1 (meaning root is leaf)
    is_leaf = state_high["is_leaf"]
    assert is_leaf[0] == 1  # Root node must be a leaf

def test_colsample_bynode():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    clf = OQBoostClassifier(n_estimators=5, max_depth=3, colsample_bynode=0.5, early_stopping_rounds=None, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)
    assert not np.isnan(clf.predict_proba(X)).any()

def test_sample_weight():
    # Construct a dataset where class 1 is only in the second half.
    # If we set sample_weights to 0 for the second half, the model will only see class 0.
    X = np.random.randn(100, 2)
    y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
    
    # Zero out the weights of class 1
    sample_weight = np.concatenate([np.ones(50), np.zeros(50)]).astype(np.float32)
    
    clf = OQBoostClassifier(n_estimators=5, max_depth=2, early_stopping_rounds=None, random_state=42)
    clf.fit(X, y, sample_weight=sample_weight)
    
    # Since class 1 samples were weighted 0, predictions should be all 0 (or heavily biased to 0)
    preds = clf.predict(X)
    # Let's count how many 1s were predicted
    assert np.sum(preds == 1) == 0

def test_ovr_save_load(tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=42)
    clf = OQBoostClassifier(n_estimators=3, max_depth=3, multi_strategy="ovr", early_stopping_rounds=None, random_state=42)
    clf.fit(X, y)
    
    proba_before = clf.predict_proba(X)
    
    path = str(tmp_path / "ovr_model.joblib")
    clf.save(path)
    
    clf2 = OQBoostClassifier.load(path)
    proba_after = clf2.predict_proba(X)
    
    np.testing.assert_allclose(proba_before, proba_after, atol=1e-5)
