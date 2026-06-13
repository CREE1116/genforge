"""
OQBoost — Oblique Gradient-Boosted Decision Trees.

Gradient-boosted oblique decision trees where split directions are optimized
by a C++ BFS engine with zero GPU-CPU sync overhead during training.

Quickstart
----------
>>> from oqboost import OQBoostClassifier
>>> clf = OQBoostClassifier(n_estimators=500, max_depth=6)
>>> clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
>>> clf.predict_proba(X_test)
"""

from ._classifier import OQBoostClassifier
from ._regressor import OQBoostRegressor
from ._oqboost import OQBoostTree


def load_model(path: str) -> OQBoostClassifier | OQBoostRegressor:
    """Load a model saved with ``clf.save(path)``."""
    # joblib.load retrieves the actual pickled estimator type
    import joblib
    return joblib.load(path)


__version__ = "0.1.3"
__all__ = [
    "OQBoostClassifier",
    "OQBoostRegressor",
    "OQBoostTree",
    "load_model",
]
