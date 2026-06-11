"""
Genforge — Oblique Gradient-Boosted Decision Trees.

Gradient-boosted oblique decision trees where split directions are optimized
by a C++ BFS engine with zero GPU-CPU sync overhead during training.

Quickstart
----------
>>> from genforge import GenforgeClassifier
>>> clf = GenforgeClassifier(n_estimators=500, max_depth=6)
>>> clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
>>> clf.predict_proba(X_test)
"""

from ._classifier import GenforgeClassifier
from ._tree import BFSTree
from ._genforge import GenforgeTree


def load_model(path: str) -> GenforgeClassifier:
    """Load a model saved with ``clf.save(path)``."""
    return GenforgeClassifier.load(path)


__version__ = "0.1.0"
__all__ = [
    "GenforgeClassifier",
    "BFSTree",
    "GenforgeTree",
    "load_model",
]
