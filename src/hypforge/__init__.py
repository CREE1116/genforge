"""
HypForge — Hypothesis Pool Evolution for Oblique GBDT.

Gradient-boosted oblique decision trees where split directions evolve from a
pool of gradient-aligned, SVD, and synergy projections.  Tree nodes are built
by a C++ BFS engine with zero GPU–CPU sync overhead during training.

Quickstart
----------
>>> from hypforge import HypForgeClassifier
>>> clf = HypForgeClassifier(n_estimators=500, max_depth=6)
>>> clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
>>> clf.predict_proba(X_test)
"""

from ._classifier import HypForgeClassifier
from ._pool import Hypothesis, HypForgePool
from ._tree import BFSTree

__version__ = "0.1.0"
__all__ = ["HypForgeClassifier", "Hypothesis", "HypForgePool", "BFSTree"]