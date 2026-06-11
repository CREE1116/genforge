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

from ._classifier import HypForgeClassifier, SALOTClassifier, GOSClassifier
from ._classifier import TestForgeClassifier  # backwards compat alias
from ._pool import Hypothesis, HypForgePool
from ._testforge import SALOTPool
from ._testforge import TestForgePool  # backwards compat alias
from ._tree import BFSTree
from ._salot import SALOTTree
from ._evopool import EvoPoolClassifier


def load_model(path: str) -> HypForgeClassifier:
    """Load a model saved with ``clf.save(path)``."""
    return HypForgeClassifier.load(path)


__version__ = "0.1.0"
__all__ = [
    # Primary API
    "HypForgeClassifier", "SALOTClassifier", "GOSClassifier", "EvoPoolClassifier",
    "Hypothesis", "HypForgePool", "SALOTPool",
    "BFSTree", "SALOTTree",
    "load_model",
    # Backwards compat
    "TestForgeClassifier", "TestForgePool",
]