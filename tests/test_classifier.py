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
    assert "credit" in df.columns
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


def test_genealogical_evolution(binary_data):
    X, y = binary_data
    clf = HypForgeClassifier(
        n_estimators=10,
        max_depth=3,
        pool_size=40,
        random_state=42
    )
    clf.fit(X, y)

    pool = clf.get_hypothesis_pool()
    assert len(pool) > 0

    for h in pool:
        assert hasattr(h, "parent1")
        assert hasattr(h, "parent2")
        assert hasattr(h, "birth_round")
        assert hasattr(h, "family_id")
        assert hasattr(h, "credit")
        assert h.family_id >= 0

    df = clf.get_hypothesis_summary()
    assert "family_id" in df.columns
    assert "birth_round" in df.columns
    assert "parent1" in df.columns
    assert "parent2" in df.columns
    assert "credit" in df.columns


def test_credit_accumulation(binary_data):
    """Hypotheses used in splits should accumulate positive credit."""
    X, y = binary_data
    clf = HypForgeClassifier(
        n_estimators=30,
        max_depth=3,
        pool_size=50,
        random_state=42,
    )
    clf.fit(X, y)

    pool = clf.get_hypothesis_pool()
    df = clf.get_hypothesis_summary()
    assert "credit" in df.columns
    # At least some hypotheses should have accumulated credit
    assert df["credit"].max() >= 0.0
    used = [h for h in pool if h.use_count > 0]
    assert all(h.credit >= 0.0 for h in used)


def test_dag_combination_evaluation():
    from hypforge import Hypothesis
    # 1. Create a synthetic dataset
    np.random.seed(42)
    X = np.random.randn(10, 5).astype(np.float32)
    
    # 2. Construct three hypotheses manually:
    # h1: base linear projection on feature 0 (w = [1, 0, 0, 0, 0])
    # h2: base linear projection on feature 1 (w = [0, 1, 0, 0, 0])
    # h3: crossover linear combination of h1 and h2 with weights [0.3, 0.7]
    h1 = Hypothesis(hyp_type="linear", w=np.array([1, 0, 0, 0, 0], dtype=np.float32))
    h1.is_base = True
    h1.global_id = 0
    h1.parent1 = -1
    h1.parent2 = -1
    
    h2 = Hypothesis(hyp_type="linear", w=np.array([0, 1, 0, 0, 0], dtype=np.float32))
    h2.is_base = True
    h2.global_id = 1
    h2.parent1 = -1
    h2.parent2 = -1
    
    h3 = Hypothesis(hyp_type="linear", w=np.array([0.3, 0.7, 0.0, 0.0, 0.0], dtype=np.float32), h1=h1, h2=h2)
    h3.is_base = False
    h3.global_id = 2
    h3.parent1 = 0
    h3.parent2 = 1
    
    # 3. Test Python evaluation
    val_h1 = h1.eval(X)
    val_h2 = h2.eval(X)
    val_h3 = h3.eval(X)
    
    expected_h3 = 0.3 * val_h1 + 0.7 * val_h2
    assert np.allclose(val_h3, expected_h3, atol=1e-5)
    
    # 4. Import the pop to C++ and test C++ evaluation
    from hypforge._pool import HypForgePool
    pool = HypForgePool(D=5, max_size=10)
    # import active population containing [h1, h2, h3]
    pool.import_pop([h1, h2, h3])
    
    # Evaluate using the C++ pool
    out_Z = pool.eval(X) # shape (3, 10)
    
    assert np.allclose(out_Z[0], val_h1, atol=1e-5)
    assert np.allclose(out_Z[1], val_h2, atol=1e-5)
    assert np.allclose(out_Z[2], val_h3, atol=1e-5)


def test_pool_mechanics():
    """Credit accumulates; crossover children are generated; base features survive eviction."""
    from hypforge import Hypothesis
    from hypforge._pool import HypForgePool
    import numpy as np

    np.random.seed(42)
    X = np.random.randn(10, 5).astype(np.float32)
    G = np.random.randn(10, 1).astype(np.float32)
    H = np.ones((10, 1), dtype=np.float32)
    sub_indices = np.arange(10, dtype=np.int32)

    h1 = Hypothesis(hyp_type="linear", w=np.array([1, 0, 0, 0, 0], dtype=np.float32))
    h1.is_base = True; h1.parent1 = -1; h1.parent2 = -1; h1.birth_round = 0
    h1.credit = 0.8

    h2 = Hypothesis(hyp_type="linear", w=np.array([0, 1, 0, 0, 0], dtype=np.float32))
    h2.is_base = True; h2.parent1 = -1; h2.parent2 = -1; h2.birth_round = 0
    h2.credit = 0.4

    pool = HypForgePool(D=5, max_size=10)
    pool.import_pop([h1, h2])
    pool.evolve(X, G, H, sub_indices, D_num=2, current_round=1)

    pop = pool.export_pop()
    children = [h for h in pop if h.parent1 >= 0 and h.parent2 >= 0]
    assert len(children) > 0, "Crossover child should be generated."

    # Newborn protection during eviction
    X2 = np.random.randn(10, 2).astype(np.float32)
    G2 = np.random.randn(10, 1).astype(np.float32)
    H2 = np.ones((10, 1), dtype=np.float32)
    sub2 = np.arange(10, dtype=np.int32)
    pool3 = HypForgePool(D=2, max_size=5)
    old_hyps = []
    for i in range(10):
        angle = i * np.pi / 10
        w = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        w /= np.linalg.norm(w) + 1e-8
        h = Hypothesis(hyp_type="linear", w=w)
        h.is_base = False; h.parent1 = -1; h.parent2 = -1
        h.credit = 0.0; h.birth_round = 0; h.rounds_since_last_use = 50
        old_hyps.append(h)
    base2_a = Hypothesis(hyp_type="linear", w=np.array([1.0, 0.0], dtype=np.float32))
    base2_a.is_base = True; base2_a.parent1 = -1; base2_a.parent2 = -1; base2_a.birth_round = 0
    base2_b = Hypothesis(hyp_type="linear", w=np.array([0.0, 1.0], dtype=np.float32))
    base2_b.is_base = True; base2_b.parent1 = -1; base2_b.parent2 = -1; base2_b.birth_round = 0

    pool3.import_pop(old_hyps + [base2_a, base2_b])
    pool3.evolve(X2, G2, H2, sub2, D_num=2, current_round=1)
    pop3 = pool3.export_pop()

    retained_base = [h for h in pop3 if h.is_base]
    assert len(retained_base) > 0, "Base hyps should survive eviction."
    newborns3 = [h for h in pop3 if h.birth_round == 1]
    assert len(newborns3) > 0, "Newborn hypotheses should be admitted."

    # Tree selection: used hypotheses should have accumulated credit
    from hypforge import HypForgeClassifier
    clf = HypForgeClassifier(n_estimators=3, max_depth=2, pool_size=10, random_state=42)
    clf.fit(X, (G > 0).astype(int).ravel())

    pool4 = clf.get_hypothesis_pool()
    assert len(pool4) > 0

    used_hyps = [h for h in pool4 if h.use_count > 0]
    for h in used_hyps:
        assert h.rounds_since_last_use == 0, (
            f"Used hyp should have rounds_since_last_use==0, got {h.rounds_since_last_use}"
        )
        assert h.credit >= 0.0, f"Used hyp should have non-negative credit, got {h.credit}"

