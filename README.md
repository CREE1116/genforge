# OQBoost

**Gradient-boosted oblique decision trees with gradient-guided random projections.**

OQBoost replaces axis-aligned splits with gradient-guided oblique hyperplanes, combining a diverse sparse-random-projection pool with a dataset-level cache of proven directions — exploiting the geometric structure of the data without expensive numerical optimization.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/cree1116/oqboost/actions/workflows/ci.yml/badge.svg)](https://github.com/cree1116/oqboost/actions)
[![PyPI](https://img.shields.io/pypi/v/oqboost)](https://pypi.org/project/oqboost/)

---

## Key Properties

| Feature | OQBoost |
|---------|---------|
| Split type | Oblique (linear projection of multiple features) |
| Direction finding | GG-SRP: gradient-guided sparse random projection |
| Direction reuse | Dataset-level cache of winning oblique directions (cross-tree) |
| Missing values | Native — NaN handled via mean imputation in C++ |
| Categorical features | Native — gradient-rank encoding per round |
| API | scikit-learn compatible |
| Backend | Compiled C++ with OpenMP parallelism |

---

## Install

```bash
pip install oqboost
```

Pre-compiled wheels are available for macOS (arm64, x86_64) and Linux (x86_64).
On unsupported platforms, `clang++` or `g++` is required to compile from source.

---

## Quickstart

```python
from oqboost import OQBoostClassifier

clf = OQBoostClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)

clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
clf.predict(X_test)
clf.predict_proba(X_test)
```

### Native NaN support

```python
import numpy as np
X_train[50, 3] = np.nan   # NaN anywhere is fine
clf.fit(X_train, y_train)
clf.predict(X_test)        # NaN → column-mean imputation at inference
```

### Native categorical support

```python
import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# auto-detects pandas Categorical / object columns
clf = OQBoostClassifier(n_estimators=500)
clf.fit(X, y)

# or specify explicitly
clf = OQBoostClassifier(cat_features=["city", "product"])
clf.fit(X, y)
```

### Early stopping

```python
clf = OQBoostClassifier(n_estimators=2000, early_stopping_rounds=50)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
print(f"Stopped at {clf.get_n_trees()} trees")
```

### Save / load

```python
clf.save("model.joblib")
from oqboost import load_model
clf2 = load_model("model.joblib")
```

---

## Benchmark Results

All benchmarks: 80/20 train-test split, 3 repetitions, mean reported. Default hyperparameters throughout. No model-specific tuning.

### Main Benchmark Table

| Dataset | Model | Bal. Acc ↑ | F1 Macro ↑ | Log Loss ↓ | Train (s) | Infer (s) |
|---------|-------|-----------|-----------|----------|-----------|-----------|
| Adult Income | XGBoost | 0.7983 | 0.8151 | 0.2748 | **0.32** | **0.003** |
| | LightGBM | 0.8012 | 0.8180 | 0.2752 | 1.73 | 0.021 |
| | CatBoost | 0.7965 | 0.8153 | 0.2734 | 11.11 | 0.053 |
| | **OQBoost** | **0.8444** | 0.8030 | 0.3248 | 6.17 | 0.212 |
| Credit Default | XGBoost | 0.6539 | 0.6795 | 0.4303 | **0.32** | **0.002** |
| | LightGBM | 0.6605 | 0.6865 | 0.4271 | 1.68 | 0.008 |
| | **CatBoost** | **0.6619** | **0.6879** | **0.4257** | 6.33 | 0.005 |
| | OQBoost | 0.5936 | 0.5900 | 0.4893 | 0.83 | 0.004 |
| Give Me Credit | XGBoost | 0.6496 | 0.6603 | 0.5246 | **0.28** | **0.001** |
| | LightGBM | 0.6532 | 0.6631 | 0.5501 | 1.03 | 0.003 |
| | CatBoost | 0.6841 | 0.6987 | **0.4919** | 0.72 | 0.001 |
| | **OQBoost** | **0.6909** | 0.6865 | 0.5241 | 0.96 | 0.044 |
| Rotated Synth. | XGBoost | 0.9704 | 0.9704 | 0.0978 | **2.37** | **0.017** |
| | LightGBM | 0.9731 | 0.9731 | 0.0898 | 12.13 | 0.302 |
| | CatBoost | 0.9759 | 0.9758 | 0.0876 | 11.71 | 0.020 |
| | **OQBoost** | **0.9782** | **0.9782** | **0.0778** | 25.16 | 0.900 |
| CoverType | — | — | — | — | — | — |
| Higgs | — | — | — | — | — | — |

> CoverType and Higgs results pending (large-scale runs). See `benchmark/results/` for live updates.

### Highlights

- **Best Balanced Accuracy (Adult Income):** OQBoost **0.844** vs XGBoost 0.798 (+5.8 pp)
- **Best Balanced Accuracy (Give Me Credit):** OQBoost **0.691** vs XGBoost 0.650 (+6.4 pp)
- **Best on Rotated Synthetic:** OQBoost **0.978** — highest balanced accuracy AND lowest log loss; validates oblique split advantage over axis-aligned methods
- **Credit Default:** OQBoost underperforms baselines here (highly imbalanced, 3% positive rate) — an open problem; tree depth / early stopping sensitivity under extreme imbalance

### Figure 1 — Balanced Accuracy Comparison

![Balanced Accuracy](benchmark/results/figures/fig2_balanced_accuracy.png)

### Figure 2 — Performance vs Training Cost

![Perf vs Cost](benchmark/results/figures/fig5_perf_vs_cost.png)

### Figure 3 — Rotation Robustness

![Rotation Robustness](benchmark/results/figures/fig6_rotation_robustness.png)

### Figure 4 — Missing Value Robustness

![Missing Value](benchmark/results/figures/fig7_missing_value.png)

### Figure 5 — Categorical Cardinality Robustness

![Categorical](benchmark/results/figures/fig8_categorical.png)

---

## Algorithm

Every node runs a full 256-bin axis scan, then challenges it with a pool of oblique candidate directions built from two complementary sources:

**Source 1 — GG-SRP (Gradient-Guided Sparse Random Projection)**  
Features are sampled with probability proportional to SIS gradient-importance scores. Each selected feature gets a weight sign aligned with the steepest gradient descent direction. No Gram matrix, no linear system — $O(D)$ per node. Mechanism studies show this diversity pool is the essential ingredient: it widens feature coverage, and its win share grows over boosting rounds as residuals become oblique.

**Source 2 — Direction Cache (dataset-level direction reuse)**  
A ring buffer keeps the most recent globally winning oblique directions (up to 32). Candidates blend a cached direction with the current parent direction at a random ratio. Good oblique directions are a property of the dataset's rotated subspaces, not of any single node — so reuse happens across trees, not from parent to child.

`inherited_rp_ratio` splits the candidate budget between the two sources (default 0.5). Earlier versions also mutated the parent node's own direction (Strategies "A/B"); controlled ablations showed parent-direction mutation consistently underperforms random replacement, and it was removed — see [`research/FINDINGS.md`](research/FINDINGS.md).

**Strategy ablation (synthetic 6k×30 / multiclass 10k×30, 5 seeds, accuracy):**

| Inherited-slot contents | Synthetic | Multiclass |
|--------------|---------|---------|
| Parent mutation only (removed A/B) | 0.9588–0.9620 | 0.8927–0.8986 |
| Global random only | 0.9643 | 0.9027 |
| Cache only | 0.9664 | 0.9002 |
| **Cache + random (current)** | **0.9729**¹ | **0.9192**¹ |

¹ current default `inherited_rp_ratio=0.5`, measured post-change.

See [`docs/algorithm.md`](docs/algorithm.md) for the full derivation and [`research/FINDINGS.md`](research/FINDINGS.md) for the mechanism studies.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 500 | Number of boosting rounds |
| `learning_rate` | 0.05 | Shrinkage per tree |
| `max_depth` | 6 | Leaf budget = 2^max_depth (64 leaves, matches XGBoost/CatBoost) |
| `reg_lambda` | 1.0 | L2 leaf regularization |
| `subsample` | 0.8 | Row fraction per tree |
| `early_stopping_rounds` | 50 | Stop if class-weighted val loss stagnates |
| `cat_features` | None | Categorical column names or indices |
| `class_weight` | "balanced" | Reweight by inverse class frequency |
| `inherited_rp_ratio` | 0.5 | Oblique-candidate split: cache-blend vs fresh random projections |
| `mutation_rate` | — | Deprecated, ignored (parent-mutation strategies removed) |
| `mutation_strength` | — | Deprecated, ignored (parent-mutation strategies removed) |
| `random_state` | None | Seed |
| `verbose` | False | Print per-round metrics |

---

## Running Benchmarks

```bash
cd benchmark

# Run all (takes ~hours for Higgs + CoverType)
python run_all.py

# Skip large datasets
python run_all.py --skip higgs covertype

# Individual benchmarks
python adult.py
python rotated_synthetic.py
python missing_value_robustness.py
python categorical_robustness.py

# Generate figures from completed results
python generate_figures.py

# Generate summary table (results/summary.md)
python generate_tables.py
```

Large datasets require manual download:
- **HIGGS**: https://archive.ics.uci.edu/dataset/280 → `benchmark/data/HIGGS.csv.gz`
- **Give Me Some Credit (Kaggle)**: https://www.kaggle.com/competitions/GiveMeSomeCredit → `benchmark/data/cs-training.csv`

---

## Repository Structure

```
oqboost/
├── src/oqboost/
│   ├── __init__.py
│   ├── _classifier.py      # OQBoostClassifier
│   ├── _oqboost.py        # C bindings + OQBoostTree, OQBoostContext
│   └── _ext/
│       ├── oqboost.cpp    # full engine: boosting round, routing, serialization
│       ├── oqboost_types.h # OQTree structure
│       ├── oqboost_core.h # Shared constants and helpers
│       └── liboqboost.dylib / .so / .dll  (compiled)
├── benchmark/
│   ├── *.py                # Per-dataset benchmark scripts
│   ├── _utils.py           # Shared train/eval utilities
│   ├── generate_figures.py
│   ├── generate_tables.py
│   ├── run_all.py
│   └── results/
│       ├── csv/            # Raw results
│       ├── figures/        # Generated plots
│       └── summary.md
├── docs/
│   ├── algorithm.md        # Theory and derivations
│   ├── api.md              # Full API reference
│   └── quickstart.md
├── tests/
└── pyproject.toml
```

---

## License

[MIT](LICENSE) — Copyright (c) 2025 cree1116
